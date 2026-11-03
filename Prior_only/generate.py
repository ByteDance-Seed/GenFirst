# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# That sentence refers to the tree this file came from, not to this one. It was
# vendored here through REPA (DiT -> SiT -> REPA), and DiT is CC BY-NC 4.0. See
# NOTICE for how it is licensed in this repository.

"""
Samples a large number of images from a pre-trained SiT model using DDP and writes
them as individual PNGs, which is what eval_fid.py scores with torch-fidelity.

Pass --weak-ckpt to enable autoguidance (Karras et al., "Guiding a Diffusion Model
with a Bad Version of Itself"): an earlier, deliberately under-trained checkpoint
of the same run supplies the guidance reference, composed with CFG. This is the
setting the paper's guided ImageNet-256 numbers use.

Class labels are drawn to be exactly balanced -- num_fid_samples/num_classes images
per class, 50 at the default 50k -- and shuffled across the run. This is the paper
protocol; --label-sampling=random restores i.i.d. uniform labels.

To just eyeball samples, run this with a small --num-fid-samples on one GPU. Small
counts do not divide evenly across the classes, so pass --label-sampling=random too.
"""
import torch
import torch.distributed as dist
from models.sit import SiT_models
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
from datetime import timedelta
from samplers import euler_sampler, euler_maruyama_sampler
from utils import load_legacy_checkpoints, download_model
from e2e_vae import load_e2e_vae, latents_stats_to_repa, vae_arch_spec

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def build_sit(args, ckpt_path, latent_size, latent_channels, device, block_kwargs):
    """Instantiate a SiT and load one checkpoint's EMA weights into it."""
    model = SiT_models[args.model](
        input_size=latent_size,
        in_channels=latent_channels,
        num_classes=args.num_classes,
        use_cfg=True,
        z_dims=[int(z_dim) for z_dim in args.projector_embed_dims.split(',')],
        encoder_depth=args.encoder_depth,
        **block_kwargs,
    ).to(device)
    if ckpt_path is None:
        assert args.model == 'SiT-XL/2'
        assert len(args.projector_embed_dims.split(',')) == 1
        assert int(args.projector_embed_dims.split(',')[0]) == 768
        state_dict = download_model('last.pt')
    else:
        state_dict = torch.load(
            ckpt_path, map_location=f'cuda:{device}', weights_only=False)['ema']
    if args.legacy:
        state_dict = load_legacy_checkpoints(
            state_dict=state_dict, encoder_depth=args.encoder_depth
            )
    model.load_state_dict(state_dict)
    model.eval()  # important!
    return model


def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP. The only collectives here are the label broadcast and the barrier
    # after the sampling loop, but ranks reach that barrier minutes apart: they do
    # equal work, and a 1% drift over a 50k run at 250 steps is enough for the
    # fastest rank to sit in the barrier past NCCL's 10-minute default and take the
    # whole job down at 99% complete. The timeout only bounds how long a rank waits
    # for its peers, so raising it costs nothing.
    dist.init_process_group("nccl", timeout=timedelta(hours=2))
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Load model:
    # Latent geometry depends on the VAE arch (must match training): SD / e2e-f8d4
    # downsample by 8 with 4 channels; e2e-f16d32 (VA-VAE) by 16 with 32 channels.
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    if args.vae_type == 'e2e':
        vae_downsample, latent_channels = vae_arch_spec(args.vae_arch)
    else:
        vae_downsample, latent_channels = 8, 4
    latent_size = args.resolution // vae_downsample
    # Auto-download a pre-trained model or load a custom SiT checkpoint from train_prior_only.py:
    if args.ckpt is None:
        args.ckpt = 'SiT-XL-2-256x256.pt'
        model = build_sit(args, None, latent_size, latent_channels, device, block_kwargs)
    else:
        model = build_sit(args, args.ckpt, latent_size, latent_channels, device, block_kwargs)

    # Autoguidance: an earlier checkpoint of the same run, used as the "bad" model
    # the guidance direction points away from. Same architecture, so it is built
    # with the same arguments.
    weak_model = None
    if args.weak_ckpt is not None:
        if args.autoguidance_scale == 1.0:
            raise ValueError(
                "--weak-ckpt was given but --autoguidance-scale is 1.0, which makes "
                "autoguidance a no-op. The paper uses 1.3."
                )
        weak_model = build_sit(
            args, args.weak_ckpt, latent_size, latent_channels, device, block_kwargs
            )
        if rank == 0:
            print(f"Autoguidance: weak model {args.weak_ckpt} at scale {args.autoguidance_scale}")
    elif args.autoguidance_scale != 1.0:
        raise ValueError("--autoguidance-scale requires --weak-ckpt")

    # Load the VAE used to decode latents. Must match the VAE used in training,
    # along with its latent normalization stats.
    if args.vae_type == 'e2e':
        assert args.vae_ckpt is not None and args.latents_stats is not None, \
            "--vae-ckpt and --latents-stats are required when --vae-type=e2e"
        vae = load_e2e_vae(args.vae_arch, args.vae_ckpt, device=device)
        latents_stats = torch.load(
            args.latents_stats, map_location='cpu', weights_only=False)
        latents_scale, latents_bias = latents_stats_to_repa(latents_stats)
        latents_scale = latents_scale.to(device)
        latents_bias = latents_bias.to(device)
    else:
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
        latents_scale = torch.tensor(
            [0.18215, 0.18215, 0.18215, 0.18215]
            ).view(1, 4, 1, 1).to(device)
        latents_bias = torch.tensor(
            [0., 0., 0., 0.]
            ).view(1, 4, 1, 1).to(device)
    vae.eval()
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    vae_tag = args.vae_type if args.vae_type == 'e2e' else args.vae
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.resolution}-vae-{vae_tag}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}-{args.mode}-" \
                  f"labels-{args.label_sampling}"
    if weak_model is not None:
        # The interval is part of the setting, so keep it out of the folder name
        # only when it is the full range.
        folder_name += f"-ag-{args.autoguidance_scale}"
        if (args.ag_low, args.ag_high) != (0., 1.):
            folder_name += f"-agint-{args.ag_low}-{args.ag_high}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    world_size = dist.get_world_size()
    if rank == 0:
        print(f"Total number of images that will be sampled: {args.num_fid_samples}")
        print(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"projector Parameters: {sum(p.numel() for p in model.projectors.parameters()):,}")

    # Exact class balance: 50_000 images, 1_000 classes => 50 per class
    assert args.label_sampling != "equal" or args.num_fid_samples % args.num_classes == 0, \
        f"--label-sampling=equal needs num_fid_samples ({args.num_fid_samples}) divisible by " \
        f"num_classes ({args.num_classes}). Pass --label-sampling=random for small runs; " \
        f"note that scoring FID on those is not the paper protocol."
    per_class = args.num_fid_samples // args.num_classes  # 50 when 50k/1k

    # Build a global label schedule with exact counts, then shuffle it. The shuffle
    # keeps a class out of any single batch's guidance/BN statistics while leaving
    # the marginal exactly uniform.
    # IMPORTANT: all ranks must see the same permutation => build on rank 0 and broadcast.
    if rank == 0:
        if args.label_sampling == "equal":
            y_all = torch.arange(args.num_classes, device=device).repeat_interleave(per_class)  # [0..999] each repeated 50x
            gen = torch.Generator(device=device).manual_seed(args.global_seed)
            y_all = y_all[torch.randperm(y_all.numel(), generator=gen, device=device)]
        elif args.label_sampling == "random":
            y_all = torch.randint(0, args.num_classes, (args.num_fid_samples,), device=device)
        else:
            raise NotImplementedError(f"Unknown label_sampling: {args.label_sampling}")
    else:
        y_all = torch.empty(args.num_fid_samples, device=device, dtype=torch.long)
    dist.broadcast(y_all, src=0)

    # Equal contiguous shard per rank, so the written indices cover 0..num_fid_samples-1 exactly once.
    assert args.num_fid_samples % world_size == 0, \
        f"num_fid_samples ({args.num_fid_samples}) must be divisible by world_size ({world_size})."
    labels_per_rank = args.num_fid_samples // world_size
    start = rank * labels_per_rank
    y_this_rank = y_all[start : start + labels_per_rank]

    # The last batch may be partial; no need to force divisibility by n.
    total_to_make = y_this_rank.numel()
    iterations = int(math.ceil(total_to_make / n))
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    offset = 0  # offset within this rank's shard
    for _ in pbar:
        m = min(n, total_to_make - offset)

        # Sample inputs:
        z = torch.randn(m, model.in_channels, latent_size, latent_size, device=device)
        y = y_this_rank[offset : offset + m]

        # Sample images:
        sampling_kwargs = dict(
            model=model, 
            latents=z,
            y=y,
            num_steps=args.num_steps, 
            heun=args.heun,
            cfg_scale=args.cfg_scale,
            guidance_low=args.guidance_low,
            guidance_high=args.guidance_high,
            path_type=args.path_type,
            weak_model=weak_model,
            autoguidance_scale=args.autoguidance_scale,
            ag_low=args.ag_low,
            ag_high=args.ag_high,
        )
        with torch.no_grad():
            if args.mode == "sde":
                samples = euler_maruyama_sampler(**sampling_kwargs).to(torch.float32)
            elif args.mode == "ode":
                samples = euler_sampler(**sampling_kwargs).to(torch.float32)
            else:
                raise NotImplementedError()

            samples = vae.decode((samples - latents_bias) / latents_scale).sample
            samples = (samples + 1) / 2.
            samples = torch.clamp(
                255. * samples, 0, 255
                ).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            # Save samples to disk as individual .png files
            for i, sample in enumerate(samples):
                index = start + offset + i
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        offset += m

    # Make sure all processes have finished saving their samples before scoring.
    dist.barrier()
    if rank == 0:
        # eval_fid.py reads the PNG folder directly, so the .npz is only built on
        # request -- it is a second full copy of 50k images on disk and in memory.
        if args.build_npz:
            create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print(f"Done. Samples: {sample_folder_dir}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # seed
    parser.add_argument("--global-seed", type=int, default=0)

    # precision
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")

    # logging/saving:
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a SiT checkpoint.")
    parser.add_argument("--sample-dir", type=str, default="samples")

    # model
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)

    # vae
    parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--vae-type", type=str, default="sd", choices=["sd", "e2e"],
                        help="'sd' = stabilityai/sd-vae-ft-*; 'e2e' = VAE extracted from an end-to-end checkpoint.")
    parser.add_argument("--vae-arch", type=str, default="f8d4",
                        help="End-to-end VAE architecture (used when --vae-type=e2e).")
    parser.add_argument("--vae-ckpt", type=str, default=None,
                        help="Path to extracted end-to-end VAE state_dict (used when --vae-type=e2e).")
    parser.add_argument("--latents-stats", type=str, default=None,
                        help="Path to extracted end-to-end latents-stats .pt (used when --vae-type=e2e).")

    # number of samples
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--label-sampling", type=str, choices=["equal", "random"], default="equal",
                        help="How to draw the class labels. 'equal' gives every class exactly "
                             "num_fid_samples/num_classes images (50 at 50k), shuffled across the "
                             "run; 'random' draws them i.i.d. uniform. The paper uses 'equal'.")

    # sampling related hyperparameters
    parser.add_argument("--mode", type=str, default="ode")
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--projector-embed-dims", type=str, default="768,1024")
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--heun", action=argparse.BooleanOptionalAction, default=False) # only for ode
    parser.add_argument("--guidance-low", type=float, default=0.)
    parser.add_argument("--guidance-high", type=float, default=1.)
    parser.add_argument("--weak-ckpt", type=str, default=None,
                        help="Earlier checkpoint of the same run, used as the autoguidance "
                             "reference. The paper uses the 20-epoch checkpoint.")
    parser.add_argument("--autoguidance-scale", type=float, default=1.0,
                        help="Autoguidance scale; 1.0 disables it. The paper uses 1.3. "
                             "Applied inside each CFG branch, so it composes with --cfg-scale.")
    parser.add_argument("--ag-low", type=float, default=0.,
                        help="Autoguidance is gated on its own time interval, separate "
                             "from --guidance-low/--guidance-high. The paper's runs leave "
                             "it at the full [0, 1] while restricting CFG to [0.2, 0.7], "
                             "so autoguidance stays on where the weak model differs most.")
    parser.add_argument("--ag-high", type=float, default=1.)

    # output
    parser.add_argument("--build-npz", action=argparse.BooleanOptionalAction, default=False,
                        help="Also pack the PNGs into an .npz (for the ADM evaluator). "
                             "eval_fid.py does not need it.")

    # will be deprecated
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False) # only for ode


    args = parser.parse_args()
    main(args)
