# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

"""Frozen-VAE joint image+text training: the prior-only stage of the joint pipeline.

This is to ``train_joint_e2e.py`` what ``train_t2i_frozen.py`` is to
``train_e2e.py``. The end-to-end stage exists to *reshape a VAE* and throws its
DiT away; this stage freezes both VAEs and trains the model that is actually
evaluated, on the same three-phase schedule as the image runs:

    phase 1  256x256, 200k steps, lr 1e-4
    phase 2  512x512,  40k steps, lr 1e-4, warm-started from phase 1
    phase 3  512x512,  20k steps, lr 5e-5, SFT captions, from phase 2

Two runs differ only in where the text VAE comes from, which isolates the text
VAE's contribution the same way the image-only ablation isolates the image VAE:

  * ``--text-vae-ckpt <extracted e2e text VAE>``: the text VAE that the joint
    end-to-end stage reshaped.
  * no ``--text-vae-ckpt``: Cola's pretrained text VAE, untouched.

One block-causal joint MMDiT models both directions. A direction is drawn per
*sample* with probability ``--t2i-prob``, so every step carries both. The
objective is each sample's flow-matching loss in its own direction, plus REPA
alignment on the image stream::

    total = denoising_loss + proj_coeff * proj_loss

Alignment only applies to ``t2i`` samples, and the frozen encoder runs on just
that subset. For ``i2t`` the image latent is a clean condition, and matching
DINOv2 features from a clean latent is not the REPA objective. Classifier-free
guidance comes from the model's own condition dropout (``--cond-dropout``), not
from a null token.
"""

import argparse
import copy
import json
import logging
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from torchvision.utils import make_grid
from tqdm.auto import tqdm

# Every joint module -- the MMDiT, the text VAE, the caption collator, the
# samplers -- lives at the repo root, so the root goes first on the path:
# `dataset`, `models` and `utils` must resolve there rather than to the
# same-named copies in this directory.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys.path and sys.path[0] != _ROOT:
    sys.path.insert(0, _ROOT)

from dataset import CaptionCollator, ParquetT2IDataset  # noqa: E402
from models.autoencoder import FluxVAE  # noqa: E402
from models.joint_mmdit import JointMMDiT_models  # noqa: E402
from models.text_vae import ColaTextVAE, na_to_padded  # noqa: E402
from samplers_joint import decode_text_ids, i2t_sample, ids_to_text, t2i_sample  # noqa: E402
from utils import count_trainable_params, load_encoders, preprocess_imgs_vae  # noqa: E402

logger = logging.getLogger(__name__)

VIS_PROMPTS = [
    "a photograph of an astronaut riding a horse",
    "a cute corgi puppy in a field of flowers, bokeh",
    "an oil painting of a lighthouse during a storm",
    "a bowl of ramen with an egg, food photography",
]


def preprocess_raw_image(x, enc_type, resolution=256):
    """Match ``train_t2i_frozen.py`` so alignment targets are identical."""
    if "clip" in enc_type:
        x = x / 255.0
        x = F.interpolate(x, 224 * (resolution // 256), mode="bicubic")
        x = Normalize((0.48145466, 0.4578275, 0.40821073),
                      (0.26862954, 0.26130258, 0.27577711))(x)
    elif "mocov3" in enc_type or "mae" in enc_type:
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x / 255.0)
    elif "dinov2" in enc_type:
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x / 255.0)
        x = F.interpolate(x, 224 * (resolution // 256), mode="bicubic")
    elif "dinov1" in enc_type:
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x / 255.0)
    elif "jepa" in enc_type:
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x / 255.0)
        x = F.interpolate(x, 224, mode="bicubic")
    return x


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    return x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()


def requires_grad(model, flag):
    for p in model.parameters():
        p.requires_grad_(flag)


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = dict(ema_model.named_parameters())
    for name, param in model.named_parameters():
        ema_params[name].mul_(decay).add_(param.detach(), alpha=1 - decay)
    ema_buffers = dict(ema_model.named_buffers())
    for name, buf in model.named_buffers():
        ema_buffers[name].copy_(buf)


def load_warmstart(model, ema, ckpt_path):
    """Weights-only warm start from the previous stage.

    Tensors whose shape changed are skipped: at 512 the image position embedding
    covers four times as many patches, and ``initialize_weights`` has already
    filled it with the sincos grid for the new resolution. Optimizer and step
    are not restored.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    for tgt, key in ((model, "model"), (ema, "ema")):
        sd = ckpt.get(key, ckpt)
        cur = tgt.state_dict()
        filtered = {k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}
        skipped = [k for k in sd if k not in filtered]
        missing, unexpected = tgt.load_state_dict(filtered, strict=False)
        logger.info(
            f"warm-start [{key}]: loaded={len(filtered)} skipped(shape)={skipped} "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )


def build_frozen_image_vae(args, device):
    """Frozen image VAE plus the per-channel statistics the DiT normalizes with.

    Either an extracted end-to-end checkpoint (which ships measured statistics)
    or the pretrained FLUX folder, whose only statistics are the scalar
    ``shift``/``scale`` from its config -- the same fallback the image-only
    baseline (run ③) trained on.
    """
    vae = FluxVAE(pretrained_path=args.image_vae_pretrained)
    if args.image_vae_ckpt:
        sd = torch.load(args.image_vae_ckpt, map_location="cpu")
        if isinstance(sd, dict) and "vae" in sd and "vae.encoder.conv_in.weight" not in sd:
            sd = sd["vae"]
        missing, unexpected = vae.load_state_dict(sd, strict=False)
        if missing or unexpected:
            logger.warning(f"image VAE load: missing={len(missing)} unexpected={len(unexpected)}")
    vae = vae.to(device).eval()
    requires_grad(vae, False)

    if args.image_latents_stats:
        stats = torch.load(args.image_latents_stats, map_location="cpu")
        scale = stats["latents_scale"].squeeze().to(device)
        bias = stats["latents_bias"].squeeze().to(device)
    else:
        n = vae.embed_dim
        scale = torch.full((n,), float(FluxVAE.scaling_factor), device=device)
        bias = torch.full((n,), float(FluxVAE.shift_factor), device=device)
    return vae, scale, bias


def build_frozen_text_vae(args, device):
    """Cola's text VAE, optionally with weights from the end-to-end stage."""
    vae = ColaTextVAE(
        pretrained_path=args.text_vae_path, train_trunk=False, train_embeddings=False
    )
    source = "cola-pretrained"
    if args.text_vae_ckpt:
        sd = torch.load(args.text_vae_ckpt, map_location="cpu")
        if isinstance(sd, dict) and "text_vae" in sd:
            # A joint end-to-end checkpoint, used directly instead of extracting.
            sd = sd["text_vae"]
        # Strict: a silently unloaded text VAE would turn this arm into the
        # baseline and the comparison would be meaningless.
        vae.load_state_dict(sd)
        source = args.text_vae_ckpt
    vae = vae.to(device).eval()
    requires_grad(vae, False)
    return vae, source


def alignment_loss(zs, zs_tilde):
    """REPA alignment: per-sample negative cosine similarity, averaged.

    Same reduction as ``Prior_only/loss.py``: cosine over channels, mean over tokens,
    mean over encoders.
    """
    total = 0.0
    for z, z_tilde in zip(zs, zs_tilde):
        z = F.normalize(z.float(), dim=-1)
        z_tilde = F.normalize(z_tilde.float(), dim=-1)
        total = total - (z * z_tilde).sum(dim=-1).mean(dim=-1)
    return total / max(len(zs), 1)


@torch.no_grad()
def sample_t2i(model, image_vae, text_vae, collator, args, device, n):
    """Images for ``n`` fixed prompts, as ``(n, 3, H, W)`` in [0, 1]."""
    prompts = [VIS_PROMPTS[i % len(VIS_PROMPTS)] for i in range(n)]
    seqs = [collator.encode_one(p) for p in prompts]
    L = max(len(s) for s in seqs)
    text_ids = [torch.tensor(s, dtype=torch.long, device=device) for s in seqs]
    valid = torch.zeros(n, L, dtype=torch.bool, device=device)
    for i, s in enumerate(seqs):
        valid[i, : len(s)] = True

    posterior, txt_shape = text_vae.encode(text_ids)
    z_txt = na_to_padded(posterior.mode().float(), txt_shape, L)
    z_img = t2i_sample(
        model, z_txt, valid, num_steps=args.t2i_sample_steps,
        cfg_scale=args.sample_cfg_scale,
    )
    img = image_vae.decode(z_img.float()).sample.float()
    return (img + 1) / 2


@torch.no_grad()
def sample_i2t(model, image_vae, text_vae, collator, args, images):
    """Captions generated from real images, against the ground truth."""
    z_img = image_vae.encode(preprocess_imgs_vae(images)).mode().float()
    z_txt, valid = i2t_sample(
        model, z_img, args.i2t_vis_blocks, num_steps=args.i2t_sample_steps,
        cfg_scale=args.sample_i2t_cfg_scale,
        eos_id=collator.eos_id, text_vae=text_vae,
    )
    ids = decode_text_ids(text_vae, z_txt)
    return ids_to_text(collator.tok, ids, valid, eos_id=collator.eos_id)


def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    device = accelerator.device
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    save_dir = os.path.join(args.output_dir, args.exp_name)
    if accelerator.is_main_process:
        os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{save_dir}/log.txt")],
        )

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    assert args.resolution % 8 == 0
    latent_size = args.resolution // 8

    if args.use_repa:
        encoders, encoder_types, architectures = load_encoders(
            args.enc_type, device, args.resolution
        )
        z_dims = [enc.embed_dim for enc in encoders]
    else:
        encoders, encoder_types, architectures, z_dims = [], [], [], []

    image_vae, latents_scale, latents_bias = build_frozen_image_vae(args, device)
    text_vae, text_vae_source = build_frozen_text_vae(args, device)

    # Recorded so `sample_joint.py` and `generate.py` can rebuild the model with
    # matching projection heads from the checkpoint's args alone.
    args.z_dims = list(z_dims)

    model = JointMMDiT_models[args.model](
        input_size=latent_size,
        image_channels=16,
        text_channels=text_vae.latent_dim,
        text_block_size=args.text_block_size,
        max_text_tokens=args.max_text_tokens,
        path_type=args.path_type,
        cond_dropout=args.cond_dropout,
        z_dims=z_dims,
        encoder_depth=args.encoder_depth,
    ).to(device)
    model.init_bn(latents_scale=latents_scale, latents_bias=latents_bias)

    ema = copy.deepcopy(model).to(device)
    requires_grad(ema, False)
    ema.eval()

    if accelerator.is_main_process:
        with open(os.path.join(save_dir, "args.json"), "w") as f:
            json.dump({k: v for k, v in vars(args).items()}, f, indent=4, default=str)
        logger.info(f"joint MMDiT params: {count_trainable_params(model):,}")
        logger.info(f"text VAE (frozen): {text_vae_source}")
        logger.info(
            f"image VAE (frozen): ckpt={args.image_vae_ckpt or args.image_vae_pretrained}, "
            f"stats={args.image_latents_stats or 'FLUX scalar shift/scale'}"
        )
        logger.info(
            f"REPA: on={args.use_repa} enc={args.enc_type} z_dims={z_dims} "
            f"layer={args.encoder_depth} coeff={args.proj_coeff} (t2i steps only)"
        )
        logger.info(
            f"t2i_prob={args.t2i_prob} cond_dropout={args.cond_dropout} "
            f"text_latent_mode={args.text_latent_mode}"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,
    )

    dataset = ParquetT2IDataset(
        [r for r in args.data_dir.split(",") if r],
        resolution=args.resolution,
        caption_columns=[c for c in args.caption_columns.split(",") if c],
        rows_per_file=args.parquet_rows_per_file,
        random_caption=args.random_caption,
    )
    collator = CaptionCollator(
        args.tokenizer_path, block_size=args.text_block_size, max_tokens=args.max_text_tokens
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size // accelerator.num_processes),
        shuffle=True, num_workers=args.num_workers, pin_memory=True,
        drop_last=True, collate_fn=collator,
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} images ({args.data_dir})")

    update_ema(ema, model, decay=0)
    model.train()

    global_step = 0
    if args.resume_step > 0:
        ckpt = torch.load(f"{save_dir}/checkpoints/{args.resume_step:07d}.pt", map_location="cpu")
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["opt"])
        global_step = ckpt["steps"]
        if accelerator.is_main_process:
            logger.info(f"resumed {args.exp_name} at step {global_step}")
    elif args.init_ckpt:
        assert os.path.isfile(args.init_ckpt), f"--init-ckpt not found: {args.init_ckpt}"
        load_warmstart(model, ema, args.init_ckpt)

    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    use_wandb = accelerator.is_main_process and args.report_to == "wandb"
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.exp_name, dir=save_dir,
                   config=vars(copy.deepcopy(args)), settings={"console": "off"})

    def save_checkpoint(step):
        state = {
            "model": accelerator.unwrap_model(model).state_dict(),
            "ema": ema.state_dict(),
            "opt": optimizer.state_dict(),
            # Frozen, but carried along so a checkpoint is self-contained for
            # sampling and so the arm is identifiable after the fact.
            "text_vae": text_vae.state_dict(),
            "args": args,
            "steps": step,
        }
        torch.save(state, f"{save_dir}/checkpoints/{step:07d}.pt")

    progress = tqdm(range(args.max_train_steps), initial=global_step, desc="Steps",
                    disable=not accelerator.is_local_main_process)

    # Directions are drawn per sample, so each rank varies its own mix.
    task_gen = torch.Generator(device="cpu")
    task_gen.manual_seed(
        (args.seed if args.seed is not None else 0) + accelerator.process_index
    )

    n_vis = max(1, args.num_vis_samples // accelerator.num_processes)
    done = False
    for epoch in range(args.epochs):
        if done:
            break
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            text_ids = [t.to(device, non_blocking=True) for t in batch["text_ids"]]
            text_valid = batch["text_valid"].to(device, non_blocking=True)
            L_pad = text_valid.shape[1]
            # Direction per sample, so both are in every step.
            is_t2i = (
                torch.rand(text_valid.shape[0], generator=task_gen) < args.t2i_prob
            ).to(device)

            with torch.no_grad():
                # Alignment targets only exist for t2i samples, so the encoder
                # runs on that subset rather than the whole batch: same average
                # cost as when a whole step was one direction.
                zs = None
                if args.use_repa and bool(is_t2i.any()):
                    with accelerator.autocast():
                        zs = []
                        for encoder, enc_type, _arch in zip(
                            encoders, encoder_types, architectures
                        ):
                            raw = preprocess_raw_image(
                                images[is_t2i].float(), enc_type,
                                resolution=args.resolution,
                            )
                            z = encoder.forward_features(raw)
                            if "mocov3" in enc_type:
                                z = z[:, 1:]
                            if "dinov2" in enc_type:
                                z = z["x_norm_patchtokens"]
                            zs.append(z)

                with accelerator.autocast():
                    z_img = image_vae.encode(preprocess_imgs_vae(images)).sample()
                    posterior, txt_shape = text_vae.encode(text_ids)
                    z_txt_na = (
                        posterior.sample()
                        if args.text_latent_mode == "sample"
                        else posterior.mode()
                    )
                z_txt = na_to_padded(z_txt_na.float(), txt_shape, L_pad)

            with accelerator.accumulate(model), accelerator.autocast():
                out = model(z_img.float(), z_txt, text_valid, is_t2i)
                denoising_loss = out["prior_loss"].mean()

                if not out["zs"]:
                    proj_loss = torch.zeros((), device=device)
                elif zs is not None:
                    assert zs[0].shape[1] == out["zs"][0].shape[1], (
                        f"encoder gives {zs[0].shape[1]} tokens but the image stream has "
                        f"{out['zs'][0].shape[1]}"
                    )
                    proj_loss = alignment_loss(
                        zs, [z[is_t2i] for z in out["zs"]]
                    ).mean()
                else:
                    # Zero coefficient, not a skipped forward: the heads have to
                    # stay in the graph or DDP reports them unused on the rare
                    # batch that drew no t2i sample at all.
                    proj_loss = 0.0 * torch.stack(
                        [z.float().mean() for z in out["zs"]]
                    ).sum()

                loss = denoising_loss + args.proj_coeff * proj_loss
                accelerator.backward(loss)
                grad_norm = None
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if accelerator.sync_gradients:
                    update_ema(ema, accelerator.unwrap_model(model), decay=args.ema_decay)

            if not accelerator.sync_gradients:
                continue

            progress.update(1)
            global_step += 1
            # The two directions differ ~4x in scale, so a single mean would hide
            # either one moving; each is averaged over its own samples.
            prior_all = accelerator.gather(out["prior_loss"].detach())
            t2i_all = accelerator.gather(is_t2i)
            logs = {
                "loss": accelerator.gather(loss.detach()).mean().item(),
                "t2i_frac": t2i_all.float().mean().item(),
                "text_len": float(L_pad),
                "epoch": epoch,
            }
            for name, sel in (("t2i", t2i_all), ("i2t", ~t2i_all)):
                if sel.any():
                    logs[f"denoising_loss/{name}"] = prior_all[sel].mean().item()
            if args.use_repa:
                # Every rank has to reach this gather, including one whose batch drew
                # no t2i sample at all -- gathering under a rank-local condition
                # mismatches the collective and hangs the run. Weighting each rank's
                # mean by its own t2i count makes the ratio the cosine averaged over
                # every t2i sample there is.
                proj_sum = accelerator.gather(proj_loss.detach() * is_t2i.sum()).sum()
                if t2i_all.any():
                    logs["proj_loss"] = (proj_sum / t2i_all.sum()).item()
            if grad_norm is not None:
                logs["grad_norm"] = accelerator.gather(grad_norm).mean().item()
            progress.set_postfix(loss=logs["loss"])
            if accelerator.is_main_process:
                if use_wandb:
                    wandb.log(logs, step=global_step)
                if global_step % args.log_every == 0 or global_step == 1:
                    logger.info(
                        f"step {global_step:>7} loss {logs['loss']:.4f} "
                        f"denoise t2i {logs.get('denoising_loss/t2i', float('nan')):.4f} "
                        f"i2t {logs.get('denoising_loss/i2t', float('nan')):.4f}"
                        + (f" proj {logs['proj_loss']:+.4f}" if "proj_loss" in logs else "")
                    )

            if global_step == 1 or global_step % args.sampling_steps == 0:
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.eval()
                with accelerator.autocast():
                    imgs = sample_t2i(
                        unwrapped, image_vae, text_vae, collator, args, device, n_vis
                    )
                gathered = accelerator.gather(imgs)
                # These samples exist only to be logged, so skip them without wandb.
                if use_wandb:
                    with accelerator.autocast():
                        caps = sample_i2t(
                            unwrapped, image_vae, text_vae, collator, args,
                            images[: args.num_vis_samples],
                        )
                    table = wandb.Table(columns=["ground truth", "generated"])
                    for gt, gen in zip(batch["captions"], caps):
                        table.add_data(gt, gen)
                    wandb.log(
                        {"samples/t2i": wandb.Image(array2grid(gathered)),
                         "samples/i2t": table},
                        step=global_step,
                    )
                unwrapped.train()

            if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                save_checkpoint(global_step)
                logger.info(f"Saved checkpoint at step {global_step}")

            if global_step >= args.max_train_steps:
                done = True
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_checkpoint(global_step)
        logger.info(f"Saved final checkpoint at step {global_step}")


def parse_args():
    p = argparse.ArgumentParser(description="Frozen-VAE joint image+text training (REPA stage)")
    p.add_argument("--exp-name", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="exps")
    p.add_argument("--report-to", type=str, default="wandb", choices=["wandb", "none"])
    p.add_argument("--wandb-project", type=str, default="genfirst")
    p.add_argument("--model", type=str, default="JointMMDiT-XL/2", choices=list(JointMMDiT_models))
    p.add_argument("--resume-step", type=int, default=0)
    p.add_argument("--init-ckpt", type=str, default="",
                   help="warm-start model+ema from the previous stage (opt/step reset, "
                        "image pos_embed regenerated); chains 256 -> 512 -> SFT")

    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--caption-columns", type=str, default="caption,prompt,text",
                   help="Comma-separated candidate caption columns in the parquet shards.")
    p.add_argument("--random-caption", action="store_true")
    p.add_argument("--parquet-rows-per-file", type=int, default=128)
    p.add_argument("--resolution", type=int, default=256, choices=[256, 512])
    p.add_argument("--num-workers", type=int, default=8)

    p.add_argument("--image-vae-ckpt", type=str, default=None,
                   help="extracted e2e image VAE .pt; omit to use --image-vae-pretrained")
    p.add_argument("--image-vae-pretrained", type=str, default=None,
                   help="pretrained FLUX `vae` folder (diffusers layout)")
    p.add_argument("--image-latents-stats", type=str, default=None,
                   help="measured per-channel stats; omit to use FLUX's scalar shift/scale, "
                        "as the image-only baseline did")
    p.add_argument("--text-vae-path", type=str, required=True,
                   help="Cola text VAE directory (config + weights); the architecture, and "
                        "the weights themselves unless --text-vae-ckpt is given")
    p.add_argument("--text-vae-ckpt", type=str, default=None,
                   help="text VAE reshaped by the joint end-to-end stage: either an "
                        "extracted .pt or a joint checkpoint with a `text_vae` key")
    p.add_argument("--tokenizer-path", type=str, required=True)
    p.add_argument("--text-latent-mode", type=str, default="sample", choices=["sample", "mode"],
                   help="posterior sample (as in the end-to-end stage) or its mean")

    p.add_argument("--text-block-size", type=int, default=16)
    p.add_argument("--max-text-tokens", type=int, default=256)
    p.add_argument("--t2i-prob", type=float, default=0.5)
    p.add_argument("--cond-dropout", type=float, default=0.1,
                   help="probability of hiding the condition modality, which trains the "
                        "unconditional branch classifier-free guidance needs")
    p.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])

    p.add_argument("--use-repa", action=argparse.BooleanOptionalAction, default=True,
                   help="align the image stream to a frozen visual encoder; --no-use-repa "
                        "builds the model without projection heads")
    p.add_argument("--enc-type", type=str, default="dinov2-vit-b")
    p.add_argument("--encoder-depth", type=int, default=8,
                   help="joint block whose image hidden state is aligned (1-indexed)")
    p.add_argument("--proj-coeff", type=float, default=0.5)

    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.999)
    p.add_argument("--adam-weight-decay", type=float, default=0.0)
    p.add_argument("--adam-epsilon", type=float, default=1e-8)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.9999)

    p.add_argument("--epochs", type=int, default=1400)
    p.add_argument("--max-train-steps", type=int, default=200000)
    p.add_argument("--checkpointing-steps", type=int, default=20000)
    p.add_argument("--sampling-steps", type=int, default=5000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--num-vis-samples", type=int, default=8)
    p.add_argument("--t2i-sample-steps", type=int, default=50)
    p.add_argument("--i2t-sample-steps", type=int, default=20)
    p.add_argument("--i2t-vis-blocks", type=int, default=4,
                   help="text blocks to generate for the i2t sample log")
    p.add_argument("--sample-cfg-scale", type=float, default=4.0)
    p.add_argument("--sample-i2t-cfg-scale", type=float, default=1.5)

    p.add_argument("--mixed-precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--allow-tf32", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not args.image_vae_ckpt and not args.image_vae_pretrained:
        p.error("need --image-vae-ckpt or --image-vae-pretrained")
    if args.cond_dropout <= 0:
        # Guidance needs the text-blind / image-blind branch to have been trained.
        args.sample_cfg_scale = 1.0
        args.sample_i2t_cfg_scale = 1.0
    return args


if __name__ == "__main__":
    main(parse_args())
