# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

"""Frozen-VAE text-to-image training on top of a FROZEN FLUX.1 VAE (REPA).

Trains a fresh text-conditional MMDiT (REPA's SD3-style ``models/mmdit.py``) on
image+caption parquet data, encoding images on the fly with a **frozen** FLUX.1
autoencoder (16ch, f8) and captions with a frozen CLIP text encoder. This is the
REPA frozen-VAE stage adapted to FLUX + captions. It serves both arms of the
comparison, which differ only in which VAE is frozen:

  * end-to-end VAE: ``--vae-ckpt <extracted e2e-flux-vae>.pt`` plus its
    ``--latents-stats`` from the end-to-end stage;
  * baseline: ``--vae-ckpt <FLUX.1 vae dir>``, the original pretrained
    autoencoder, with FLUX's constant shift/scale.

Everything else is held fixed, so the comparison isolates the VAE.

Supports a multi-resolution schedule via ``--init-ckpt`` (weights-only warm
start; optimizer/step reset and ``pos_embed`` regenerated for the new grid):
  256x256 -> 512x512 -> an optional 512x512 finetune on higher-quality captions.
"""

import argparse
import copy
import json
import logging
import math
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from torchvision.utils import make_grid
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from tqdm.auto import tqdm
import wandb

from dataset import ParquetT2IDataset
from e2e_vae import load_e2e_vae, latents_stats_to_repa, FluxVAE, VAE_LATENT_CHANNELS
from loss import SILoss
from models.mmdit import MMDiT
from models.text_encoder import FrozenTextEncoder
from samplers_t2i import euler_sampler
from utils import load_encoders

logger = get_logger(__name__)

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)


def preprocess_raw_image(x, enc_type, resolution=256):
    if 'clip' in enc_type:
        x = x / 255.
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    elif 'dinov1' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224, mode='bicubic')
    return x


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")],
    )
    return logging.getLogger(__name__)


def build_latents_scale_bias(args, device, channels=16, vae=None):
    """Return (latents_scale, latents_bias) as (1,C,1,1) tensors in REPA's
    convention: normalized = z * scale + bias, decode = (z - bias) / scale."""
    if args.latents_stats and os.path.isfile(args.latents_stats):
        stats = torch.load(args.latents_stats, map_location="cpu", weights_only=False)
        scale_r, bias_r = latents_stats_to_repa(stats)
        scale_r = scale_r.view(1, channels, 1, 1).to(device)
        bias_r = bias_r.view(1, channels, 1, 1).to(device)
        return scale_r, bias_r
    if vae is not None and hasattr(vae, "initial_latents_stats"):
        scale_e, bias_e = vae.initial_latents_stats()
        scale_r = scale_e.view(1, channels, 1, 1).to(device)
        bias_r = (-bias_e * scale_e).view(1, channels, 1, 1).to(device)
        return scale_r, bias_r
    # Fallback: FLUX.1 constant scaling/shift (used for the original VAE).
    scale = float(FluxVAE.scaling_factor)
    shift = float(FluxVAE.shift_factor)
    latents_scale = torch.full((1, channels, 1, 1), scale, device=device)
    latents_bias = torch.full((1, channels, 1, 1), -shift * scale, device=device)
    return latents_scale, latents_bias


def load_warmstart(model, ema, ckpt_path):
    """Load model+ema weights from a previous phase, skipping any tensor whose
    shape differs from the current model (e.g. ``pos_embed`` on a resolution
    change). Optimizer/step are NOT restored."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    for tgt, key in ((model, "model"), (ema, "ema")):
        sd = ckpt.get(key, ckpt)
        cur = tgt.state_dict()
        filtered = {k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}
        skipped = [k for k in sd if k not in filtered]
        missing, unexpected = tgt.load_state_dict(filtered, strict=False)
        logger.info(f"warm-start [{key}]: loaded={len(filtered)} skipped(shape)={skipped} "
                    f"missing={len(missing)} unexpected={len(unexpected)}")


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None if args.report_to == "none" else args.report_to,
        project_config=accelerator_project_config,
    )
    use_wandb = accelerator.is_main_process and args.report_to == "wandb"

    save_dir = os.path.join(args.output_dir, args.exp_name)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)
        with open(os.path.join(save_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
        logger_local = create_logger(save_dir)
        logger_local.info(f"Experiment directory: {save_dir}")
    device = accelerator.device
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    assert args.resolution % 8 == 0
    latent_size = args.resolution // 8
    in_channels = VAE_LATENT_CHANNELS[args.vae_arch]

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # REPA visual encoders (dinov2) for the alignment loss. Skipped entirely when
    # REPA is disabled (--no-use-repa): no encoder is loaded and the MMDiT is built
    # without projection heads, so training is a plain diffusion loss.
    if args.use_repa:
        encoders, encoder_types, architectures = load_encoders(args.enc_type, device, args.resolution)
        z_dims = [enc.embed_dim for enc in encoders]
    else:
        encoders, encoder_types, architectures = [], [], []
        z_dims = []
        if accelerator.is_main_process:
            logger.info("REPA disabled (--no-use-repa): no alignment encoder / projection loss.")

    # Frozen text encoder (CLIP -> 768-dim context matches MMDiT context_embedder).
    text_encoder = FrozenTextEncoder(
        encoder_type=args.text_encoder, model_name=args.text_encoder_name,
        max_length=args.text_max_length, device=device,
    )

    # Frozen FLUX / Qwen-ViT VAE (original dir or extracted e2e .pt/.safetensors) + latent normalization.
    vae = load_e2e_vae(
        vae_arch=args.vae_arch,
        vae_ckpt=args.vae_ckpt,
        device=device,
        eval_mode=True,
        model_path=args.vae_vit_path,
        latent_channels=in_channels,
        decoder_width=args.vae_decoder_width,
        decoder_depth=args.vae_decoder_depth,
        decoder_heads=args.vae_decoder_heads,
        decoder_grid=16,
    )
    latents_scale, latents_bias = build_latents_scale_bias(args, device, channels=in_channels, vae=vae)

    # Fresh MMDiT (SD3-style, REPA). Defaults are MMDiT-XL/2 (hidden=1152,
    # depth=28, heads=16), the target model. The end-to-end stage uses a smaller
    # MMDiT-L/2, since that prior is discarded once the latent space is shaped.
    model = MMDiT(
        input_size=latent_size,
        in_channels=in_channels,
        depth=args.depth,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        z_dims=z_dims,
        encoder_depth=args.encoder_depth,
    ).to(device)
    ema = copy.deepcopy(model).to(device)
    requires_grad(ema, False)
    if accelerator.is_main_process:
        logger.info(f"MMDiT params: {sum(p.numel() for p in model.parameters()):,}")

    loss_fn = SILoss(
        prediction=args.prediction, path_type=args.path_type,
        encoders=encoders, accelerator=accelerator,
        latents_scale=latents_scale, latents_bias=latents_bias, weighting=args.weighting,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,
    )

    data_roots = [r for r in args.data_dir.split(",") if r]
    caption_columns = [c for c in args.caption_columns.split(",") if c]
    train_dataset = ParquetT2IDataset(
        data_roots, resolution=args.resolution,
        caption_columns=caption_columns, rows_per_file=args.parquet_rows_per_file,
    )
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset, batch_size=local_batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")

    update_ema(ema, model, decay=0)
    model.train(); ema.eval()

    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) + '.pt'
        ckpt = torch.load(f'{save_dir}/checkpoints/{ckpt_name}', map_location='cpu')
        model.load_state_dict(ckpt['model']); ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt']); global_step = ckpt['steps']
    elif args.init_ckpt:
        assert os.path.isfile(args.init_ckpt), f"--init-ckpt not found: {args.init_ckpt}"
        load_warmstart(model, ema, args.init_ckpt)
        if accelerator.is_main_process:
            logger.info(f"Warm-started weights from {args.init_ckpt} (opt/step reset)")

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"{args.exp_name}",
            dir=save_dir,
            config=vars(copy.deepcopy(args)),
            settings={"console": "off"},
        )

    # Fixed prompts for periodic qualitative samples.
    sample_batch_size = max(1, args.num_vis_samples // accelerator.num_processes)
    _prompts = [
        "a photograph of an astronaut riding a horse",
        "a cute corgi puppy in a field of flowers, bokeh",
        "an oil painting of a lighthouse during a storm",
        "a bowl of ramen with an egg, food photography",
    ]
    sample_prompts = [_prompts[i % len(_prompts)] for i in range(sample_batch_size)]
    with torch.no_grad():
        sample_context, _, _ = text_encoder.encode(sample_prompts)
        null_context_1, _, _ = text_encoder.encode([""])
    null_context = null_context_1.repeat(sample_batch_size, 1, 1)
    xT = torch.randn((sample_batch_size, in_channels, latent_size, latent_size), device=device)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=global_step,
                        desc="Steps", disable=not accelerator.is_local_main_process)

    for epoch in range(args.epochs):
        model.train()
        for raw_image, captions in train_dataloader:
            raw_image = raw_image.to(device)
            with torch.no_grad():
                if args.use_repa:
                    with accelerator.autocast():
                        zs = []
                        for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
                            raw_image_ = preprocess_raw_image(raw_image, encoder_type, resolution=args.resolution)
                            z = encoder.forward_features(raw_image_)
                            if 'mocov3' in encoder_type:
                                z = z[:, 1:]
                            if 'dinov2' in encoder_type:
                                z = z['x_norm_patchtokens']
                            zs.append(z)
                else:
                    zs = None
                # Frozen VAE encode (float32, no grad) + REPA-style normalization.
                img_vae = raw_image.float() / 127.5 - 1.0
                latent = vae.encode(img_vae).sample()
                x = latent * latents_scale + latents_bias
                # Text encode + classifier-free-guidance caption dropout.
                context, _, _ = text_encoder.encode(list(captions))
                if args.cfg_prob > 0:
                    drop = torch.rand(context.shape[0], device=context.device) < args.cfg_prob
                    if drop.any():
                        context[drop] = null_context_1.to(context.dtype)

            with accelerator.accumulate(model):
                model_kwargs = dict(context=context)
                loss, proj_loss = loss_fn(model, x, model_kwargs, zs=zs)
                loss_mean = loss.mean()
                proj_loss_mean = proj_loss.mean()
                total_loss = loss_mean + proj_loss_mean * args.proj_coeff
                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if accelerator.sync_gradients:
                    update_ema(ema, model)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                logs = {
                    "loss": accelerator.gather(loss_mean).mean().detach().item(),
                    "proj_loss": accelerator.gather(proj_loss_mean).mean().detach().item(),
                    "grad_norm": accelerator.gather(grad_norm).mean().detach().item(),
                }
                progress_bar.set_postfix(**logs)
                if use_wandb:
                    wandb.log(logs, step=global_step)

                if global_step % args.checkpointing_steps == 0 and global_step > 0:
                    if accelerator.is_main_process:
                        checkpoint = {
                            "model": accelerator.unwrap_model(model).state_dict(),
                            "ema": ema.state_dict(),
                            "opt": optimizer.state_dict(),
                            "args": args, "steps": global_step,
                        }
                        path = f"{save_dir}/checkpoints/{global_step:07d}.pt"
                        torch.save(checkpoint, path)
                        logger.info(f"Saved checkpoint to {path}")

                if global_step == 1 or (global_step % args.sampling_steps == 0 and global_step > 0):
                    model.eval()
                    with torch.no_grad():
                        samples = euler_sampler(
                            accelerator.unwrap_model(model), xT, sample_context,
                            y_null=null_context, num_steps=50, cfg_scale=args.sample_cfg_scale,
                            guidance_low=0., guidance_high=1., path_type=args.path_type, heun=False,
                        ).to(torch.float32)
                        img = vae.decode((samples - latents_bias) / latents_scale).sample
                        img = (img.float() + 1) / 2.
                    out_samples = accelerator.gather(img)
                    if accelerator.is_main_process and use_wandb:
                        wandb.log({"samples": wandb.Image(array2grid(out_samples))}, step=global_step)
                    model.train()

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        checkpoint = {
            "model": accelerator.unwrap_model(model).state_dict(),
            "ema": ema.state_dict(),
            "opt": optimizer.state_dict(),
            "args": args, "steps": global_step,
        }
        path = f"{save_dir}/checkpoints/{global_step:07d}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Saved final checkpoint to {path}")
    # accelerator.trackers only exists once init_trackers has run, which it
    # never does here: logging goes through wandb directly.
    if getattr(accelerator, "trackers", None):
        accelerator.end_training()


def parse_args(input_args=None):
    p = argparse.ArgumentParser(description="Frozen-VAE T2I training on FLUX VAE (REPA)")
    p.add_argument("--output-dir", type=str, default="exps")
    p.add_argument("--exp-name", type=str, required=True)
    p.add_argument("--logging-dir", type=str, default="logs")
    p.add_argument("--report-to", type=str, default="wandb", choices=["wandb", "none"])
    p.add_argument("--wandb-project", type=str, default="genfirst")
    p.add_argument("--resume-step", type=int, default=0)
    p.add_argument("--init-ckpt", type=str, default="",
                   help="Warm-start weights (model+ema) from a previous phase's .pt "
                        "(opt/step reset, pos_embed regenerated). Chains 256->512->ft.")

    # model (defaults are the MMDiT-XL/2 target: depth=28, hidden=1152, heads=16)
    p.add_argument("--depth", type=int, default=28, help="MMDiT number of joint blocks.")
    p.add_argument("--hidden-size", type=int, default=1152,
                   help="MMDiT width. If unset (None) REPA uses 32*depth.")
    p.add_argument("--num-heads", type=int, default=16,
                   help="MMDiT attention heads. If unset (None) REPA uses depth.")
    p.add_argument("--encoder-depth", type=int, default=8,
                   help="Which MMDiT joint block to align in REPA: the hidden state "
                        "after block #encoder_depth (1-indexed) is projected and matched "
                        "to the encoder features. Ignored when --no-use-repa.")

    # data
    p.add_argument("--data-dir", type=str, required=True,
                   help="Comma-separated parquet roots (image + caption).")
    p.add_argument("--caption-columns", type=str,
                   default="caption,prompt,text")
    p.add_argument("--parquet-rows-per-file", type=int, default=None)
    p.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)

    # text encoder
    p.add_argument("--text-encoder", type=str, default="clip", choices=["clip", "t5"])
    p.add_argument("--text-encoder-name", type=str, default=None)
    p.add_argument("--text-max-length", type=int, default=77)

    # vae (frozen)
    p.add_argument("--vae-arch", type=str, default="flux", choices=["flux", "flux2", "qwenvit"],
                   help="Frozen VAE architecture: flux (FLUX.1, 16ch f8), flux2 (FLUX.2, 32ch f8), or qwenvit (Qwen3-VL ViT AE, 16ch f8).")
    p.add_argument("--vae-ckpt", type=str, required=True,
                   help="Either an original FLUX.1/FLUX.2 diffusers `vae` directory, or an "
                        "end-to-end VAE .pt/.safetensors extracted from an end-to-end checkpoint.")
    p.add_argument("--vae-vit-path", type=str, default="Qwen/Qwen3-VL-2B-Instruct",
                   help="For --vae-arch qwenvit: Qwen3-VL model ID or path used to construct the vision tower.")
    p.add_argument("--vae-decoder-width", type=int, default=1024)
    p.add_argument("--vae-decoder-depth", type=int, default=12)
    p.add_argument("--vae-decoder-heads", type=int, default=16)
    p.add_argument("--latents-stats", type=str, default="",
                   help="Per-channel latent stats .pt, as written next to an extracted "
                        "end-to-end VAE. If unset, the VAE's own latent statistics are used.")

    # precision / opt
    p.add_argument("--allow-tf32", action="store_true")
    p.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--epochs", type=int, default=1400)
    p.add_argument("--max-train-steps", type=int, default=400000)
    p.add_argument("--checkpointing-steps", type=int, default=50000)
    p.add_argument("--sampling-steps", type=int, default=5000)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.999)
    p.add_argument("--adam-weight-decay", type=float, default=0.)
    p.add_argument("--adam-epsilon", type=float, default=1e-8)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)

    # loss / repa
    p.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    p.add_argument("--prediction", type=str, default="v", choices=["v"])
    p.add_argument("--weighting", type=str, default="uniform")
    p.add_argument("--cfg-prob", type=float, default=0.1)
    p.add_argument("--use-repa", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable REPA representation-alignment loss (default). Pass "
                        "--no-use-repa to disable it: no visual encoder is loaded, the "
                        "MMDiT is built without projection heads, and only the diffusion "
                        "loss is optimized.")
    p.add_argument("--enc-type", type=str, default="dinov2-vit-b",
                   help="REPA visual encoder(s) to align to, e.g. dinov2-vit-b.")
    p.add_argument("--proj-coeff", type=float, default=0.5,
                   help="Weight of the REPA alignment (projection) loss.")

    # sampling / logging
    p.add_argument("--num-vis-samples", type=int, default=8)
    p.add_argument("--sample-cfg-scale", type=float, default=4.0)

    return p.parse_args(input_args) if input_args is not None else p.parse_args()


if __name__ == "__main__":
    main(parse_args())
