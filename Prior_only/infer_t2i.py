# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

"""Standalone text-to-image inference for the frozen-VAE MMDiT (REPA).

Loads a checkpoint saved by ``train_t2i_frozen.py`` and generates images from
text prompts, using the SAME MMDiT / FLUX VAE / CLIP / euler-sampler as training.
Model/VAE config is auto-restored from ``ckpt["args"]`` so you only need to pass
the checkpoint path and prompts.

Example (single GPU):
  python infer_t2i.py \
    --ckpt exps/t2i-frozen-e2e-flux-vae-mmdit-phase1-256/checkpoints/0200000.pt \
    --out  exps/t2i-frozen-e2e-flux-vae-mmdit-phase1-256/infer \
    --prompts "a photograph of an astronaut riding a horse||a cute corgi puppy in a field of flowers, bokeh" \
    --cfg-scale 4.0 --num-steps 50 --seed 0
"""

import argparse
import math
import os

import torch
from torchvision.utils import make_grid, save_image

from e2e_vae import load_e2e_vae, latents_stats_to_repa, FluxVAE, VAE_LATENT_CHANNELS
from models.mmdit import MMDiT
from models.text_encoder import FrozenTextEncoder
from samplers_t2i import euler_sampler


def build_latents_scale_bias(latents_stats, device, channels=16, vae=None):
    """Same convention as training: normalized = z*scale + bias."""
    if latents_stats and os.path.isfile(latents_stats):
        stats = torch.load(latents_stats, map_location="cpu", weights_only=False)
        scale_r, bias_r = latents_stats_to_repa(stats)
        return (scale_r.view(1, channels, 1, 1).to(device),
                bias_r.view(1, channels, 1, 1).to(device))
    if vae is not None and hasattr(vae, "initial_latents_stats"):
        scale_e, bias_e = vae.initial_latents_stats()
        return (scale_e.view(1, channels, 1, 1).to(device),
                (-bias_e * scale_e).view(1, channels, 1, 1).to(device))
    scale = float(FluxVAE.scaling_factor)
    shift = float(FluxVAE.shift_factor)
    return (torch.full((1, channels, 1, 1), scale, device=device),
            torch.full((1, channels, 1, 1), -shift * scale, device=device))


def get_cfg(ckpt_args, name, default):
    """Pull a value from the checkpoint's stored args (Namespace or dict)."""
    if ckpt_args is None:
        return default
    if isinstance(ckpt_args, dict):
        return ckpt_args.get(name, default)
    return getattr(ckpt_args, name, default)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f">>> loading checkpoint {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ca = ckpt.get("args", None)

    # ---- config from checkpoint (CLI can override) ----
    depth = args.depth or get_cfg(ca, "depth", 28)
    hidden_size = args.hidden_size or get_cfg(ca, "hidden_size", 1152)
    num_heads = args.num_heads or get_cfg(ca, "num_heads", 16)
    encoder_depth = get_cfg(ca, "encoder_depth", 8)
    resolution = args.resolution or get_cfg(ca, "resolution", 256)
    vae_arch = args.vae_arch or get_cfg(ca, "vae_arch", "flux")
    vae_ckpt = args.vae_ckpt or get_cfg(ca, "vae_ckpt", None)
    vae_vit_path = args.vae_vit_path if args.vae_vit_path is not None \
        else get_cfg(ca, "vae_vit_path", "Qwen/Qwen3-VL-2B-Instruct")
    if vae_arch == "qwenvit" and vae_vit_path and os.path.isabs(vae_vit_path) and not os.path.exists(vae_vit_path):
        vae_vit_path = "Qwen/Qwen3-VL-2B-Instruct"
    vae_decoder_width = get_cfg(ca, "vae_decoder_width", 1024)
    vae_decoder_depth = get_cfg(ca, "vae_decoder_depth", 12)
    vae_decoder_heads = get_cfg(ca, "vae_decoder_heads", 16)
    latents_stats = args.latents_stats if args.latents_stats is not None \
        else get_cfg(ca, "latents_stats", "")
    text_encoder_type = get_cfg(ca, "text_encoder", "clip")
    text_encoder_name = args.text_encoder_name if args.text_encoder_name is not None \
        else get_cfg(ca, "text_encoder_name", None)
    text_max_length = get_cfg(ca, "text_max_length", 77)
    path_type = get_cfg(ca, "path_type", "linear")
    steps_done = ckpt.get("steps", "?")
    print(f">>> ckpt steps={steps_done} | MMDiT depth={depth} hidden={hidden_size} "
          f"heads={num_heads} | res={resolution} | vae_arch={vae_arch} vae={vae_ckpt}")

    latent_size = resolution // 8
    in_channels = VAE_LATENT_CHANNELS[vae_arch]

    # ---- frozen text encoder + VAE ----
    text_encoder = FrozenTextEncoder(
        encoder_type=text_encoder_type, model_name=text_encoder_name,
        max_length=text_max_length, device=device,
    )
    vae = load_e2e_vae(
        vae_arch=vae_arch,
        vae_ckpt=vae_ckpt,
        device=device,
        eval_mode=True,
        model_path=vae_vit_path,
        latent_channels=in_channels,
        decoder_width=vae_decoder_width,
        decoder_depth=vae_decoder_depth,
        decoder_heads=vae_decoder_heads,
        decoder_grid=16,
    )
    latents_scale, latents_bias = build_latents_scale_bias(latents_stats, device, in_channels, vae=vae)

    # ---- MMDiT (weights-only) ----
    model = MMDiT(
        input_size=latent_size, in_channels=in_channels,
        depth=depth, hidden_size=hidden_size, num_heads=num_heads,
        encoder_depth=encoder_depth,
    ).to(device)
    which = "ema" if (args.weights == "ema" and "ema" in ckpt) else "model"
    sd = ckpt[which]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f">>> loaded [{which}] weights (missing={len(missing)} unexpected={len(unexpected)})")
    model.eval()

    # ---- prompts ----
    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompts = [ln.strip() for ln in f if ln.strip()]
    else:
        prompts = [p for p in args.prompts.split("||") if p.strip()]
    assert prompts, "no prompts given"
    print(f">>> {len(prompts)} prompt(s), {args.num_samples_per_prompt} sample(s) each")

    # repeat each prompt num_samples_per_prompt times
    all_prompts = [p for p in prompts for _ in range(args.num_samples_per_prompt)]

    os.makedirs(args.out, exist_ok=True)
    g = torch.Generator(device=device).manual_seed(args.seed)

    with torch.no_grad():
        context, _, _ = text_encoder.encode(all_prompts)
        null_1, _, _ = text_encoder.encode([""])
        null_context = null_1.repeat(len(all_prompts), 1, 1).to(context.dtype)

    imgs = []
    bs = args.batch_size
    for i in range(0, len(all_prompts), bs):
        ctx = context[i:i + bs]
        nctx = null_context[i:i + bs]
        n = ctx.shape[0]
        xT = torch.randn((n, in_channels, latent_size, latent_size),
                         device=device, generator=g)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            samples = euler_sampler(
                model, xT, ctx, y_null=nctx,
                num_steps=args.num_steps, cfg_scale=args.cfg_scale,
                guidance_low=0., guidance_high=1., path_type=path_type, heun=False,
            ).to(torch.float32)
        with torch.no_grad():
            img = vae.decode((samples - latents_bias) / latents_scale).sample
            img = ((img.float() + 1) / 2.).clamp(0, 1)
        imgs.append(img.cpu())
        print(f"    generated {min(i + bs, len(all_prompts))}/{len(all_prompts)}")

    imgs = torch.cat(imgs, dim=0)

    # individual images
    for idx in range(imgs.shape[0]):
        save_image(imgs[idx], os.path.join(args.out, f"sample_{idx:03d}.png"))
    # grid
    nrow = args.num_samples_per_prompt if args.num_samples_per_prompt > 1 \
        else max(1, round(math.sqrt(imgs.shape[0])))
    grid = make_grid(imgs, nrow=nrow, padding=2)
    grid_path = os.path.join(args.out, f"grid_step{steps_done}_cfg{args.cfg_scale}.png")
    save_image(grid, grid_path)
    print(f">>> saved {imgs.shape[0]} images + grid to {grid_path}")

    # also dump the prompt list for reference
    with open(os.path.join(args.out, "prompts.txt"), "w") as f:
        f.write("\n".join(all_prompts))


def parse_args():
    p = argparse.ArgumentParser(description="T2I inference for frozen-VAE MMDiT (REPA)")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out", type=str, default="infer_out")
    p.add_argument("--weights", type=str, default="ema", choices=["ema", "model"])
    p.add_argument("--prompts", type=str,
                   default="a photograph of an astronaut riding a horse||"
                           "a cute corgi puppy in a field of flowers, bokeh||"
                           "an oil painting of a lighthouse during a storm||"
                           "a bowl of ramen with an egg, food photography")
    p.add_argument("--prompt-file", type=str, default=None,
                   help="One prompt per line (overrides --prompts).")
    p.add_argument("--num-samples-per-prompt", type=int, default=1)
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--cfg-scale", type=float, default=4.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow-tf32", action="store_true", default=True)

    # optional overrides (else taken from ckpt["args"])
    p.add_argument("--resolution", type=int, default=None)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--hidden-size", type=int, default=None)
    p.add_argument("--num-heads", type=int, default=None)
    p.add_argument("--vae-arch", type=str, default=None, choices=["flux", "flux2", "qwenvit"])
    p.add_argument("--vae-ckpt", type=str, default=None)
    p.add_argument("--vae-vit-path", type=str, default=None,
                   help="Override Qwen3-VL model path for --vae-arch qwenvit.")
    p.add_argument("--latents-stats", type=str, default=None)
    p.add_argument("--text-encoder-name", type=str, default=None,
                   help="Override the text encoder path stored in the checkpoint "
                        "(useful when evaluating a checkpoint from another machine).")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
