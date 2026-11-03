# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

"""Batch image generation for GenEval / DPG-Bench using the frozen-VAE MMDiT.

Reuses the SAME model / FLUX VAE / CLIP / euler-sampler as train_t2i_frozen.py
(config auto-restored from ckpt["args"]). Supports multi-GPU sharding: launch one
process per GPU with --shard-id/--num-shards.

GenEval layout (consumed by geneval/evaluation/evaluate_images.py):
  <out>/<00000>/metadata.jsonl            # the metadata line for this prompt
  <out>/<00000>/samples/0000.png ... 000{n-1}.png

DPG layout (flat PNGs; later stitched to 2x2 grid by cat step):
  <out>/<item_id>_0.png ... <item_id>_{n-1}.png
"""

import argparse
import json
import math
import os

import numpy as np
import torch
from PIL import Image

from e2e_vae import load_e2e_vae, latents_stats_to_repa, FluxVAE, VAE_LATENT_CHANNELS
from models.mmdit import MMDiT
from models.text_encoder import FrozenTextEncoder
from samplers_t2i import euler_sampler


def get_cfg(ca, name, default):
    if ca is None:
        return default
    if isinstance(ca, dict):
        return ca.get(name, default)
    return getattr(ca, name, default)


def build_latents_scale_bias(latents_stats, device, channels=16, vae=None):
    if latents_stats and os.path.isfile(latents_stats):
        stats = torch.load(latents_stats, map_location="cpu", weights_only=False)
        s, b = latents_stats_to_repa(stats)
        return s.view(1, channels, 1, 1).to(device), b.view(1, channels, 1, 1).to(device)
    if vae is not None and hasattr(vae, "initial_latents_stats"):
        scale_e, bias_e = vae.initial_latents_stats()
        return (scale_e.view(1, channels, 1, 1).to(device),
                (-bias_e * scale_e).view(1, channels, 1, 1).to(device))
    scale = float(FluxVAE.scaling_factor)
    shift = float(FluxVAE.shift_factor)
    return (torch.full((1, channels, 1, 1), scale, device=device),
            torch.full((1, channels, 1, 1), -shift * scale, device=device))


def load_model_and_vae(args, device):
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ca = ckpt.get("args", None)
    depth = args.depth or get_cfg(ca, "depth", 28)
    hidden = args.hidden_size or get_cfg(ca, "hidden_size", 1152)
    heads = args.num_heads or get_cfg(ca, "num_heads", 16)
    enc_depth = get_cfg(ca, "encoder_depth", 8)
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
    te_type = get_cfg(ca, "text_encoder", "clip")
    te_name = args.text_encoder_name if args.text_encoder_name is not None \
        else get_cfg(ca, "text_encoder_name", None)
    te_max = get_cfg(ca, "text_max_length", 77)
    path_type = get_cfg(ca, "path_type", "linear")

    latent_size = resolution // 8
    in_ch = VAE_LATENT_CHANNELS[vae_arch]
    text_encoder = FrozenTextEncoder(encoder_type=te_type, model_name=te_name,
                                     max_length=te_max, device=device)
    vae = load_e2e_vae(
        vae_arch=vae_arch,
        vae_ckpt=vae_ckpt,
        device=device,
        eval_mode=True,
        model_path=vae_vit_path,
        latent_channels=in_ch,
        decoder_width=vae_decoder_width,
        decoder_depth=vae_decoder_depth,
        decoder_heads=vae_decoder_heads,
        decoder_grid=16,
    )
    scale, bias = build_latents_scale_bias(latents_stats, device, in_ch, vae=vae)
    model = MMDiT(input_size=latent_size, in_channels=in_ch, depth=depth,
                  hidden_size=hidden, num_heads=heads, encoder_depth=enc_depth).to(device)
    which = "ema" if (args.weights == "ema" and "ema" in ckpt) else "model"
    sd = {k.replace("module.", ""): v for k, v in ckpt[which].items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    meta = dict(latent_size=latent_size, in_ch=in_ch, path_type=path_type,
                resolution=resolution, which=which, steps=ckpt.get("steps", "?"))
    return model, vae, text_encoder, scale, bias, meta


@torch.no_grad()
def gen_batch(model, vae, text_encoder, scale, bias, meta, prompts, seeds, args, device):
    """Generate one image per (prompt, seed). Returns list of uint8 HxWx3 arrays."""
    n = len(prompts)
    ls = meta["latent_size"]; ic = meta["in_ch"]
    context, _, _ = text_encoder.encode(list(prompts))
    null_1, _, _ = text_encoder.encode([""])
    null_context = null_1.repeat(n, 1, 1).to(context.dtype)
    # deterministic noise per seed
    xT = torch.empty((n, ic, ls, ls), device=device)
    for i, s in enumerate(seeds):
        g = torch.Generator(device=device).manual_seed(int(s))
        xT[i] = torch.randn((ic, ls, ls), device=device, generator=g)
    with torch.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
        samples = euler_sampler(model, xT, context, y_null=null_context,
                                num_steps=args.num_steps, cfg_scale=args.cfg_scale,
                                guidance_low=args.guidance_low,
                                guidance_high=args.guidance_high,
                                path_type=meta["path_type"], heun=False).to(torch.float32)
    img = vae.decode((samples - bias) / scale).sample
    img = ((img.float() + 1) / 2.).clamp(0, 1)
    img = (img * 255 + 0.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    return [img[i] for i in range(n)]


def build_tasks_geneval(args):
    """Each task = (folder_path, prompt, seed, sample_name, metadata_line)."""
    with open(args.metadata) as f:
        metas = [json.loads(l) for l in f if l.strip()]
    tasks = []
    for pi, m in enumerate(metas):
        folder = os.path.join(args.out, f"{pi:05d}")
        for s in range(args.n_per_prompt):
            tasks.append((folder, m["prompt"], s, f"{s:04d}.png", m))
    return tasks


def build_tasks_dpg(args):
    files = sorted([f for f in os.listdir(args.prompt_dir) if f.endswith(".txt")])
    tasks = []
    for fn in files:
        stem = os.path.splitext(fn)[0]
        with open(os.path.join(args.prompt_dir, fn)) as f:
            prompt = f.read().strip().split("\n")[0].strip()
        for s in range(args.n_per_prompt):
            tasks.append((args.out, prompt, s, f"{stem}_{s}.png", None))
    return tasks


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.bench == "geneval":
        tasks = build_tasks_geneval(args)
    else:
        tasks = build_tasks_dpg(args)

    # shard across processes/GPUs
    tasks = tasks[args.shard_id::args.num_shards]
    print(f">>> shard {args.shard_id}/{args.num_shards}: {len(tasks)} images to gen", flush=True)

    os.makedirs(args.out, exist_ok=True)
    model, vae, text_encoder, scale, bias, meta = load_model_and_vae(args, device)
    print(f">>> model steps={meta['steps']} which={meta['which']} res={meta['resolution']}", flush=True)

    # skip already-done, write geneval metadata.jsonl once per folder
    todo = []
    for folder, prompt, seed, name, m in tasks:
        if args.bench == "geneval":
            samples_dir = os.path.join(folder, "samples")
            outpath = os.path.join(samples_dir, name)
        else:
            outpath = os.path.join(folder, name)
        if os.path.isfile(outpath) and not args.overwrite:
            continue
        todo.append((folder, prompt, seed, name, m, outpath))

    print(f">>> {len(todo)} images after skip-existing", flush=True)
    done = 0
    for i in range(0, len(todo), args.batch_size):
        chunk = todo[i:i + args.batch_size]
        prompts = [c[1] for c in chunk]
        seeds = [c[2] for c in chunk]
        imgs = gen_batch(model, vae, text_encoder, scale, bias, meta, prompts, seeds, args, device)
        for (folder, prompt, seed, name, m, outpath), im in zip(chunk, imgs):
            os.makedirs(os.path.dirname(outpath), exist_ok=True)
            Image.fromarray(im).save(outpath)
            if args.bench == "geneval" and m is not None:
                mp = os.path.join(folder, "metadata.jsonl")
                if not os.path.isfile(mp):
                    with open(mp, "w") as fp:
                        json.dump(m, fp)
        done += len(chunk)
        if (i // args.batch_size) % 10 == 0:
            print(f"    [{args.shard_id}] {done}/{len(todo)}", flush=True)
    print(f">>> shard {args.shard_id} DONE ({done} imgs)", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--bench", choices=["geneval", "dpg"], required=True)
    p.add_argument("--metadata", default=None, help="geneval evaluation_metadata.jsonl")
    p.add_argument("--prompt-dir", default=None, help="dpg prompts dir (*.txt)")
    p.add_argument("--n-per-prompt", type=int, default=4)
    # Defaults are the paper's text-to-image inference settings (table 16).
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--cfg-scale", type=float, default=7.0)
    p.add_argument("--guidance-low", type=float, default=0.35)
    p.add_argument("--guidance-high", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--weights", default="ema", choices=["ema", "model"])
    p.add_argument("--overwrite", action="store_true")
    # sharding
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    # optional overrides
    p.add_argument("--resolution", type=int, default=None)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--hidden-size", type=int, default=None)
    p.add_argument("--num-heads", type=int, default=None)
    p.add_argument("--vae-arch", default=None, choices=["flux", "flux2", "qwenvit"])
    p.add_argument("--vae-ckpt", default=None)
    p.add_argument("--vae-vit-path", default=None,
                   help="Override Qwen3-VL model path for --vae-arch qwenvit.")
    p.add_argument("--latents-stats", default=None)
    p.add_argument("--text-encoder-name", default=None,
                   help="Override the text encoder path stored in the checkpoint "
                        "(useful for released checkpoints trained on another machine).")
    a = p.parse_args()
    if a.bench == "geneval":
        assert a.metadata, "--metadata required for geneval"
    else:
        assert a.prompt_dir, "--prompt-dir required for dpg"
    return a


if __name__ == "__main__":
    main(parse_args())
