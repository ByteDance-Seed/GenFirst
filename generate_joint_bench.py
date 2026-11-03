# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

"""Generate the GenEval and DPG-Bench prompt sets from a joint MMDiT checkpoint.

Only the t2i direction is exercised here. Both harnesses score a directory of
images and each wants its own layout, which is what this writes:

* ``geneval`` -- ``{out}/{index:05d}/samples/{i:05d}.png`` plus that prompt's
  ``metadata.jsonl``, the tree ``geneval/evaluation/evaluate_images.py`` walks.
  ``index`` is the prompt's line number in ``evaluation_metadata.jsonl``.
* ``dpg`` -- ``{out}/{item_id}.png``, one 2x2 grid per prompt. The scorer
  ``compute_dpg_bench.py`` crops each grid back into four tiles, so *its own*
  ``--resolution`` has to be the checkpoint's resolution, not the grid's.

Scoring is a separate step; set up the harnesses and their detectors with
``eval_bench/setup_eval_env.sh`` and see ``docs/T2I_EVAL.md``.

Sampling is sharded over ranks by prompt, and a prompt whose images are already
on disk is skipped, so an interrupted run can simply be relaunched.

Usage::

    torchrun --standalone --nproc_per_node=8 generate_joint_bench.py \
        --ckpt Prior_only/exps/<run>/checkpoints/0020000.pt \
        --bench geneval --out-dir evals/<run>/geneval \
        --geneval-metadata <bench>/geneval/prompts/evaluation_metadata.jsonl
"""

import argparse
import csv
import json
import os
import zlib
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from dataset import CaptionCollator
from models.autoencoder import FluxVAE
from models.joint_mmdit import JointMMDiT_models
from models.text_vae import ColaTextVAE
from samplers_joint import t2i_sample


def to_uint8(x):
    """``(B, 3, H, W)`` in [-1, 1] -> ``(B, H, W, 3)`` uint8."""
    x = torch.clamp(255.0 * (x + 1) / 2, 0, 255)
    return x.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()


def load_geneval(args):
    """``(key, prompt, metadata)`` per line of GenEval's metadata file."""
    path = args.geneval_metadata
    items = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                meta = json.loads(line)
                items.append((f"{i:05d}", meta["prompt"], meta))
    return items


def load_dpg(args):
    """``(item_id, prompt, None)`` per DPG prompt.

    Driven by the csv rather than the ``prompts/`` folder because the scorer
    keys its questions off ``item_id``, and image filenames have to match those.
    """
    seen = {}
    with open(args.dpg_csv) as f:
        for row in csv.DictReader(f):
            seen.setdefault(row["item_id"], row["text"])
    return [(k, v, None) for k, v in seen.items()]


def is_done(bench, out_dir, key, n_samples):
    if bench == "dpg":
        return os.path.exists(os.path.join(out_dir, f"{key}.png"))
    sample_dir = os.path.join(out_dir, key, "samples")
    return all(
        os.path.exists(os.path.join(sample_dir, f"{i:05d}.png")) for i in range(n_samples)
    )


def write_geneval(out_dir, key, meta, imgs):
    outpath = os.path.join(out_dir, key)
    sample_dir = os.path.join(outpath, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    with open(os.path.join(outpath, "metadata.jsonl"), "w") as f:
        json.dump(meta, f)
    for i, img in enumerate(imgs):
        Image.fromarray(img).save(os.path.join(sample_dir, f"{i:05d}.png"))


def write_dpg(out_dir, key, imgs):
    """One 2x2 grid, in the tile order compute_dpg_bench.py crops back out."""
    h, w = imgs[0].shape[:2]
    grid = np.zeros((2 * h, 2 * w, 3), dtype=np.uint8)
    for i, img in enumerate(imgs[:4]):
        r, c = divmod(i, 2)
        grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = img
    Image.fromarray(grid).save(os.path.join(out_dir, f"{key}.png"))


def load_everything(args, device):
    # mmap: two thirds of the checkpoint is optimizer state sampling never reads.
    ckpt = torch.load(args.ckpt, map_location="cpu", mmap=True, weights_only=False)
    targs = ckpt["args"]
    step = ckpt.get("steps", -1)

    image_vae_pretrained = (
        getattr(args, "image_vae_pretrained", None)
        or getattr(targs, "image_vae_pretrained", None)
    )
    if image_vae_pretrained and not os.path.exists(image_vae_pretrained):
        image_vae_pretrained = None
    image_vae = FluxVAE(pretrained_path=image_vae_pretrained)
    if ckpt.get("image_vae") is not None:
        # An end-to-end run moved these weights away from whatever warm-started
        # them, so the checkpoint's copy is the only correct decoder.
        image_vae.load_state_dict(ckpt["image_vae"])
    else:
        # --image-vae-ckpt overrides what training recorded, which a run whose VAE
        # has since moved needs.
        vae_ckpt = (getattr(args, "image_vae_ckpt", None)
                    or getattr(targs, "image_vae_ckpt", None))
        if vae_ckpt and not os.path.exists(vae_ckpt):
            # Frozen-stage runs keep the image VAE out of the checkpoint, and
            # FluxVAE(pretrained_path=None) builds it from config with random
            # weights, so a stale path would silently encode noise.
            raise SystemExit(
                f"image VAE not found: {vae_ckpt}\n"
                f"It is the path {args.ckpt} recorded at training time; pass "
                f"--image-vae-ckpt to point at where those weights live now."
            )
        if vae_ckpt:
            sd = torch.load(vae_ckpt, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "vae" in sd and "vae.encoder.conv_in.weight" not in sd:
                sd = sd["vae"]
            image_vae.load_state_dict(sd, strict=False)
    image_vae = image_vae.to(device).eval()

    text_vae_path = (
        getattr(args, "text_vae_path", None)
        or getattr(targs, "text_vae_path", None)
    )
    if not text_vae_path or not os.path.exists(text_vae_path):
        default_cola = os.path.join("pretrained", "cola-dlm", "cola_dlm", "cola_vae")
        if os.path.exists(default_cola):
            text_vae_path = default_cola
        else:
            raise SystemExit(
                f"text VAE config directory not found: {text_vae_path}\n"
                f"Pass --text-vae-path (or download Cola DLM to pretrained/cola-dlm)."
            )
    text_vae = ColaTextVAE(pretrained_path=text_vae_path, train_trunk=False)
    text_vae.load_state_dict(ckpt["text_vae"])
    text_vae = text_vae.to(device).eval()

    model = JointMMDiT_models[targs.model](
        input_size=targs.resolution // 8,
        image_channels=16,
        text_channels=text_vae.latent_dim,
        text_block_size=targs.text_block_size,
        max_text_tokens=targs.max_text_tokens,
        path_type=targs.path_type,
        # Unused at sampling time, but the state dict carries them.
        z_dims=getattr(targs, "z_dims", None) or (),
        encoder_depth=getattr(targs, "encoder_depth", 8),
    ).to(device)
    model.load_state_dict(ckpt["ema" if args.use_ema else "model"])
    model.eval()

    tokenizer_path = (
        getattr(args, "tokenizer_path", None)
        or getattr(targs, "tokenizer_path", None)
    )
    if not tokenizer_path or not os.path.exists(tokenizer_path):
        default_tok = os.path.join("pretrained", "cola-dlm", "tokenizer.json")
        if os.path.exists(default_tok):
            tokenizer_path = default_tok
        else:
            raise SystemExit(
                f"tokenizer not found: {tokenizer_path}\n"
                f"Pass --tokenizer-path (or download Cola DLM to pretrained/cola-dlm)."
            )
    collator = CaptionCollator(
        tokenizer_path,
        block_size=targs.text_block_size,
        max_tokens=targs.max_text_tokens,
    )
    return model, image_vae, text_vae, collator, targs, step


def encode_prompts(text_vae, collator, prompts, device):
    """Prompts -> ``(z_txt (B, L, C), text_valid (B, L))`` on the padded grid.

    ``encode_one`` already rounds each prompt up to a whole number of blocks, so
    the padded width stays block-aligned.
    """
    seqs = [collator.encode_one(p) for p in prompts]
    L = max(len(s) for s in seqs)
    text_valid = torch.zeros(len(seqs), L, dtype=torch.bool, device=device)
    for i, s in enumerate(seqs):
        text_valid[i, : len(s)] = True
    text_ids = [torch.tensor(s, dtype=torch.long, device=device) for s in seqs]

    posterior, txt_shape = text_vae.encode(text_ids)
    z_txt_na = posterior.mode()
    z_txt = torch.zeros(len(seqs), L, text_vae.latent_dim, device=device)
    for i, part in enumerate(z_txt_na.split(txt_shape.flatten().tolist())):
        z_txt[i, : part.shape[0]] = part.float()
    return z_txt, text_valid


def main(args):
    distributed = "RANK" in os.environ
    if distributed:
        # Generous timeout: ranks drift apart over a long generation run and then
        # wait for each other in a barrier, which NCCL's 10-minute default turns
        # into a crash near completion.
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
        rank = dist.get_rank()
        world = dist.get_world_size()
        torch.cuda.set_device(rank % torch.cuda.device_count())
    else:
        rank, world = 0, 1
    device = torch.device("cuda")
    is_main = rank == 0
    torch.set_grad_enabled(False)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    model, image_vae, text_vae, collator, targs, step = load_everything(args, device)
    if args.cfg_scale != 1.0 and getattr(targs, "cond_dropout", 0.0) <= 0:
        raise SystemExit(
            f"--cfg-scale={args.cfg_scale} needs a checkpoint trained with condition "
            "dropout; this one has cond_dropout=0, so it has no unconditional branch."
        )
    if args.bench == "dpg" and args.n_samples != 4:
        raise SystemExit("dpg scores a 2x2 grid, so --n-samples must be 4")

    items = (load_geneval if args.bench == "geneval" else load_dpg)(args)
    if args.limit:
        items = items[: args.limit]
    os.makedirs(args.out_dir, exist_ok=True)

    todo = [it for it in items if not is_done(args.bench, args.out_dir, it[0], args.n_samples)]
    if is_main:
        print(
            f"{args.ckpt} (step {step}, {'EMA' if args.use_ema else 'raw'}) "
            f"-> {args.out_dir}\n"
            f"{args.bench}: {len(items)} prompts, {len(todo)} left x {args.n_samples} "
            f"images @ {targs.resolution}px, cfg={args.cfg_scale}, "
            f"steps={args.t2i_steps}, {world} rank(s)",
            flush=True,
        )
    mine = todo[rank::world]

    groups = [
        mine[i : i + args.prompts_per_batch]
        for i in range(0, len(mine), args.prompts_per_batch)
    ]
    g_cpu = torch.Generator()  # CPU so the noise does not depend on the device
    for gi, group in enumerate(groups):
        keys = [k for k, _, _ in group]
        z_txt, valid = encode_prompts(text_vae, collator, [p for _, p, _ in group], device)
        n = args.n_samples
        z_txt = z_txt.repeat_interleave(n, dim=0)
        valid = valid.repeat_interleave(n, dim=0)

        # Seeded per (prompt, sample) so batching and resumption cannot alter it.
        # crc32 rather than hash(), which is salted per process.
        side = targs.resolution // 8
        noise = []
        for key, _, _ in group:
            base = zlib.crc32(f"{args.seed}:{key}".encode()) % (2**31 - n)
            for s in range(n):
                g_cpu.manual_seed(base + s)
                noise.append(torch.randn(16, side, side, generator=g_cpu))
        x_init = torch.stack(noise).to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            z_img = t2i_sample(
                model, z_txt, valid, num_steps=args.t2i_steps,
                cfg_scale=args.cfg_scale, heun=args.heun, x_init=x_init,
            )
            imgs = []
            for chunk in z_img.split(args.decode_chunk):
                imgs.append(to_uint8(image_vae.decode(chunk.to(torch.bfloat16)).sample.float()))
        imgs = np.concatenate(imgs)

        for i, (key, _, meta) in enumerate(group):
            batch = list(imgs[i * n : (i + 1) * n])
            if args.bench == "geneval":
                write_geneval(args.out_dir, key, meta, batch)
            else:
                write_dpg(args.out_dir, key, batch)

        if is_main and (gi % args.log_every == 0 or gi == len(groups) - 1):
            print(f"  [{gi + 1}/{len(groups)}] {keys[0]}..{keys[-1]}", flush=True)

    if distributed:
        dist.barrier()
    if is_main:
        done = sum(
            1 for it in items if is_done(args.bench, args.out_dir, it[0], args.n_samples)
        )
        print(f"done: {done}/{len(items)} prompts in {args.out_dir}", flush=True)
    if distributed:
        dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--bench", type=str, required=True, choices=["geneval", "dpg"])
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--use-ema", action="store_true")
    p.add_argument("--image-vae-ckpt", type=str, default="",
                   help="override the frozen image VAE recorded in the checkpoint, "
                        "for a run whose VAE has since moved")
    p.add_argument("--image-vae-pretrained", type=str, default="",
                   help="override the pretrained FLUX VAE directory recorded in the checkpoint")
    p.add_argument("--text-vae-path", type=str, default="",
                   help="override the Cola text VAE directory recorded in the checkpoint")
    p.add_argument("--tokenizer-path", type=str, default="",
                   help="override the tokenizer.json path recorded in the checkpoint")
    p.add_argument("--n-samples", type=int, default=4,
                   help="images per prompt; dpg's 2x2 grid requires 4")
    p.add_argument("--prompts-per-batch", type=int, default=2,
                   help="prompts per forward; the batch is this x --n-samples, "
                        "and doubles again when cfg is on")
    p.add_argument("--decode-chunk", type=int, default=4,
                   help="latents per VAE decode call")
    p.add_argument("--t2i-steps", type=int, default=50)
    p.add_argument("--cfg-scale", type=float, default=4.0,
                   help="4.0 matches the training-time sample logging")
    p.add_argument("--heun", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="first N prompts only, for smoke tests")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--allow-tf32", action="store_true")
    p.add_argument("--geneval-metadata", type=str, default=None,
                   help="GenEval's evaluation_metadata.jsonl; required for --bench geneval")
    p.add_argument("--dpg-csv", type=str, default=None,
                   help="DPG-Bench's dpg_bench.csv; required for --bench dpg")
    args = p.parse_args()
    if args.bench == "geneval":
        assert args.geneval_metadata, "--geneval-metadata is required for --bench geneval"
    else:
        assert args.dpg_csv, "--dpg-csv is required for --bench dpg"
    return args


if __name__ == "__main__":
    main(parse_args())
