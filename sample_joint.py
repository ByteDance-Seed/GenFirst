# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

"""Sample both directions from a joint MMDiT checkpoint.

Runs, on the same batch of real image/caption pairs:

1. a text VAE round-trip, which fails loudly if the checkpoint's text VAE
   weights did not load;
2. ``t2i`` — captions in, images out, written as a PNG grid next to the
   ground-truth image for each caption;
3. ``i2t`` — images in, captions out, printed against the ground truth.

Usage::

    python sample_joint.py --ckpt exps/joint-e2e-256-L2-ent001/checkpoints/0050000.pt
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image

from dataset import CaptionCollator, ParquetT2IDataset
from models.autoencoder import FluxVAE
from models.joint_mmdit import JointMMDiT_models
from models.text_vae import ColaTextVAE
from samplers_joint import decode_text_ids, i2t_sample, ids_to_text, t2i_sample
from utils import preprocess_imgs_vae


def to_uint8(x):
    """``(B, 3, H, W)`` in [-1, 1] -> ``(B, H, W, 3)`` uint8."""
    x = torch.clamp(255.0 * (x + 1) / 2, 0, 255)
    return x.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()


def save_grid(rows, path):
    """Save a list of equal-length uint8 image rows as one grid."""
    h, w = rows[0][0].shape[:2]
    grid = np.zeros((h * len(rows), w * len(rows[0]), 3), dtype=np.uint8)
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = img
    Image.fromarray(grid).save(path)


def main(args):
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    # mmap: a joint checkpoint is ~19G, two thirds of it optimizer state that
    # sampling never touches, and it usually sits on a fuse mount.
    ckpt = torch.load(args.ckpt, map_location="cpu", mmap=True, weights_only=False)
    targs = ckpt["args"]
    step = ckpt.get("steps", -1)
    print(f"loaded {args.ckpt} (step {step}), model={targs.model}")

    image_vae_pretrained = (
        args.image_vae_pretrained
        or getattr(targs, "image_vae_pretrained", None)
    )
    if image_vae_pretrained and not os.path.exists(image_vae_pretrained):
        image_vae_pretrained = None
    image_vae = FluxVAE(pretrained_path=image_vae_pretrained)
    if ckpt.get("image_vae") is not None:
        # The run trained the image VAE, so its weights moved away from whatever
        # --image-vae-ckpt warm-started them from; decoding with the warm-start
        # copy would misrepresent the model.
        image_vae.load_state_dict(ckpt["image_vae"])
        print("image VAE: trained copy from the checkpoint")
    else:
        vae_ckpt = args.image_vae_ckpt or getattr(targs, "image_vae_ckpt", None)
        if vae_ckpt:
            sd = torch.load(vae_ckpt, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "vae" in sd and "vae.encoder.conv_in.weight" not in sd:
                sd = sd["vae"]
            image_vae.load_state_dict(sd, strict=False)
            print(f"image VAE: frozen, from {vae_ckpt}")
    image_vae = image_vae.to(device).eval()

    text_vae_path = args.text_vae_path or getattr(targs, "text_vae_path", None)
    if not text_vae_path or not os.path.exists(text_vae_path):
        default_cola = os.path.join("pretrained", "cola-dlm", "cola_dlm", "cola_vae")
        if os.path.exists(default_cola):
            text_vae_path = default_cola
    text_vae = ColaTextVAE(pretrained_path=text_vae_path, train_trunk=False)
    text_vae.load_state_dict(ckpt["text_vae"])
    text_vae = text_vae.to(device).eval()

    latent_size = targs.resolution // 8
    model = JointMMDiT_models[targs.model](
        input_size=latent_size,
        image_channels=16,
        text_channels=text_vae.latent_dim,
        text_block_size=targs.text_block_size,
        max_text_tokens=targs.max_text_tokens,
        path_type=targs.path_type,
        # Frozen-stage checkpoints carry REPA projection heads; they are unused
        # at sampling time but have to exist for the state dict to load.
        z_dims=getattr(targs, "z_dims", None) or (),
        encoder_depth=getattr(targs, "encoder_depth", 8),
    ).to(device)
    model.load_state_dict(ckpt["ema" if args.use_ema else "model"])
    model.eval()
    print(f"using {'EMA' if args.use_ema else 'raw'} weights")

    trained_dropout = getattr(targs, "cond_dropout", 0.0)
    if args.cfg_scale != 1.0 and trained_dropout <= 0:
        raise SystemExit(
            f"--cfg-scale={args.cfg_scale} needs a checkpoint trained with "
            "condition dropout, but this one has cond_dropout=0: it has no "
            "unconditional branch. Re-run with --cfg-scale=1."
        )

    data_dir = args.data_dir or targs.data_dir
    dataset = ParquetT2IDataset(
        [r for r in data_dir.split(",") if r],
        resolution=targs.resolution,
        caption_columns=[c for c in targs.caption_columns.split(",") if c],
        rows_per_file=targs.parquet_rows_per_file,
        random_caption=targs.random_caption,
    )
    tokenizer_path = args.tokenizer_path or getattr(targs, "tokenizer_path", None)
    if not tokenizer_path or not os.path.exists(tokenizer_path):
        default_tok = os.path.join("pretrained", "cola-dlm", "tokenizer.json")
        if os.path.exists(default_tok):
            tokenizer_path = default_tok
    collator = CaptionCollator(
        tokenizer_path,
        block_size=targs.text_block_size,
        max_tokens=targs.max_text_tokens,
    )
    prompts = read_prompts(args)
    n_samples = len(prompts) if prompts else args.num_samples
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=n_samples, num_workers=2, collate_fn=collator
    )
    batch = next(iter(loader))

    # i2t always conditions on real images; --prompts only replaces the text side,
    # in which case the dataset captions are no longer the t2i ground truth.
    images = batch["images"].to(device)
    if prompts:
        seqs = [collator.encode_one(p) for p in prompts]
        L = max(len(s) for s in seqs)
        text_valid = torch.zeros(len(seqs), L, dtype=torch.bool, device=device)
        for i, s in enumerate(seqs):
            text_valid[i, : len(s)] = True
        text_ids = [torch.tensor(s, dtype=torch.long, device=device) for s in seqs]
        captions = list(prompts)
    else:
        text_ids = [t.to(device) for t in batch["text_ids"]]
        text_valid = batch["text_valid"].to(device)
        captions = batch["captions"]
        L = text_valid.shape[1]

    os.makedirs(args.out_dir, exist_ok=True)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        # ---------------------------------------------------- text round-trip
        posterior, txt_shape = text_vae.encode(text_ids)
        z_txt_na = posterior.mode()
        logits = text_vae.decode(z_txt_na, txt_shape)
        targets = torch.cat(text_ids)
        acc = (logits.argmax(-1) == targets).float().mean().item()
        print(f"\ntext VAE round-trip accuracy: {100 * acc:.2f}%  "
              f"(expect ~100%; low means the text VAE did not load)")

        z_txt = torch.zeros(len(text_ids), L, text_vae.latent_dim, device=device)
        for i, part in enumerate(z_txt_na.split(txt_shape.flatten().tolist())):
            z_txt[i, : part.shape[0]] = part.float()

        # ------------------------------------------------------------- t2i
        print(f"\nsampling t2i ({args.t2i_steps} steps, cfg={args.cfg_scale}, "
              f"heun={args.heun}) ...")
        z_img = t2i_sample(
            model, z_txt, text_valid, num_steps=args.t2i_steps,
            cfg_scale=args.cfg_scale, heun=args.heun,
        )
        recon = image_vae.decode(z_img.to(torch.bfloat16)).sample.float()
        gen = to_uint8(recon)
        rows = [list(gen)]
        note = "generated"
        if not prompts:
            rows.insert(0, list(to_uint8(preprocess_imgs_vae(images))))
            note = "row 1 ground truth, row 2 generated"
        save_grid(rows, os.path.join(args.out_dir, f"t2i_{step}.png"))
        print(f"wrote {args.out_dir}/t2i_{step}.png  ({note})")
        for i, c in enumerate(captions):
            print(f"  [{i}] {c}")

        # ------------------------------------------------------------- i2t
        # Length comes from the dataset captions, so --prompts cannot change how
        # much text i2t is asked to produce.
        n_blocks = max(1, batch["text_valid"].shape[1] // targs.text_block_size)
        print(f"\nsampling i2t ({args.i2t_steps} steps/block, {n_blocks} blocks) ...")
        z_img_gt = image_vae.encode(preprocess_imgs_vae(images)).mode().float()
        z_gen, valid_gen = i2t_sample(
            model, z_img_gt, n_blocks, num_steps=args.i2t_steps,
            cfg_scale=args.cfg_scale, heun=args.heun,
            eos_id=collator.eos_id, text_vae=text_vae,
        )
        ids = decode_text_ids(text_vae, z_gen)
        texts = ids_to_text(collator.tok, ids, valid_gen, eos_id=collator.eos_id)

    print("\n" + "=" * 78)
    # The reference for i2t is always the dataset caption of the image it saw.
    for i, (gt_cap, gen_cap) in enumerate(zip(batch["captions"], texts)):
        print(f"[{i}] GT : {gt_cap}")
        print(f"[{i}] GEN: {gen_cap}")
        print("-" * 78)


def read_prompts(args):
    """Custom t2i prompts from ``--prompts`` or one-per-line ``--prompt-file``."""
    if args.prompt_file:
        with open(args.prompt_file) as f:
            return [ln.strip() for ln in f if ln.strip()]
    return list(args.prompts or [])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--image-vae-ckpt", type=str, default="")
    p.add_argument("--image-vae-pretrained", type=str, default="")
    p.add_argument("--text-vae-path", type=str, default="")
    p.add_argument("--tokenizer-path", type=str, default="")
    p.add_argument("--data-dir", type=str, default="")
    p.add_argument("--use-ema", action="store_true")
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--prompts", type=str, nargs="+",
                   help="t2i prompts to use instead of dataset captions; i2t "
                        "still conditions on dataset images")
    p.add_argument("--prompt-file", type=str, help="one prompt per line")
    p.add_argument("--t2i-steps", type=int, default=50)
    p.add_argument("--i2t-steps", type=int, default=20)
    p.add_argument("--cfg-scale", type=float, default=1.0,
                   help="needs a checkpoint trained with --cond-dropout > 0")
    p.add_argument("--heun", action="store_true")
    p.add_argument("--out-dir", type=str, default="samples")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
