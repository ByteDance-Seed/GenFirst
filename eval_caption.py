# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

"""Generate COCO captions with the i2t direction of a joint MMDiT.

Writes one JSONL row per image (``cocoid`` + generated caption); scoring lives in
``score_caption.py`` so that a single expensive sampling pass can be scored on
several splits and with several metrics.

Two splits matter here and they are *not* interchangeable:

* ``val2017`` -- 5000 images, disjoint from COCO train2017 and therefore from
  this project's training mix. The clean number.
* ``karpathy-test`` -- the 5000 images the captioning literature reports on, but
  4407 of them (88%) live in COCO train2017, so the model has seen those images
  during training (with synthetic Qwen2.5-VL captions, not the human ones).

``--split union`` samples both at once (9407 unique images) so the two can be
scored from one pass.

Usage (scripts/eval_caption.sh wraps this):
    torchrun --standalone --nproc_per_node=8 eval_caption.py \
        --ckpt <ckpt> --use-ema --split union --coco-root /data/coco \
        --karpathy-json /data/coco_karpathy/dataset_coco.json --out captions.jsonl
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from dataset import _center_crop_resize
from generate_joint_bench import load_everything
from samplers_joint import decode_text_ids, i2t_sample, ids_to_text
from utils import preprocess_imgs_vae


def build_image_index(coco_root):
    """``cocoid -> local jpg path`` over both COCO 2017 directories.

    Both are scanned even for val2017 alone, because the Karpathy test images are
    spread across the two 2017 directories.
    """
    index = {}
    for sub in ("train2017", "val2017"):
        d = os.path.join(coco_root, sub)
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if name.endswith(".jpg"):
                index[int(os.path.splitext(name)[0])] = os.path.join(d, name)
    return index


def select_ids(split, coco_root, karpathy_json):
    """Image ids for the requested split, in a fixed order."""
    ids = set()
    if split in ("val2017", "union"):
        ann = json.load(open(os.path.join(coco_root, "annotations/captions_val2017.json")))
        ids |= {im["id"] for im in ann["images"]}
    if split in ("karpathy-test", "union"):
        if not karpathy_json:
            raise SystemExit(f"--split {split} needs --karpathy-json, i.e. the "
                             f"dataset_coco.json of Karpathy's splits")
        kar = json.load(open(karpathy_json))
        ids |= {im["cocoid"] for im in kar["images"] if im["split"] == "test"}
    return sorted(ids)


class CocoImages(torch.utils.data.Dataset):
    """Center-crop-and-resize exactly as the training parquets were packed."""

    def __init__(self, ids, index, resolution):
        self.ids = ids
        self.index = index
        self.resolution = resolution

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        cid = self.ids[i]
        img = Image.open(self.index[cid]).convert("RGB")
        img = _center_crop_resize(img, self.resolution)
        arr = torch.from_numpy(np.asarray(img, dtype="uint8").copy()).permute(2, 0, 1)
        return arr, cid


def main(args):
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model, image_vae, text_vae, collator, targs, step = load_everything(args, device)
    if rank == 0:
        print(f"{args.ckpt} (step {step}, {'EMA' if args.use_ema else 'raw'}) "
              f"res={targs.resolution} block={targs.text_block_size}", flush=True)

    index = build_image_index(args.coco_root)
    ids = select_ids(args.split, args.coco_root, args.karpathy_json)
    missing = [i for i in ids if i not in index]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(ids)} images are not on disk, "
                         f"e.g. {missing[:5]}")
    if args.limit:
        ids = ids[: args.limit]
    if rank == 0:
        print(f"split={args.split}: {len(ids)} images, {world} rank(s), "
              f"{args.n_blocks} blocks x {args.i2t_steps} steps, cfg={args.cfg_scale}",
              flush=True)

    shard = ids[rank::world]
    loader = torch.utils.data.DataLoader(
        CocoImages(shard, index, targs.resolution),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, drop_last=False,
    )

    rows = []
    for bi, (imgs, cids) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        # Without no_grad the VAE encoder's graph pins its activations -- at
        # 512px that is ~19GiB of (B, 128, 512, 512) feature maps held for as
        # long as the latent is referenced, which caps the batch at 8.
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            # i2t conditions on the clean latent, so take the posterior mode
            # rather than a sample: nothing here should be stochastic.
            z_img = image_vae.encode(preprocess_imgs_vae(imgs)).mode().float()
            z_txt, valid = i2t_sample(
                model, z_img, args.n_blocks, num_steps=args.i2t_steps,
                cfg_scale=args.cfg_scale, heun=args.heun,
                eos_id=collator.eos_id, text_vae=text_vae,
            )
            ids_out = decode_text_ids(text_vae, z_txt)
        texts = ids_to_text(collator.tok, ids_out, valid, eos_id=collator.eos_id)
        rows.extend({"cocoid": int(c), "caption": t} for c, t in zip(cids.tolist(), texts))
        if rank == 0 and bi % args.log_every == 0:
            live = torch.cuda.memory_allocated(device) / 2**30
            peak = torch.cuda.max_memory_allocated(device) / 2**30
            print(f"  [{bi + 1}/{len(loader)}] live {live:.1f}GiB peak {peak:.1f}GiB "
                  f"{rows[-1]['caption'][:55]!r}", flush=True)
            torch.cuda.reset_peak_memory_stats(device)

    gathered = [None] * world
    dist.all_gather_object(gathered, rows)
    if rank == 0:
        merged = {r["cocoid"]: r for part in gathered for r in part}
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            for cid in sorted(merged):
                f.write(json.dumps(merged[cid]) + "\n")
        empty = sum(1 for r in merged.values() if not r["caption"].strip())
        print(f"wrote {len(merged)} captions to {args.out} ({empty} empty)", flush=True)
    dist.barrier()
    dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--image-vae-ckpt", type=str,
                   help="override the frozen image VAE recorded in the checkpoint")
    p.add_argument("--image-vae-pretrained", type=str, default="",
                   help="override the pretrained FLUX VAE directory recorded in the checkpoint")
    p.add_argument("--text-vae-path", type=str, default="",
                   help="override the Cola text VAE directory recorded in the checkpoint")
    p.add_argument("--tokenizer-path", type=str, default="",
                   help="override the tokenizer.json path recorded in the checkpoint")
    p.add_argument("--use-ema", action="store_true")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--coco-root", type=str, required=True,
                   help="COCO root holding annotations/, train2017/ and val2017/")
    p.add_argument("--karpathy-json", type=str, default="",
                   help="dataset_coco.json of Karpathy's splits; needed for "
                        "--split karpathy-test and --split union")
    p.add_argument("--split", type=str, default="union",
                   choices=["val2017", "karpathy-test", "union"])
    p.add_argument("--n-blocks", type=int, default=4,
                   help="text blocks to generate; 4 x block_size=16 = 64 tokens, "
                        "ample for COCO-style captions (~13 words)")
    p.add_argument("--i2t-steps", type=int, default=20)
    p.add_argument("--cfg-scale", type=float, default=1.0,
                   help="i2t guidance; training logged 1.5")
    p.add_argument("--heun", action="store_true")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--allow-tf32", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
