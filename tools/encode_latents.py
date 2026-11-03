#!/usr/bin/env python3
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

"""Pre-compute VAE posterior moments for the prior-only stage.

The output contract is fixed by ``CustomDataset`` in ``Prior_only/dataset.py``:

    <dest>/00000/img00000000.npy    float array of shape (2C, H', W'), namely
                                    concat([posterior.mean, posterior.std])
    ...
    <dest>/dataset.json             {"labels": [[relpath, class_index], ...]}

``Prior_only/train_prior_only.py:sample_posterior`` splits each array back into
``mean, std`` and draws ``z = mean + std * eps``, so the second half must be the
standard deviation, not the log-variance.

Feature files mirror the relative layout of the source image tree with the
suffix swapped to ``.npy``. The loader pairs images and features by sorting each
list independently, so identical stems keep the two in lockstep.

The encoder is this repository's own VAE (``models/autoencoder.py``); there is
no pretrained-encoder zoo here. Run under ``accelerate launch`` for multi-GPU;
each rank writes its own shard of ``.npy`` files and only the label table is
gathered.

Example:
  accelerate launch tools/encode_latents.py \\
      --vae-type=e2e --vae-arch=f16d32 \\
      --vae-ckpt=exps/my-vae/my-vae.pt \\
      --source=data/imagenet256/images \\
      --dest=data/imagenet256/vae-my-vae
"""

import argparse
import gc
import json
import os
import sys

import accelerate
import numpy as np
import torch
from accelerate.utils import gather_object
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
from image_prep import (  # noqa: E402
    LABEL_ITEM,
    discover,
    image_suffixes,
    labels_from_layout,
    labels_from_sidecar,
)
from models.autoencoder import vae_models  # noqa: E402
from utils import preprocess_imgs_vae  # noqa: E402


class ImageTree(Dataset):
    """Walk a directory of already-cropped images, yielding (pixels, label, rel)."""

    def __init__(self, root):
        self.root = root
        self.rels = discover(root, image_suffixes())
        if not self.rels:
            raise SystemExit(f"no images found under {root}")
        table = labels_from_sidecar(root) or labels_from_layout(self.rels)
        self.labels = [int(table.get(rel, -1)) for rel in self.rels]

    def __len__(self):
        return len(self.rels)

    def __getitem__(self, idx):
        rel = self.rels[idx]
        with Image.open(os.path.join(self.root, rel)) as handle:
            pixels = np.array(handle.convert("RGB"))
        # (H, W, 3) uint8 -> (3, H, W), matching what preprocess_imgs_vae expects.
        pixels = torch.from_numpy(pixels.transpose(2, 0, 1).copy())
        return pixels, self.labels[idx], rel


def load_vae(arch, ckpt_path, device):
    vae = vae_models[arch]()
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    vae.load_state_dict(state_dict)
    vae.to(device).eval()
    del state_dict
    gc.collect()
    torch.cuda.empty_cache()
    return vae


def feature_name(rel):
    return os.path.splitext(rel)[0] + ".npy"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, help="directory of cropped source images")
    parser.add_argument("--dest", required=True, help="output directory for the .npy moments")
    parser.add_argument("--vae-ckpt", required=True, help="extracted VAE weights (.pt)")
    parser.add_argument("--vae-arch", required=True, choices=sorted(vae_models))
    parser.add_argument(
        "--vae-type",
        default="e2e",
        choices=["e2e"],
        help="accepted for compatibility with the pipeline scripts; only the "
             "in-repo VAE is supported",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", "--workers-per-gpu", dest="num_workers",
                        type=int, default=8)
    args = parser.parse_args()

    accelerator = accelerate.Accelerator()
    device = accelerator.device

    vae = load_vae(args.vae_arch, args.vae_ckpt, device)
    dataset = ImageTree(args.source)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    vae, loader = accelerator.prepare(vae, loader)

    os.makedirs(args.dest, exist_ok=True)
    label_rows = []

    for pixels, labels, rels in tqdm(
        loader, desc="encoding", disable=not accelerator.is_local_main_process
    ):
        batch = preprocess_imgs_vae(pixels.to(device))
        with torch.no_grad():
            posterior, _, _ = vae(batch, return_recon=False)
            # sample_posterior() chunks this back into (mean, std).
            moments = torch.cat([posterior.mean, posterior.std], dim=1)

        moments = moments.float().cpu().numpy()
        labels = labels.tolist()

        # Each rank owns a disjoint slice of the dataset, so writing straight to
        # disk needs no cross-rank coordination.
        for array, rel, label in zip(moments, rels, labels):
            out_rel = feature_name(rel)
            out_path = os.path.join(args.dest, out_rel)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.save(out_path, array)
            label_rows.append([out_rel, label])

    accelerator.wait_for_everyone()
    all_rows = gather_object(label_rows)

    if accelerator.is_main_process:
        # Ranks can overlap on the tail batch when the sampler pads; keep the
        # first row per filename and sort so the table is reproducible.
        merged = {}
        for rel, label in all_rows:
            merged.setdefault(rel, label)
        rows = sorted(merged.items())
        if any(label < 0 for _, label in rows):
            raise SystemExit("some images had no resolvable class label")
        with open(os.path.join(args.dest, LABEL_ITEM), "w") as handle:
            json.dump({"labels": [[rel, label] for rel, label in rows]}, handle)
        print(f"wrote {len(rows)} feature files and {LABEL_ITEM} under {args.dest}")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
