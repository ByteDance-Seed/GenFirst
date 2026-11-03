# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

"""Distributed full-set, multi-round VAE reconstruction evaluation.

Computes single-round and iterative (e.g., 1x, 3x, 5x, 10x, 20x) reconstruction
metrics over the 50,000 ImageNet validation images. Instead of writing 300k PNGs
to disk, this script accumulates Inception feature moments online and only saves
a small deterministic visual subset.

All pixel metrics (MSE, RMSE, MAE, PSNR, SSIM) use float reconstructions in [0, 1].
rFID uses standard uint8 quantization and torch-fidelity Inception features.
Each reconstruction round feeds its float output (clamped to [-1, 1]) directly into
the next encode/decode pass without intermediate uint8 quantization, and every round
is evaluated against the original input image.

Usage:
    torchrun --standalone --nproc_per_node=8 scripts/eval_vae_multiround_full.py \
        --label e2e-flux2-s3-20k \
        --vae-arch flux2 \
        --vae pretrained/extracted_vae/e2e-vae-flux2-sitxl2-auxkl2p5e8-s3-20k/e2e-vae-flux2-sitxl2-auxkl2p5e8-s3-20k.safetensors \
        --data /path/to/imagenet/val \
        --resolution 512 \
        --rounds 1 3 5 10 20 \
        --out recon_eval/multiround
"""

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loss.lpips import LPIPS  # noqa: E402
from models.autoencoder import vae_models  # noqa: E402


class ImageNetValShard(Dataset):
    def __init__(
        self,
        root: str,
        resolution: int,
        rank: int,
        world_size: int,
        max_images: int | None = None,
    ):
        paths = []
        for class_name in sorted(os.listdir(root)):
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
            for name in sorted(os.listdir(class_dir)):
                if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    paths.append(os.path.join(class_dir, name))
        if not paths:
            # Support flat image directory as well
            for name in sorted(os.listdir(root)):
                if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    paths.append(os.path.join(root, name))
        if max_images is not None:
            paths = paths[:max_images]
        self.total_size = len(paths)
        self.items = list(enumerate(paths))[rank::world_size]
        self.resolution = resolution

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        global_index, path = self.items[item]
        image = Image.open(path).convert("RGB")
        width, height = image.size
        scale = self.resolution / min(width, height)
        image = image.resize(
            (
                max(self.resolution, round(width * scale)),
                max(self.resolution, round(height * scale)),
            ),
            Image.Resampling.BICUBIC,
        )
        width, height = image.size
        left = (width - self.resolution) // 2
        top = (height - self.resolution) // 2
        image = image.crop(
            (left, top, left + self.resolution, top + self.resolution)
        )
        tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)
        return global_index, tensor


def distributed_context():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def load_vae(arch: str, path: str, device: torch.device):
    if arch in ("flux", "flux2") and (os.path.isdir(path) or path.endswith(".safetensors")):
        vae = vae_models[arch](pretrained_path=path)
    else:
        vae = vae_models[arch]()
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["vae"] if isinstance(checkpoint, dict) and "vae" in checkpoint else checkpoint
        vae.load_state_dict(state_dict, strict=False)
        del checkpoint, state_dict
    return vae.to(device).eval().requires_grad_(False)


def save_visual_subset(images, indices, directory, limit):
    os.makedirs(directory, exist_ok=True)
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    for image, index in zip(images, indices.tolist()):
        if index < limit:
            Image.fromarray(image).save(
                os.path.join(directory, f"{index:05d}.png"), compress_level=1
            )


def reduce_sum(tensor, world_size):
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)


def moments_to_fid(sums, outers, counts, real_slot, fake_slot):
    from torchmetrics.image.fid import _compute_fid

    def stats(slot):
        count = counts[slot]
        mean = sums[slot] / count
        covariance = (
            outers[slot] - count * mean.unsqueeze(1).mm(mean.unsqueeze(0))
        ) / (count - 1)
        return mean, covariance

    real_mean, real_covariance = stats(real_slot)
    fake_mean, fake_covariance = stats(fake_slot)
    return _compute_fid(
        real_mean, real_covariance, fake_mean, fake_covariance
    ).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--vae-arch", default="flux2", choices=list(vae_models.keys()))
    parser.add_argument("--vae", required=True, help="Path to VAE weights (.safetensors, .pt, or diffusers vae folder).")
    parser.add_argument("--data", required=True, help="ImageNet validation directory.")
    parser.add_argument("--out", default="recon_eval/multiround")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--rounds", type=int, nargs="+", default=[1, 3, 5, 10, 20]
    )
    parser.add_argument("--visual-samples", type=int, default=64)
    args = parser.parse_args()

    rank, local_rank, world_size = distributed_context()
    device = torch.device("cuda", local_rank)
    rounds = sorted(set(args.rounds))
    selected = set(rounds)
    round_to_row = {round_index: i for i, round_index in enumerate(rounds)}
    run_dir = os.path.join(args.out, args.label)

    if rank == 0:
        shutil.rmtree(run_dir, ignore_errors=True)
        os.makedirs(run_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    dataset = ImageNetValShard(
        args.data,
        args.resolution,
        rank=rank,
        world_size=world_size,
        max_images=args.max_images,
    )
    if dataset.total_size != 50_000 and rank == 0:
        print(
            f"[warning] expected 50,000 ImageNet val images, "
            f"found {dataset.total_size}",
            flush=True,
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    vae = load_vae(args.vae_arch, args.vae, device)
    lpips_fn = LPIPS().to(device).eval().requires_grad_(False)
    from torchmetrics.functional import (
        structural_similarity_index_measure as ssim_fn,
    )
    from torchmetrics.image.fid import FrechetInceptionDistance

    fid_wrapper = FrechetInceptionDistance(
        feature=2048, normalize=False
    ).to(device).eval()
    inception = fid_wrapper.inception

    metric_names = ["mse", "mae", "psnr", "ssim", "lpips"]
    totals = torch.zeros(
        len(rounds), len(metric_names), dtype=torch.float64, device=device
    )
    image_count = torch.zeros(1, dtype=torch.float64, device=device)

    feature_dim = 2048
    fid_sums = torch.zeros(
        len(rounds) + 1, feature_dim, dtype=torch.float64, device=device
    )
    fid_outers = torch.zeros(
        len(rounds) + 1,
        feature_dim,
        feature_dim,
        dtype=torch.float64,
        device=device,
    )
    fid_counts = torch.zeros(
        len(rounds) + 1, dtype=torch.float64, device=device
    )

    def update_fid(slot, images_uint8):
        features = inception(images_uint8).double()
        if features.ndim == 1:
            features = features.unsqueeze(0)
        fid_sums[slot] += features.sum(0)
        fid_outers[slot].addmm_(features.t(), features)
        fid_counts[slot] += features.shape[0]

    start_time = time.time()
    local_seen = 0
    with torch.inference_mode():
        for batch_index, (indices, original_u8) in enumerate(loader):
            indices = indices.to(device, non_blocking=True)
            original_u8 = original_u8.to(device, non_blocking=True)
            original = original_u8.float() / 127.5 - 1.0
            current = original
            update_fid(0, original_u8)
            save_visual_subset(
                original_u8,
                indices,
                os.path.join(run_dir, "visuals", "reference"),
                args.visual_samples,
            )

            for round_index in range(1, max(rounds) + 1):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    posterior = vae.encode(current)
                    current = vae.decode(posterior.mode()).sample
                current = current.float().clamp(-1.0, 1.0)
                if round_index not in selected:
                    continue

                row = round_to_row[round_index]
                reference_01 = (original + 1.0) / 2.0
                current_01 = (current + 1.0) / 2.0
                per_image_mse = (reference_01 - current_01).square().mean(
                    (1, 2, 3)
                )
                per_image_mae = (reference_01 - current_01).abs().mean(
                    (1, 2, 3)
                )
                totals[row, 0] += per_image_mse.double().sum()
                totals[row, 1] += per_image_mae.double().sum()
                totals[row, 2] += (
                    -10.0 * torch.log10(per_image_mse.clamp_min(1e-12))
                ).double().sum()
                totals[row, 3] += ssim_fn(
                    current_01,
                    reference_01,
                    data_range=1.0,
                    reduction="sum",
                ).double()
                totals[row, 4] += lpips_fn(original, current).sum().double()

                current_u8 = (
                    (current + 1.0)
                    .mul(127.5)
                    .round()
                    .clamp(0, 255)
                    .to(torch.uint8)
                )
                update_fid(row + 1, current_u8)
                save_visual_subset(
                    current_u8,
                    indices,
                    os.path.join(
                        run_dir, "visuals", f"round_{round_index:02d}"
                    ),
                    args.visual_samples,
                )

            batch_size = original.shape[0]
            local_seen += batch_size
            image_count += batch_size
            if rank == 0 and (batch_index % 20 == 0 or local_seen == len(dataset)):
                approximate_seen = min(
                    dataset.total_size, local_seen * world_size
                )
                elapsed = (time.time() - start_time) / 60
                print(
                    f"[progress] {approximate_seen}/{dataset.total_size} "
                    f"({elapsed:.1f} min)",
                    flush=True,
                )

    for tensor in (
        totals,
        image_count,
        fid_sums,
        fid_outers,
        fid_counts,
    ):
        reduce_sum(tensor, world_size)

    if rank == 0:
        count = int(image_count.item())
        results = {
            "protocol": {
                "dataset": args.data,
                "num_images": count,
                "resolution": args.resolution,
                "rounds": rounds,
                "posterior": "mode",
                "pixel_range": "[0,1]",
                "round_input": "previous float reconstruction, clamped to [-1,1]",
                "intermediate_uint8_quantization": False,
                "distributed_world_size": world_size,
            },
            "metrics": {},
        }
        for round_index, row in round_to_row.items():
            values = (totals[row] / count).tolist()
            round_metrics = dict(zip(metric_names, values))
            round_metrics["rmse"] = round_metrics["mse"] ** 0.5
            round_metrics["rfid"] = moments_to_fid(
                fid_sums, fid_outers, fid_counts, 0, row + 1
            )
            results["metrics"][str(round_index)] = round_metrics
            print(
                f"[round {round_index:02d}] "
                + " ".join(
                    f"{name}={value:.8f}"
                    for name, value in round_metrics.items()
                ),
                flush=True,
            )

        results["elapsed_minutes"] = (time.time() - start_time) / 60
        output_json = os.path.join(run_dir, "metrics.json")
        with open(output_json, "w") as file:
            json.dump(results, file, indent=2)
        print(f"[done] {output_json}", flush=True)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
