# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

"""VAE reconstruction metrics: rFID, PSNR, SSIM, LPIPS.

Follows the protocol used by latent-diffusion (CompVis) and reported in the paper:
posterior-**mean** reconstructions of the 50k ImageNet validation images, with rFID
measured between those reconstructions and the same validation images (not against
a training-set reference).

Because rFID needs both image sets on disk, this writes 2 x N PNGs under
--out-dir (roughly 15 GB for 50k at 256x256). Pass --no-keep-images to delete them
once the metrics are computed.

Usage:
    python eval_recon.py \\
        --vae-arch f8d4 \\
        --vae-ckpt pretrained/extracted_vae/e2e-f8d4/e2e-f8d4.pt \\
        --data-dir /data/imagenet/val \\
        --out-dir recon_eval/e2e-f8d4

    # the initialization, for the comparison the numbers only mean something against
    python eval_recon.py --vae-arch f8d4 --vae-ckpt pretrained/sdvae/sdvae-f8d4.pt ...

    # the Qwen3-VL-tower autoencoder of docs/UND.md, whose geometry is read out of
    # the training checkpoint (needs transformers >= 4.57)
    python eval_recon.py --vae-arch qwenvit \\
        --vae-ckpt exps/<run>/checkpoints/0100000.pt --data-dir /data/imagenet/val
"""
import argparse
import os
import shutil

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ImageFolderDataset
from models.autoencoder import vae_models
from models.qwen_vit_ae import DEFAULT_QWEN3VL


def build_vae(args, device):
    if args.vae_arch == "qwenvit":
        from models.qwen_vit_ae import load_qwenvit_vae

        # The decoder is built for a fixed token grid, so it has to be the one
        # this resolution produces: the tower is patch-16 and each token becomes a
        # merge x merge latent block, i.e. a 256px image is a 16x16 grid.
        vae, _ = load_qwenvit_vae(
            args.vae_ckpt, decoder_grid=args.resolution // 16, device=device,
            fallback=dict(vae_vit_path=args.vae_vit_path),
        )
    elif args.vae_arch == "flux2":
        vae = vae_models["flux2"](pretrained_path=args.vae_ckpt).to(device)
    elif args.vae_arch == "flux" and os.path.isdir(args.vae_ckpt):
        # A stock diffusers FLUX VAE is a directory. End-to-end checkpoints
        # extracted by save_vae_weights.py are bare state_dict .pt files, just
        # like the class-conditional VAE architectures below.
        vae = vae_models["flux"](pretrained_path=args.vae_ckpt).to(device)
    else:
        vae = vae_models[args.vae_arch]().to(device)
        state = torch.load(
            args.vae_ckpt, map_location="cpu", weights_only=False)
        # Extracted checkpoints are a bare state_dict; training checkpoints are not.
        if isinstance(state, dict) and "vae" in state:
            state = state["vae"]
        missing, unexpected = vae.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[eval_recon] load_state_dict missing={len(missing)} "
                  f"unexpected={len(unexpected)}")
            if len(missing) > 10:
                raise RuntimeError(
                    f"{len(missing)} missing keys -- is --vae-arch ({args.vae_arch}) "
                    "the architecture this checkpoint was trained with?"
                    )
    vae.eval()
    vae.requires_grad_(False)
    return vae


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vae-arch", type=str, default="f8d4",
                   choices=list(vae_models.keys()) + ["qwenvit"])
    p.add_argument("--vae-ckpt", type=str, required=True,
                   help="Extracted VAE weights, or a FLUX VAE directory for --vae-arch flux.")
    p.add_argument("--vae-vit-path", type=str, default=DEFAULT_QWEN3VL,
                   help="for --vae-arch qwenvit: the tower to build, used only when "
                        "the checkpoint is an extraction and carries no training args")
    p.add_argument("--data-dir", type=str, required=True,
                   help="ImageNet validation root, laid out as <root>/<synset>/*.JPEG.")
    p.add_argument("--out-dir", type=str, default="recon_eval")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--num-images", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--keep-images", action=argparse.BooleanOptionalAction, default=True,
                   help="Keep the written PNGs (needed to re-score without re-running).")
    p.add_argument("--lpips-net", type=str, default="vgg", choices=["vgg", "alex"],
                   help="latent-diffusion's perceptual metric is VGG-based.")
    args = p.parse_args()

    # No seeding needed: posterior-mean reconstruction is deterministic and the
    # loader is unshuffled, so this run is reproducible as written.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    import lpips
    import torch_fidelity
    from torchmetrics.image import (
        PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure,
        )

    vae = build_vae(args, device)
    lpips_fn = lpips.LPIPS(net=args.lpips_net).to(device)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    dataset = ImageFolderDataset(args.data_dir, resolution=args.resolution)
    n = min(args.num_images, len(dataset))
    if n < args.num_images:
        print(f"!!! only {n} images under {args.data_dir}, asked for {args.num_images}")
    loader = DataLoader(
        torch.utils.data.Subset(dataset, range(n)),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False,
        )

    real_dir = os.path.join(args.out_dir, "real")
    recon_dir = os.path.join(args.out_dir, "recon")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)

    lpips_total, seen = 0.0, 0
    idx = 0
    for images, _ in tqdm(loader, desc="reconstructing"):
        # uint8 CHW -> [-1, 1], which is what the VAE was trained on.
        x01 = images.to(device).float() / 255.0
        x = x01 * 2.0 - 1.0

        with torch.no_grad():
            posterior = vae.encode(x)
            # Posterior MEAN, not a sample: this is the protocol the paper reports.
            recon = vae.decode(posterior.mean).sample

        recon01 = ((recon + 1.0) / 2.0).clamp(0, 1)
        psnr.update(recon01, x01)
        ssim.update(recon01, x01)
        with torch.no_grad():
            # lpips expects [-1, 1]
            d = lpips_fn(recon01 * 2 - 1, x01 * 2 - 1)
        lpips_total += d.sum().item()
        seen += x01.size(0)

        real_np = (x01 * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        recon_np = (recon01 * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for r, c in zip(real_np, recon_np):
            Image.fromarray(r).save(os.path.join(real_dir, f"{idx:06d}.png"))
            Image.fromarray(c).save(os.path.join(recon_dir, f"{idx:06d}.png"))
            idx += 1

    print(f"\n==> {seen} images; scoring rFID between {recon_dir} and {real_dir}")
    metrics = torch_fidelity.calculate_metrics(
        input1=recon_dir,
        input2=real_dir,
        cuda=torch.cuda.is_available(),
        isc=False,
        fid=True,
        kid=False,
        prc=False,
        verbose=False,
    )

    print(f"\nVAE      {args.vae_ckpt}")
    print(f"images   {seen} @ {args.resolution}x{args.resolution}")
    print(f"rFID     {metrics['frechet_inception_distance']:.4f}")
    print(f"PSNR     {float(psnr.compute()):.4f}")
    print(f"SSIM     {float(ssim.compute()):.4f}")
    print(f"LPIPS    {lpips_total / max(seen, 1):.4f}  ({args.lpips_net})")

    if not args.keep_images:
        shutil.rmtree(real_dir)
        shutil.rmtree(recon_dir)
        print(f"\nremoved the PNG folders under {args.out_dir}")


if __name__ == "__main__":
    main()
