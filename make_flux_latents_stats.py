# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

"""Create a `-latents-stats.pt` for the *original* (non-e2e) FLUX.1 VAE.

The frozen-VAE prior-only pipeline
(`Prior_only/scripts/train_t2i_prior_only.sh`) expects a
`<vae>-latents-stats.pt` file next to the VAE checkpoint, holding per-channel
`latents_scale` (= 1/sqrt(var)) and `latents_bias` (= mean). For the e2e-trained
VAE these come from the SiT BatchNorm (see `save_vae_weights.py`). For the
frozen-original-VAE baseline we instead use FLUX's published scalar shift/scale,
so the latent normalization matches how FLUX itself standardizes latents:

    z_norm = (z - shift_factor) * scaling_factor

Two modes:
  * default: write the constant FLUX shift/scale (fast, no data needed);
  * --from-data: estimate per-channel mean/std over a sample of images encoded
    by the pretrained FLUX VAE (recommended for the fairest baseline).

Usage:
  python make_flux_latents_stats.py \
      --vae-path pretrained/flux-vae \
      --out pretrained/extracted_vae/flux-orig/flux-orig-latents-stats.pt
"""

import argparse
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae-path", type=str, required=True,
                    help="Pretrained FLUX.1 `vae` diffusers folder.")
    ap.add_argument("--out", type=str, required=True,
                    help="Output path for the -latents-stats.pt file.")
    ap.add_argument("--channels", type=int, default=16)
    ap.add_argument("--from-data", action="store_true",
                    help="Estimate per-channel stats from real images instead of the scalar shift/scale.")
    ap.add_argument("--images", type=str, default=None,
                    help="Comma-separated parquet roots (required with --from-data).")
    ap.add_argument("--num-images", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--resolution", type=int, default=256)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    if not args.from_data:
        from models.autoencoder import FluxVAE

        latents_scale = torch.full((1, args.channels, 1, 1), float(FluxVAE.scaling_factor))
        latents_bias = torch.full((1, args.channels, 1, 1), float(FluxVAE.shift_factor))
        torch.save({"latents_scale": latents_scale, "latents_bias": latents_bias}, args.out)
        print(f"[make_flux_latents_stats] wrote constant shift/scale stats -> {args.out}")
        return

    # --- data-estimated per-channel stats ---
    assert args.images is not None, "--from-data requires --images"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from models.autoencoder import FluxVAE
    from dataset import ParquetT2IDataset
    from torch.utils.data import DataLoader

    vae = FluxVAE(pretrained_path=args.vae_path).to(device).eval()
    vae.requires_grad_(False)
    ds = ParquetT2IDataset(
        [r for r in args.images.split(",") if r],
        resolution=args.resolution,
        rows_per_file=128,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    n = 0
    csum = torch.zeros(args.channels, dtype=torch.float64)
    csqsum = torch.zeros(args.channels, dtype=torch.float64)
    with torch.no_grad():
        for image, _caption in dl:
            image = image.to(device).float() / 127.5 - 1.0
            z = vae.encode(image).sample()  # (B, C, H, W)
            z = z.permute(1, 0, 2, 3).reshape(args.channels, -1).double().cpu()
            csum += z.sum(dim=1)
            csqsum += (z * z).sum(dim=1)
            n += z.shape[1]
            if n >= args.num_images * (args.resolution // 8) ** 2:
                break
    mean = (csum / n)
    var = (csqsum / n) - mean * mean
    latents_scale = (1.0 / var.sqrt()).float().view(1, args.channels, 1, 1)
    latents_bias = mean.float().view(1, args.channels, 1, 1)
    torch.save({"latents_scale": latents_scale, "latents_bias": latents_bias}, args.out)
    print(f"[make_flux_latents_stats] wrote data-estimated stats (n={n} elems) -> {args.out}")
    print("  per-channel mean:", mean.tolist())
    print("  per-channel std :", var.sqrt().tolist())


if __name__ == "__main__":
    main()
