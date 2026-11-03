# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

"""Extract a standalone VAE (+ its latent statistics) from a training checkpoint.

Accepts either kind of checkpoint:

  * an end-to-end checkpoint (train_e2e.py), where the latent statistics live in
    the SiT EMA's BatchNorm buffers, or
  * a decoder-finetune checkpoint (train_decoder_ft.py), which stores them
    directly, having carried them over from the end-to-end run it started from.

The statistics must travel with the VAE: the prior-only stage, sampling and
evaluation all have to use the same ones, or latents get denormalized wrongly and
the samples look plausible but score badly.
"""
import argparse
import os

import torch


def latents_stats_from_ckpt(ckpt, path, override=None):
    # An explicit file wins, for checkpoints that predate the statistics being
    # embedded. A decoder-finetune run also writes them next to its checkpoints as
    # `latents-stats.pt`, which is the usual source for that case.
    if override is not None:
        return (
            torch.load(override, map_location="cpu", weights_only=False),
            f"read from {override}",
        )

    # A decoder-finetune checkpoint carries the statistics verbatim; its encoder was
    # frozen, so they are still the ones the end-to-end run produced.
    stats = ckpt.get("latents_stats")
    if stats is not None:
        return stats, "carried over from the source end-to-end run"

    # An end-to-end checkpoint keeps them in the SiT EMA's BatchNorm buffers.
    ema = ckpt.get("ema")
    if ema is not None and "bn.running_mean" in ema:
        bn_running_mean = ema["bn.running_mean"]
        bn_running_var = ema["bn.running_var"]
        in_channels = bn_running_mean.shape[0]
        return {
            "latents_scale": bn_running_var.rsqrt().view(1, in_channels, 1, 1),
            "latents_bias": bn_running_mean.view(1, in_channels, 1, 1),
        }, "derived from the SiT EMA BatchNorm buffers"

    raise KeyError(
        f"No latent statistics in {path}: expected either a 'latents_stats' entry "
        "(decoder-finetune checkpoint) or 'ema' BatchNorm buffers (end-to-end "
        "checkpoint). Without them the VAE cannot be used for sampling. Pass "
        "--latents-stats=<file> to supply them explicitly."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e-ckpt", type=str, required=True,
                        help="An end-to-end (train_e2e.py) or decoder-finetune "
                             "(train_decoder_ft.py) checkpoint.")
    parser.add_argument("--vae-name", type=str, default="e2e-vae",
                        help="Name of the saved VAE model")
    parser.add_argument("--save-dir", type=str, default="exps",
                        help="Directory to save the extracted VAE weights")
    parser.add_argument("--use-vae-ema", action="store_true",
                        help="Take the VAE's EMA weights instead of the live ones "
                             "(decoder-finetune checkpoints only). Note that over a "
                             "10k-step finetune the EMA still carries roughly a third "
                             "of the pre-finetune decoder, so the live weights are "
                             "usually what you want.")
    parser.add_argument("--latents-stats", type=str, default=None,
                        help="Latent statistics file, for checkpoints that do not "
                             "carry them. A decoder-finetune run writes one as "
                             "`latents-stats.pt` beside its checkpoints directory.")
    args = parser.parse_args()

    ckpt = torch.load(args.e2e_ckpt, map_location="cpu", weights_only=False)

    if args.use_vae_ema:
        # Older decoder-finetune checkpoints spell this key the other way round.
        key = next((k for k in ("vae_ema", "ema_vae") if k in ckpt), None)
        if key is None:
            raise KeyError(f"--use-vae-ema given but {args.e2e_ckpt} has no EMA VAE")
        vae_weights = ckpt[key]
    else:
        vae_weights = ckpt["vae"]

    latents_stats, stats_origin = latents_stats_from_ckpt(
        ckpt, args.e2e_ckpt, override=args.latents_stats)
    discriminator_weights = ckpt["discriminator"]

    save_dir = os.path.join(args.save_dir, args.vae_name)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(vae_weights, os.path.join(save_dir, f"{args.vae_name}.pt"))
    torch.save(latents_stats, os.path.join(save_dir, f"{args.vae_name}-latents-stats.pt"))
    torch.save(discriminator_weights, os.path.join(save_dir, f"{args.vae_name}-discriminator-ckpt.pt"))
    print(f"Wrote {save_dir}/ (latent statistics {stats_origin})")
