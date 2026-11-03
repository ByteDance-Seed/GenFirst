# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

"""Extract the VAEs reshaped by a joint end-to-end run, for the frozen stage.

The counterpart of ``save_vae_weights.py`` for ``train_joint_e2e.py``: pulls the
text VAE out of a joint checkpoint, and the image VAE too when that run trained
one (``--image-vae-mode e2e-*``). Latent statistics come from the EMA copy of the
DiT's BatchNorm buffers, the same convention the image extraction uses and the
one ``--image-latents-stats`` expects.

Usage::

    python save_joint_vae_weights.py \
        --joint-ckpt exps/joint-e2e-256-JointMMDiT-L-2/checkpoints/0600000.pt \
        --name e2e-joint-l2 --save-dir extracted_vae

Then point the prior-only stage at it::

    TEXT_VAE_CKPT=extracted_vae/e2e-joint-l2/e2e-joint-l2-text-vae.pt \
        PHASE=1 bash Prior_only/scripts/train_joint_prior_only.sh
"""

import argparse
import os

import torch


def main(args):
    ckpt = torch.load(args.joint_ckpt, map_location="cpu")
    targs = ckpt.get("args")
    step = ckpt.get("steps", -1)
    print(f"loaded {args.joint_ckpt} (step {step})")
    if targs is not None:
        print(f"  model={getattr(targs, 'model', '?')} "
              f"image_vae_mode={getattr(targs, 'image_vae_mode', '?')} "
              f"freeze_text_vae={getattr(targs, 'freeze_text_vae', '?')}")
        if getattr(targs, "freeze_text_vae", False):
            print("  WARNING: this run froze the text VAE, so the extracted weights are "
                  "identical to the pretrained ones.")

    out_dir = os.path.join(args.save_dir, args.name)
    os.makedirs(out_dir, exist_ok=True)
    written = []

    text_vae = ckpt["text_vae"]
    path = os.path.join(out_dir, f"{args.name}-text-vae.pt")
    torch.save(text_vae, path)
    written.append(path)

    # The image VAE only moved if that run trained it; otherwise the frozen stage
    # should keep using whatever checkpoint fed the joint run.
    if ckpt.get("image_vae") is not None:
        path = os.path.join(out_dir, f"{args.name}.pt")
        torch.save(ckpt["image_vae"], path)
        written.append(path)

        # `bn` tracked the latent distribution while the encoder drifted, so the
        # EMA buffers are the statistics the frozen stage has to normalize with.
        src = ckpt["ema" if args.stats_from_ema else "model"]
        mean, var = src["bn.running_mean"], src["bn.running_var"]
        c = mean.shape[0]
        stats = {
            "latents_scale": var.rsqrt().view(1, c, 1, 1),
            "latents_bias": mean.view(1, c, 1, 1),
        }
        path = os.path.join(out_dir, f"{args.name}-latents-stats.pt")
        torch.save(stats, path)
        written.append(path)
        print(f"  latent stats from {'ema' if args.stats_from_ema else 'model'}: "
              f"scale mean {stats['latents_scale'].mean():.4f}, "
              f"bias mean {stats['latents_bias'].mean():+.4f}")

        if ckpt.get("vae_loss_fn") is not None:
            disc = {
                k[len("discriminator."):]: v
                for k, v in ckpt["vae_loss_fn"].items()
                if k.startswith("discriminator.")
            }
            if disc:
                path = os.path.join(out_dir, f"{args.name}-discriminator-ckpt.pt")
                torch.save(disc, path)
                written.append(path)
    else:
        print("  no image_vae in this checkpoint (frozen image branch); "
              "reuse the image VAE the joint run was given.")

    for p in written:
        print(f"wrote {p}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--joint-ckpt", type=str, required=True)
    p.add_argument("--name", type=str, default="e2e-joint-vae")
    p.add_argument("--save-dir", type=str, default="extracted_vae")
    p.add_argument("--stats-from-ema", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
