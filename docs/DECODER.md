# Decoder finetune

A short decoder-only finetune, run after the end-to-end stage. It is optional and cheap — 10k steps against 600k — and worth about 0.2 unguided gFID on ImageNet-256 (3.00 → 2.79 at 80 epochs). Every ImageNet-256 number reported for this repo uses it.

```bash
INIT_CKPT=exps/e2e-f8d4-sitxl2-256-phase2/checkpoints/0600000.pt \
  bash scripts/train_decoder_ft.sh
```

## What it trains


|            |                                                     |
| ---------- | --------------------------------------------------- |
| trainable  | `vae.decoder`, `vae.post_quant_conv`, discriminator |
| frozen     | `vae.encoder`, `vae.quant_conv`                     |
| loss       | L1 + LPIPS + GAN — no SiT, no REPA alignment        |
| steps      | 10k, at global batch 256                            |
| resolution | 256 only                                            |


