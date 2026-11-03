# Evaluation

## Class-conditional ImageNet FID

FID and IS are computed with [torch-fidelity](https://github.com/LTH14/torch-fidelity) against precomputed ImageNet reference statistics, which is the protocol [MAR](https://github.com/LTH14/mar) uses. 

Install the fork, not the PyPI release:

```bash
pip install git+https://github.com/LTH14/torch-fidelity.git@master
```

Both are already in `requirements.txt` and `environment.yml`.

`scripts/eval_fid.sh`: sample 50k PNGs, then score them.

```bash
cd Prior_only
EXP_NAME=prior-only-e2e-f16d32-sitxl1-256 \
CKPT_STEP=4000000 \
VAE_ARCH=f16d32 \
VAE_CKPT=extracted_vae/e2e-f16d32/e2e-f16d32.pt \
LATENTS_STATS=extracted_vae/e2e-f16d32/e2e-f16d32-latents-stats.pt \
  bash scripts/eval_fid.sh
```

Overridable in `eval_fid.sh`: `VAE_ARCH RESOLUTION NPROC OUTPUT_DIR CKPT_STEP CKPT NUM_FID_SAMPLES PER_PROC_BATCH_SIZE MODE NUM_STEPS CFG_SCALE GUIDANCE_LOW GUIDANCE_HIGH WEAK_CKPT AUTOGUIDANCE_SCALE SAMPLE_DIR SEED MODEL`.

### Settings per row

Sampler is SDE with 250 steps throughout. At batch 256 one ImageNet epoch is 5,000 steps, so epochs map onto `CKPT_STEP` directly.


| Row                       | `CKPT_STEP` | Guidance                                                 |
| ------------------------- | ----------- | -------------------------------------------------------- |
| 256, 80 epochs, unguided  | 400000      | `CFG_SCALE=1.0`                                          |
| 256, 480 epochs, unguided | 2400000     | `CFG_SCALE=1.0`                                          |
| 256, 800 epochs, unguided | 4000000     | `CFG_SCALE=1.0`                                          |
| 256, 80 epochs, guided    | 400000      | `CFG_SCALE=1.5`, interval `[0.2, 0.7]`, autoguidance 1.3 |
| 256, 480 epochs, guided   | 2400000     | `CFG_SCALE=1.5`, interval `[0.2, 0.7]`, autoguidance 1.3 |
| 256, 800 epochs, guided   | 4000000     | same as 480                                              |
| 512, 400 epochs, guided   | 2000000     | `CFG_SCALE=2.5`, interval `[0.2, 0.8]`, no autoguidance  |

The 256 rows decode with the decoder-finetuned VAE in `extracted_vae/e2e-f16d32`. The 512 row decodes with the end-to-end VAE before the finetune, which the 512 recipe in the README extracts to `extracted_vae/e2e-f16d32-nodecft`; pass that as `VAE_CKPT` / `LATENTS_STATS` for it.




## Sampling for visualization

To look at samples rather than score them, use `generate.py` with a small `--num-fid-samples`. From a prior-only checkpoint:

```bash
cd Prior_only
torchrun --standalone --nproc_per_node=1 generate.py \
    --ckpt=exps/<run>/checkpoints/4000000.pt \
    --model=SiT-XL/2 --resolution=256 \
    --vae-type=e2e --vae-arch=f8d4 \
    --vae-ckpt=extracted_vae/e2e-f8d4/e2e-f8d4.pt \
    --latents-stats=extracted_vae/e2e-f8d4/e2e-f8d4-latents-stats.pt \
    --num-fid-samples=64 --per-proc-batch-size=32 --cfg-scale=1.5 \
    --label-sampling=random
```

`--label-sampling=random` is needed because 64 does not divide across the 1,000 classes; balanced sampling is only meaningful at the full 50k.

## VAE reconstruction quality

`eval_recon.py` reports rFID, PSNR, SSIM and LPIPS for a trained VAE, independent of the diffusion model, following [latent-diffusion](https://github.com/CompVis/latent-diffusion): posterior-**mean** reconstructions of the 50k ImageNet validation images, with rFID measured between those reconstructions and the same validation images.

```bash
python eval_recon.py \
    --vae-arch f16d32 \
    --vae-ckpt pretrained/extracted_vae/e2e-f16d32/e2e-f16d32.pt \
    --data-dir /data/imagenet/val \
    --out-dir recon_eval/e2e-f16d32
```

`--data-dir` is the ImageNet validation set laid out as `<root>/<synset>/*.JPEG`.

### Multi-round reconstruction (`1x`, `3x`, `5x`, `10x`, `20x`)

To evaluate iterative encode–decode stability over all 50,000 ImageNet validation images at 512x512 (online Inception moments without writing 300k PNGs to disk):

```bash
torchrun --standalone --nproc_per_node=8 scripts/eval_vae_multiround_full.py \
    --label e2e-flux2-s3-20k \
    --vae-arch flux2 \
    --vae vae/e2e-vae-flux2-sitxl2-auxkl2p5e8-s3-20k/e2e-vae-flux2-sitxl2-auxkl2p5e8-s3-20k.safetensors \
    --data /data/imagenet/val \
    --resolution 512 \
    --rounds 1 3 5 10 20 \
    --out recon_eval/multiround
```

## Text-to-image

GenEval and DPG-Bench have their own harness and their own heavy dependencies; see[T2I_EVAL.md](T2I_EVAL.md). Sampling there is Euler, 50 steps, CFG 7.0 on the interval `[0.35, 1.0]`, and the reported scores use no prompt rewriting.

