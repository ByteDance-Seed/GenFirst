# Understanding in the end-to-end stage

An objective for the end-to-end stage that asks the latent to serve understanding as well as generation. 

## Quick start

The encoder is Qwen3-VL's vision tower (`--vae qwenvit`) rather than a conv VAE.

```bash
PHASE=0 DATA=/path/to/parquet_roots bash scripts/train_qwenvit_e2e.sh
PHASE=1 UND=nll DATA=/path/to/parquet_roots bash scripts/train_qwenvit_e2e.sh
PHASE=2 UND=nll DATA=/path/to/parquet_roots bash scripts/train_qwenvit_e2e.sh
```

`UND=off` on phases 1 and 2 is the baseline setting. Needs `transformers >= 4.57`, which the pin in `requirements.txt` satisfies; see [the dependency note](#the-qwen3-vl-dependency) at the end.

## The two objectives

`--und-type` selects one. They ask different questions and are not redundant.


|                  | `siglip`                                                      | `nll` (default)                                              |
| ---------------- | ------------------------------------------------------------- | ------------------------------------------------------------ |
| Supervision      | frozen CLIP caption embeddings                                | frozen Qwen3-VL language model                               |
| Question         | can the latent be told apart from other captions in the batch | can a language model read the caption off it, token by token |
| Cost             | one pooling head, full batch                                  | a 2B forward **and** backward                                |
| Extra dependency | none                                                          | `transformers >= 4.57`                                       |
| Read this metric | `und/i2t_r1`                                                  | `und/pmi`                                                    |


## What to watch


| Metric                       | Meaning                                                                                                         |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `und/i2t_r1`                 | fraction of images whose own caption is the nearest text across all ranks. Compare against `und/chance_r1`.     |
| `und/chance_r1`              | chance level, i.e. `1 / (negatives + 1)`.                                                                       |
| `und/pos_sim`, `und/neg_sim` | the same signal without the argmax. Both high with a gap near zero means the latent collapsed to one direction. |
| `und/grad_norm`              | the understanding gradient, for calibration against `grad_norm_vae`.                                            |
| `und/pmi`                    | (`nll` only) image-dependent part of the caption likelihood, in nats/token.                                     |
| `und/pmi_frac`               | (`nll` only) `und/pmi` relative to `--und-pmi-ceiling`.                                                         |


On the default `nll` arm the headline number is `und/pmi`, in nats/token. On the `siglip` arm it is `und/i2t_r1`, which is the easier of the two to read: it is bounded, interpretable against `und/chance_r1` from the first steps, and needs no reference run to make sense of.

## Evaluation: the linear probe

```bash
TRAIN_DIR=/data/imagenet/c2i-256 \
LABEL_MAP=/data/imagenet/classnames.txt \
VAL_DIR=/data/imagenet/val \
VAE_CKPT=exps/<run>/checkpoints/0600000.pt \
EXP=probe-e2e-und \
bash scripts/probe_latent.sh
```

`MODE=full` additionally probes the trained projector and Qwen3-VL's own ViT features (the ceiling). It needs an `--und-type nll` checkpoint, since that is where a projector is trained, and it loads Qwen3-VL.

## Evaluating the released checkpoints

Two shared-latent checkpoints (and their matching extracted Qwen3-VL ViT VAE bundles) are published on Hugging Face:

- **E2E + SigLIP (`0.90` GenEval, `83.76` DPG-Bench, `79.25%` linear probe):**
  - Prior: `shared-latents/siglip/eit-qwenvit-siglip-t2i-512.pt`
  - VAE bundle: `vae/qwenvit-siglip-b256-s2/`
- **E2E + VLM-NLL (`0.89` GenEval, `83.49` DPG-Bench, `81.69%` linear probe):**
  - Prior: `shared-latents/vlm-nll/eit-qwenvit-vlm-nll-t2i-512.pt`
  - VAE bundle: `vae/qwenvit-nlltrunk-s2/`

To run text-to-image inference or GenEval / DPG-Bench evaluation with a released shared-latent checkpoint:

```bash
cd Prior_only
python infer_t2i.py \
    --ckpt /path/to/shared-latents/siglip/eit-qwenvit-siglip-t2i-512.pt \
    --vae-arch qwenvit \
    --vae-ckpt /path/to/vae/qwenvit-siglip-b256-s2/qwenvit-siglip-b256-s2.pt \
    --latents-stats /path/to/vae/qwenvit-siglip-b256-s2/qwenvit-siglip-b256-s2-latents-stats.pt \
    --vae-vit-path Qwen/Qwen3-VL-2B-Instruct \
    --text-encoder-name openai/clip-vit-large-patch14 \
    --out infer_out/qwenvit_siglip
```

```bash
cd eval_bench
export CKPT=/path/to/shared-latents/siglip/eit-qwenvit-siglip-t2i-512.pt
export VAE_ARCH=qwenvit
export VAE_CKPT=/path/to/vae/qwenvit-siglip-b256-s2/qwenvit-siglip-b256-s2.pt
export LATENTS_STATS=/path/to/vae/qwenvit-siglip-b256-s2/qwenvit-siglip-b256-s2-latents-stats.pt
export VAE_VIT_PATH=Qwen/Qwen3-VL-2B-Instruct
export TEXT_ENCODER_NAME=openai/clip-vit-large-patch14
bash scripts/eval_geneval.sh
RESOLUTION=512 bash scripts/eval_dpg.sh
```

## The encoder and its three phases


| Phase | Steps | Encoder              | Objective                                                    |
| ----- | ----- | -------------------- | ------------------------------------------------------------ |
| 0     | 30k   | **frozen**           | reconstruction only (`--no-gen-loss`), weak KL, GAN from 10k |
| 1     | 400k  | unfrozen at `ENC_LR` | joint reconstruction + denoising + REPA, `λ = 0.1`           |
| 2     | 200k  | unfrozen at `ENC_LR` | same, `λ = 0.01`                                             |


## The Qwen3-VL dependency

`--vae qwenvit` and `--und-type nll` need Qwen3-VL, which requires `transformers >= 4.57`. The pin in `requirements.txt` is 5.9.0, so nothing extra is needed — install the requirements and both paths work.

Only `environment.yml`, which is the archival record of the versions the published numbers were produced with, pins an older `transformers` without Qwen3-VL. If you install from it, the `nll` objective and `--vae qwenvit` are unavailable and `--und-type siglip` is the arm that runs. 