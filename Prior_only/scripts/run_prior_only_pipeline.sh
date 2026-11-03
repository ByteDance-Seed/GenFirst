#!/usr/bin/env bash
# Prior-only stage: freeze the end-to-end VAE and train a fresh SiT
# prior on its latents.
#
# The end-to-end SiT was only a means of shaping the latent space; this is the model
# that gets evaluated. Training the prior on a frozen end-to-end VAE beats using
# the jointly trained prior directly, because a diffusion model gives no exact
# likelihood over the latents and so never fits them as a proper density.
#
# Steps, meant to be run one at a time and in this order:
#
#   extract   pull the VAE + latent statistics out of an end-to-end checkpoint,
#             or out of the decoder finetune that follows it (same call)
#   encode    cache VAE latents for the whole dataset (this is the slow one)
#   train     train SiT on the cached latents, 4M steps (800 epochs)
#   eval      50k-sample FID / IS (thin wrapper over scripts/eval_fid.sh, which
#             takes the guidance settings -- run it directly to reproduce the paper)
#
# VAE_ARCH and RESOLUTION must match across steps: the extraction directory, the
# latent cache and the experiment name are all derived from them, so a step that
# disagrees writes or reads the wrong path. Export them rather than repeating.
#
# Usage:
#   E2E_CKPT=../exps/e2e-f8d4-sitxl2-256-phase2/checkpoints/0600000.pt \
#     bash scripts/run_prior_only_pipeline.sh extract
#   bash scripts/run_prior_only_pipeline.sh encode
#   bash scripts/run_prior_only_pipeline.sh train
#
#   # continue an interrupted training run
#   RESUME_STEP=1500000 bash scripts/run_prior_only_pipeline.sh train
#
#   # start from a released VAE instead of your own end-to-end run: skip `extract`
#   export VAE_CKPT=/path/to/vae.pt LATENTS_STATS=/path/to/vae-latents-stats.pt
#   bash scripts/run_prior_only_pipeline.sh encode
#   bash scripts/run_prior_only_pipeline.sh train
#
# Several steps can still be named at once ("encode train"), and no argument at
# all runs everything including eval.

set -euo pipefail
cd "$(dirname "$0")/.."   # -> Prior_only/
source ../scripts/dist_launch.sh

STEPS="${*:-all}"

VAE_ARCH="${VAE_ARCH:-f8d4}"
RESOLUTION="${RESOLUTION:-256}"
NPROC="${NPROC:-8}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-4000000}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-50000}"
OUTPUT_DIR="${OUTPUT_DIR:-exps}"
WANDB_PROJECT="${WANDB_PROJECT:-genfirst}"
REPORT_TO="${REPORT_TO:-wandb}"

# Where raw images live and where cached latents go. IMAGES_DIR is produced by
# ../tools/build_image_dir.py; see ../docs/DATA.md.
DATA_ROOT="${DATA_ROOT:-../data/imagenet${RESOLUTION}}"
IMAGES_DIR="${IMAGES_DIR:-${DATA_ROOT}/images}"

case "${VAE_ARCH}" in
  f8d4)   MODEL="${MODEL:-SiT-XL/2}" ;;
  f16d32) MODEL="${MODEL:-SiT-XL/1}" ;;
  *) echo "ERROR: unknown VAE_ARCH '${VAE_ARCH}'. Supported: f8d4, f16d32." >&2; exit 1 ;;
esac

VAE_NAME="${VAE_NAME:-e2e-${VAE_ARCH}}"
EXTRACT_DIR="${EXTRACT_DIR:-extracted_vae/${VAE_NAME}}"
VAE_CKPT="${VAE_CKPT:-${EXTRACT_DIR}/${VAE_NAME}.pt}"
LATENTS_STATS="${LATENTS_STATS:-${EXTRACT_DIR}/${VAE_NAME}-latents-stats.pt}"

# The latent cache is specific to one VAE and one resolution. Keying the
# directory on both stops a stale cache from being silently reused, which
# produces a model that trains without complaint and generates noise.
FEATURE_SUBDIR="${FEATURE_SUBDIR:-vae-${VAE_NAME}-${RESOLUTION}}"

EXP_NAME="${EXP_NAME:-prior-only-${VAE_NAME}-$(echo "${MODEL}" | tr -d '/-' | tr '[:upper:]' '[:lower:]')-${RESOLUTION}}"

want() { [ "${STEPS}" = "all" ] || printf '%s\n' ${STEPS} | grep -qx "$1"; }

# ---------------------------------------------------------------- extract ----
if want extract; then
  : "${E2E_CKPT:?set E2E_CKPT to an end-to-end or decoder-finetune checkpoint, e.g. ../exps/<run>/checkpoints/0600000.pt}"
  [ -f "${E2E_CKPT}" ] || { echo "ERROR: E2E_CKPT not found: ${E2E_CKPT}" >&2; exit 1; }
  echo "==> extract: ${E2E_CKPT} -> ${EXTRACT_DIR}"
  mkdir -p "$(dirname "${EXTRACT_DIR}")"
  python ../save_vae_weights.py \
      --e2e-ckpt="${E2E_CKPT}" \
      --vae-name="${VAE_NAME}" \
      --save-dir="$(dirname "${EXTRACT_DIR}")"
  for f in "${VAE_CKPT}" "${LATENTS_STATS}"; do
    [ -f "$f" ] || { echo "ERROR: extract did not produce $f" >&2; exit 1; }
  done
fi

# ----------------------------------------------------------------- encode ----
if want encode; then
  [ -f "${VAE_CKPT}" ] || { echo "ERROR: VAE_CKPT not found: ${VAE_CKPT} (run 'extract' first)" >&2; exit 1; }
  [ -d "${IMAGES_DIR}" ] || { echo "ERROR: IMAGES_DIR not found: ${IMAGES_DIR} (see ../docs/DATA.md)" >&2; exit 1; }
  echo "==> encode: ${IMAGES_DIR} -> ${DATA_ROOT}/${FEATURE_SUBDIR}"
  # Always single-node: the cache lands on local disk, and every node needs all of it.
  torchrun --standalone --nproc_per_node="${NPROC}" ../tools/encode_latents.py \
      --vae-type=e2e \
      --vae-arch="${VAE_ARCH}" \
      --vae-ckpt="${VAE_CKPT}" \
      --source="${IMAGES_DIR}" \
      --dest="${DATA_ROOT}/${FEATURE_SUBDIR}"
fi

# ------------------------------------------------------------------ train ----
if want train; then
  for f in "${VAE_CKPT}" "${LATENTS_STATS}"; do
    [ -f "$f" ] || { echo "ERROR: not found: $f" >&2; exit 1; }
  done
  [ -d "${DATA_ROOT}/${FEATURE_SUBDIR}" ] || {
    echo "ERROR: latent cache not found: ${DATA_ROOT}/${FEATURE_SUBDIR} (run 'encode' first)" >&2; exit 1; }

  # Fetch DINOv2 once, before the distributed launch. Every rank calling
  # torch.hub.load() at the same time races on a single cache directory and can
  # leave it corrupted.
  echo "==> pre-fetching DINOv2 into the torch hub cache"
  python -c "from utils import load_dinov2_hub; load_dinov2_hub('dinov2_vitb14')" >/dev/null

  echo "==> train: ${EXP_NAME} (${MAX_TRAIN_STEPS} steps)"
  torchrun "${TORCHRUN_LAUNCH[@]}" --nproc_per_node="${NPROC}" train_prior_only.py \
      --report-to="${REPORT_TO}" \
      --wandb-project="${WANDB_PROJECT}" \
      --allow-tf32 \
      --mixed-precision="fp16" \
      --seed=0 \
      --resolution="${RESOLUTION}" \
      --data-dir="${DATA_ROOT}" \
      --feature-subdir="${FEATURE_SUBDIR}" \
      --output-dir="${OUTPUT_DIR}" \
      --exp-name="${EXP_NAME}" \
      --batch-size="${BATCH_SIZE}" \
      --max-train-steps="${MAX_TRAIN_STEPS}" \
      --checkpointing-steps="${CHECKPOINTING_STEPS}" \
      --model="${MODEL}" \
      --path-type="linear" \
      --prediction="v" \
      --weighting="uniform" \
      --enc-type="dinov2-vit-b" \
      --proj-coeff=0.5 \
      --encoder-depth=8 \
      --vae-type=e2e \
      --vae-arch="${VAE_ARCH}" \
      --vae-ckpt="${VAE_CKPT}" \
      --latents-stats="${LATENTS_STATS}" \
      --resume-step="${RESUME_STEP:-0}"
fi

# ------------------------------------------------------------------- eval ----
if want eval; then
  echo "==> eval: 50k-sample FID for ${EXP_NAME}"
  VAE_ARCH="${VAE_ARCH}" \
  RESOLUTION="${RESOLUTION}" \
  MODEL="${MODEL}" \
  VAE_CKPT="${VAE_CKPT}" \
  LATENTS_STATS="${LATENTS_STATS}" \
  EXP_NAME="${EXP_NAME}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  NPROC="${NPROC}" \
    bash scripts/eval_fid.sh
fi
