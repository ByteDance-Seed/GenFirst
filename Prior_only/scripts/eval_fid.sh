#!/usr/bin/env bash
# 50k-sample FID / IS for a prior-only SiT.
#
# Two steps: sample 50k PNGs, then score them with torch-fidelity against the
# reference statistics in fid_stats/ (the MAR protocol -- see docs/EVAL.md).
#
# The paper's inference settings, which are the defaults below per RESOLUTION:
#
#   256, unguided     CFG_SCALE=1.0
#   256, guided       CFG_SCALE=1.5 on [0.2, 0.7], autoguidance 1.3 off a 20-epoch
#                     checkpoint on the full [0, 1]
#   512, guided       CFG_SCALE=2.5, no autoguidance, interval [0.2, 0.8]
#
# The two intervals are separate on purpose: CFG is restricted to the middle of
# the trajectory, while autoguidance stays on throughout, which is where the weak
# model differs from the strong one the most.
#
# Sampler is SDE with 250 steps throughout, and class labels are exactly balanced
# (50 per class at 50k, shuffled) -- generate.py's default.
#
# Usage:
#   # unguided
#   EXP_NAME=prior-only-e2e-f8d4-sitxl2-256 CKPT_STEP=400000 \
#   VAE_CKPT=extracted_vae/e2e-f8d4/e2e-f8d4.pt \
#   LATENTS_STATS=extracted_vae/e2e-f8d4/e2e-f8d4-latents-stats.pt \
#     CFG_SCALE=1.0 bash scripts/eval_fid.sh
#
#   # guided, with autoguidance off the 20-epoch (100k-step) checkpoint
#   ... WEAK_CKPT=exps/<run>/checkpoints/0100000.pt bash scripts/eval_fid.sh

set -euo pipefail
cd "$(dirname "$0")/.."   # -> Prior_only/

: "${EXP_NAME:?set EXP_NAME to the prior-only experiment name}"
: "${VAE_CKPT:?set VAE_CKPT to the frozen VAE weights}"
: "${LATENTS_STATS:?set LATENTS_STATS to the matching latent statistics}"

VAE_ARCH="${VAE_ARCH:-f8d4}"
RESOLUTION="${RESOLUTION:-256}"
NPROC="${NPROC:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-exps}"
CKPT_STEP="${CKPT_STEP:-4000000}"
NUM_FID_SAMPLES="${NUM_FID_SAMPLES:-50000}"
PER_PROC_BATCH_SIZE="${PER_PROC_BATCH_SIZE:-32}"
MODE="${MODE:-sde}"
NUM_STEPS="${NUM_STEPS:-250}"
SAMPLE_DIR="${SAMPLE_DIR:-samples}"
SEED="${SEED:-0}"

# Guidance defaults follow the paper, and differ between resolutions.
if [ "${RESOLUTION}" = "512" ]; then
  CFG_SCALE="${CFG_SCALE:-2.5}"
  GUIDANCE_LOW="${GUIDANCE_LOW:-0.2}"
  GUIDANCE_HIGH="${GUIDANCE_HIGH:-0.8}"
else
  CFG_SCALE="${CFG_SCALE:-1.5}"
  GUIDANCE_LOW="${GUIDANCE_LOW:-0.2}"
  GUIDANCE_HIGH="${GUIDANCE_HIGH:-0.7}"
fi
AUTOGUIDANCE_SCALE="${AUTOGUIDANCE_SCALE:-1.3}"
AG_LOW="${AG_LOW:-0.0}"
AG_HIGH="${AG_HIGH:-1.0}"

# One projector per REPA encoder, so this has to match the checkpoint: a run with
# the paper's single DINOv2-B has one 768-d projector, and a mismatch here fails
# the state_dict load rather than degrading quietly. Override for multi-encoder runs.
PROJECTOR_EMBED_DIMS="${PROJECTOR_EMBED_DIMS:-768}"
# The paper's runs train with fused attention; it carries no parameters, so this is
# an implementation switch only.
FUSED_ATTN="${FUSED_ATTN:-1}"
[ "${FUSED_ATTN}" = "1" ] && ATTN_ARGS=(--fused-attn) || ATTN_ARGS=(--no-fused-attn)

case "${VAE_ARCH}" in
  f8d4)   MODEL="${MODEL:-SiT-XL/2}" ;;
  f16d32) MODEL="${MODEL:-SiT-XL/1}" ;;
  *) echo "ERROR: unknown VAE_ARCH '${VAE_ARCH}'. Supported: f8d4, f16d32." >&2; exit 1 ;;
esac

CKPT="${CKPT:-${OUTPUT_DIR}/${EXP_NAME}/checkpoints/$(printf '%07d' "${CKPT_STEP}").pt}"
[ -f "${CKPT}" ] || { echo "ERROR: checkpoint not found: ${CKPT}" >&2; exit 1; }

# Autoguidance is opt-in: without a weak checkpoint this is plain CFG.
AG_ARGS=()
if [ -n "${WEAK_CKPT:-}" ]; then
  [ -f "${WEAK_CKPT}" ] || { echo "ERROR: weak checkpoint not found: ${WEAK_CKPT}" >&2; exit 1; }
  AG_ARGS=(--weak-ckpt="${WEAK_CKPT}" --autoguidance-scale="${AUTOGUIDANCE_SCALE}"
           --ag-low="${AG_LOW}" --ag-high="${AG_HIGH}")
fi

# ------------------------------------------------------------- generate ------
echo "==> sampling ${NUM_FID_SAMPLES} images (cfg=${CFG_SCALE}, ${MODE}, ${NUM_STEPS} steps)"
torchrun --standalone --nproc_per_node="${NPROC}" generate.py \
    --ckpt="${CKPT}" \
    --model="${MODEL}" \
    --resolution="${RESOLUTION}" \
    --vae-type=e2e \
    --vae-arch="${VAE_ARCH}" \
    --vae-ckpt="${VAE_CKPT}" \
    --latents-stats="${LATENTS_STATS}" \
    --sample-dir="${SAMPLE_DIR}" \
    --num-fid-samples="${NUM_FID_SAMPLES}" \
    --per-proc-batch-size="${PER_PROC_BATCH_SIZE}" \
    --mode="${MODE}" \
    --num-steps="${NUM_STEPS}" \
    --path-type=linear \
    --cfg-scale="${CFG_SCALE}" \
    --guidance-low="${GUIDANCE_LOW}" \
    --guidance-high="${GUIDANCE_HIGH}" \
    --global-seed="${SEED}" \
    --encoder-depth=8 \
    --projector-embed-dims="${PROJECTOR_EMBED_DIMS}" \
    "${ATTN_ARGS[@]}" \
    "${AG_ARGS[@]}"

# generate.py encodes the sampling settings into the folder name, and float
# formatting there has changed between versions, so pick the newest folder rather
# than reconstructing the name.
FOLDER="$(find "${SAMPLE_DIR}" -mindepth 1 -maxdepth 1 -type d -newer "${CKPT}" -print \
          | sort | tail -1)"
[ -n "${FOLDER}" ] || FOLDER="$(find "${SAMPLE_DIR}" -mindepth 1 -maxdepth 1 -type d -print \
                                | sort | tail -1)"
[ -n "${FOLDER}" ] || { echo "ERROR: no sample folder under ${SAMPLE_DIR}" >&2; exit 1; }

# ----------------------------------------------------------------- score -----
python eval_fid.py --sample-dir="${FOLDER}" --resolution="${RESOLUTION}"
