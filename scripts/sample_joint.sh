#!/usr/bin/env bash
# Sample both directions from a joint MMDiT checkpoint, as a sanity check.
#
# Runs a text VAE round-trip (which fails loudly if the checkpoint's text VAE
# weights did not load), then t2i (captions in, image grid out) and i2t (images
# in, captions printed against the ground truth). Model, VAEs, resolution and
# data are all restored from ckpt["args"], so the paths recorded at training time
# have to still resolve.
#
# Usage:
#   CKPT=exps/<run>/checkpoints/0400000.pt bash scripts/sample_joint.sh
#   CKPT=... USE_EMA=1 CFG_SCALE=4.0 bash scripts/sample_joint.sh

set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="${CKPT:?set CKPT to a joint checkpoint (.pt)}"
[ -f "${CKPT}" ] || { echo "ERROR: checkpoint not found: ${CKPT}" >&2; exit 1; }

OUT_DIR="${OUT_DIR:-samples}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"
EMA_ARGS=()
case "${USE_EMA:-0}" in 1|on|true|yes) EMA_ARGS=(--use-ema);; esac
HEUN_ARGS=()
case "${HEUN:-0}" in 1|on|true|yes) HEUN_ARGS=(--heun);; esac
OVERRIDE_ARGS=()
[ -n "${IMAGE_VAE_CKPT:-}" ] && OVERRIDE_ARGS+=(--image-vae-ckpt="${IMAGE_VAE_CKPT}")
[ -n "${IMAGE_VAE_PRETRAINED:-}" ] && OVERRIDE_ARGS+=(--image-vae-pretrained="${IMAGE_VAE_PRETRAINED}")
[ -n "${TEXT_VAE_PATH:-}" ] && OVERRIDE_ARGS+=(--text-vae-path="${TEXT_VAE_PATH}")
[ -n "${TOKENIZER_PATH:-}" ] && OVERRIDE_ARGS+=(--tokenizer-path="${TOKENIZER_PATH}")
[ -n "${DATA_DIR:-}" ] && OVERRIDE_ARGS+=(--data-dir="${DATA_DIR}")

python sample_joint.py \
    --ckpt="${CKPT}" \
    --out-dir="${OUT_DIR}" \
    "${OVERRIDE_ARGS[@]}" \
    --num-samples="${NUM_SAMPLES}" \
    --t2i-steps="${T2I_STEPS:-50}" \
    --i2t-steps="${I2T_STEPS:-20}" \
    --cfg-scale="${CFG_SCALE:-1.0}" \
    --seed="${SEED:-0}" \
    "${EMA_ARGS[@]}" \
    "${HEUN_ARGS[@]}"
