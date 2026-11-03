#!/usr/bin/env bash
# Decoder-only VAE finetune, run after the end-to-end stage and before the
# prior-only stage.
#
# The latent space moves quickly during end-to-end training, which can leave the
# decoder lagging behind the latents it has to decode. This adapts the decoder to
# them, which is worth about 0.2 unguided gFID on ImageNet-256 (3.00 -> 2.79 at 80
# epochs).
#
#   trainable : vae.decoder + vae.post_quant_conv, discriminator
#   frozen    : vae.encoder + vae.quant_conv
#   loss      : L1 + LPIPS + GAN, no SiT and no REPA alignment
#
# Because the encoder is frozen the latent distribution does not move, so the
# latent statistics carry over unchanged and the finetuned VAE is a drop-in
# replacement wherever the original was used.
#
# Usage:
#   INIT_CKPT=exps/e2e-f8d4-sitxl2-256-phase2/checkpoints/0600000.pt \
#     bash scripts/train_decoder_ft.sh
#   VAE_ARCH=f16d32 INIT_CKPT=... bash scripts/train_decoder_ft.sh
#
# Every setting below can be overridden from the environment.

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/dist_launch.sh

VAE_ARCH="${VAE_ARCH:-f8d4}"
NPROC="${NPROC:-8}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-10000}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-10000}"
DATA_DIR="${DATA_DIR:-data}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-val_data}"
OUTPUT_DIR="${OUTPUT_DIR:-exps}"
LOSS_CFG="${LOSS_CFG:-configs/l1_lpips_entropy_gan.yaml}"
EVAL_STEPS="${EVAL_STEPS:-10000}"
WANDB_PROJECT="${WANDB_PROJECT:-genfirst}"
REPORT_TO="${REPORT_TO:-wandb}"
RESUME_STEP="${RESUME_STEP:-0}"

case "${VAE_ARCH}" in
  f8d4|f16d32) ;;
  *) echo "ERROR: unknown VAE_ARCH '${VAE_ARCH}'. Supported: f8d4, f16d32." >&2; exit 1 ;;
esac

EXP_NAME="${EXP_NAME:-decoder-ft-${VAE_ARCH}-256}"

RESUME_ARGS=()
if [ "${RESUME_STEP}" != "0" ]; then
  # Resuming reads the finetune run's own checkpoint, so INIT_CKPT is not needed.
  RESUME_ARGS=(--resume-step="${RESUME_STEP}")
else
  : "${INIT_CKPT:?set INIT_CKPT to the end-to-end checkpoint to finetune from}"
  [ -f "${INIT_CKPT}" ] || { echo "ERROR: INIT_CKPT not found: ${INIT_CKPT}" >&2; exit 1; }
  RESUME_ARGS=(--init-ckpt="${INIT_CKPT}")
fi

[ -f "${LOSS_CFG}" ] || { echo "ERROR: missing loss config: ${LOSS_CFG}" >&2; exit 1; }
[ -d "${DATA_DIR}" ] || { echo "ERROR: missing data dir: ${DATA_DIR} (see docs/DATA.md)" >&2; exit 1; }

cat <<EOF
--------------------------------------------------------------------
 decoder-only finetune
   vae             : ${VAE_ARCH}  (encoder frozen)
   max-train-steps : ${MAX_TRAIN_STEPS}
   exp-name        : ${EXP_NAME}
--------------------------------------------------------------------
EOF

torchrun "${TORCHRUN_LAUNCH[@]}" --nproc_per_node="${NPROC}" train_decoder_ft.py \
    --vae="${VAE_ARCH}" \
    --data-dir="${DATA_DIR}" \
    --eval-data-dir="${EVAL_DATA_DIR}" \
    --output-dir="${OUTPUT_DIR}" \
    --batch-size="${BATCH_SIZE}" \
    --resolution=256 \
    --max-train-steps="${MAX_TRAIN_STEPS}" \
    --checkpointing-steps="${CHECKPOINTING_STEPS}" \
    --report-to="${REPORT_TO}" \
    --wandb-project="${WANDB_PROJECT}" \
    --allow-tf32 \
    --mixed-precision="fp16" \
    --seed=0 \
    --loss-cfg-path="${LOSS_CFG}" \
    --eval-steps="${EVAL_STEPS}" \
    --exp-name="${EXP_NAME}" \
    "${RESUME_ARGS[@]}"
