#!/usr/bin/env bash
# Prior-only stage (joint image+text): both VAEs frozen, a fresh block-causal
# joint MMDiT trained on them. This is the stage whose checkpoints get evaluated;
# the end-to-end stage (scripts/train_joint_e2e.sh) only exists to reshape the
# text VAE.
#
# Three phases, warm-started by weights only (--init-ckpt), the same schedule the
# T2I prior-only stage uses:
#   PHASE=1 -> 256, 200K steps, lr 1e-4
#   PHASE=2 -> 512,  80K steps, lr 1e-4, from phase 1
#   PHASE=3 -> 512,  40K steps, lr 5e-5, SFT captions, from phase 2
#
# RUN selects where the frozen text VAE comes from, which is the experiment:
#   RUN=cola  pretrained Cola text VAE, untouched          (baseline)
#   RUN=e2e   the text VAE the joint end-to-end stage reshaped
# IMAGE_VAE (flux|e2e) is independent, so the text VAE can be isolated on its own
# by comparing RUN=e2e IMAGE_VAE=flux against RUN=cola IMAGE_VAE=flux.
#
# Usage:
#   RUN=cola PHASE=1 DATA=/path/to/parquet_roots bash Prior_only/scripts/train_joint_prior_only.sh
#   RUN=e2e  TEXT_VAE_CKPT=extracted_vae/<name>/<name>-text-vae.pt PHASE=1 DATA=... \
#       bash Prior_only/scripts/train_joint_prior_only.sh

set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/dist_launch.sh

RUN="${RUN:-cola}"
PHASE="${PHASE:-1}"
NPROC="${NPROC:-8}"
BATCH_SIZE="${BATCH_SIZE:-256}"
OUTPUT_DIR="${OUTPUT_DIR:-Prior_only/exps}"
MODEL="${MODEL:-JointMMDiT-XL/2}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-20000}"
SAMPLING_STEPS="${SAMPLING_STEPS:-5000}"
LOG_EVERY="${LOG_EVERY:-100}"
WANDB_PROJECT="${WANDB_PROJECT:-genfirst}"
REPORT_TO="${REPORT_TO:-wandb}"

COLA_ROOT="${COLA_ROOT:-${PRETRAINED_DIR:-pretrained}/cola-dlm}"
TEXT_VAE_PATH="${TEXT_VAE_PATH:-${COLA_ROOT}/cola_dlm/cola_vae}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${COLA_ROOT}/tokenizer.json}"
FLUX_VAE="${FLUX_VAE:-${PRETRAINED_DIR:-pretrained}/flux-vae}"

# ---------------------------------------------------------------- text VAE arm
case "${RUN}" in
  cola)
    IMAGE_VAE="${IMAGE_VAE:-flux}"
    TEXT_VAE_ARGS=()
    BASE="joint-prior-only-cola-text"
    ;;
  e2e)
    IMAGE_VAE="${IMAGE_VAE:-e2e}"
    TEXT_VAE_CKPT="${TEXT_VAE_CKPT:?RUN=e2e needs TEXT_VAE_CKPT (see save_joint_vae_weights.py)}"
    [ -f "${TEXT_VAE_CKPT}" ] || { echo "ERROR: text VAE ckpt not found: ${TEXT_VAE_CKPT}" >&2; exit 1; }
    TEXT_VAE_ARGS=(--text-vae-ckpt="${TEXT_VAE_CKPT}")
    BASE="joint-prior-only-e2e-text"
    ;;
  *) echo "RUN must be cola or e2e (got '${RUN}')" >&2; exit 1;;
esac

# ---------------------------------------------------------------- image VAE arm
case "${IMAGE_VAE}" in
  e2e)
    IMAGE_VAE_CKPT="${IMAGE_VAE_CKPT:?IMAGE_VAE=e2e needs IMAGE_VAE_CKPT}"
    IMAGE_LATENTS_STATS="${IMAGE_LATENTS_STATS:-${IMAGE_VAE_CKPT%.pt}-latents-stats.pt}"
    [ -f "${IMAGE_VAE_CKPT}" ] || { echo "ERROR: image VAE not found: ${IMAGE_VAE_CKPT}" >&2; exit 1; }
    [ -f "${IMAGE_LATENTS_STATS}" ] || { echo "ERROR: latent stats not found: ${IMAGE_LATENTS_STATS}" >&2; exit 1; }
    IMAGE_VAE_ARGS=(--image-vae-ckpt="${IMAGE_VAE_CKPT}"
                    --image-latents-stats="${IMAGE_LATENTS_STATS}")
    BASE="${BASE}-e2e-img"
    ;;
  flux)
    # No measured statistics: FLUX's own scalar shift/scale, as the image-only
    # baseline uses.
    [ -d "${FLUX_VAE}" ] || { echo "ERROR: FLUX VAE dir not found: ${FLUX_VAE}" >&2; exit 1; }
    IMAGE_VAE_ARGS=(--image-vae-pretrained="${FLUX_VAE}")
    BASE="${BASE}-flux-img"
    ;;
  *) echo "IMAGE_VAE must be e2e or flux (got '${IMAGE_VAE}')" >&2; exit 1;;
esac

# ---------------------------------------------------------------- data
DATA="${DATA:?set DATA to comma-separated parquet roots (see docs/T2I.md)}"
CAPTION_COLUMNS="${CAPTION_COLUMNS:-caption,prompt,text}"
# Phase 3 finetunes on a higher-quality caption set; defaults to the same data
# so the schedule runs end to end without extra downloads.
DATA_SFT="${DATA_SFT:-${DATA}}"
CAPTION_COLUMNS_SFT="${CAPTION_COLUMNS_SFT:-${CAPTION_COLUMNS}}"

P1_STEPS="${P1_STEPS:-200000}"
P2_STEPS="${P2_STEPS:-80000}"
P3_STEPS="${P3_STEPS:-40000}"
EXP1="${BASE}-phase1-256"
EXP2="${BASE}-phase2-512"
EXP3="${BASE}-phase3-512-sft"
ckpt_of() { printf "%s/%s/checkpoints/%07d.pt" "${OUTPUT_DIR}" "$1" "$2"; }

case "${PHASE}" in
  1)
    EXP="${EXP1}"
    PHASE_ARGS=(--resolution=256 --max-train-steps="${P1_STEPS}"
                --learning-rate="${LEARNING_RATE:-1e-4}"
                --data-dir="${DATA}" --caption-columns="${CAPTION_COLUMNS}")
    ;;
  2)
    EXP="${EXP2}"
    INIT="${INIT_CKPT:-$(ckpt_of "${EXP1}" "${P1_STEPS}")}"
    PHASE_ARGS=(--resolution=512 --max-train-steps="${P2_STEPS}"
                --learning-rate="${LEARNING_RATE:-1e-4}"
                --data-dir="${DATA}" --caption-columns="${CAPTION_COLUMNS}"
                --init-ckpt="${INIT}")
    ;;
  3)
    EXP="${EXP3}"
    INIT="${INIT_CKPT:-$(ckpt_of "${EXP2}" "${P2_STEPS}")}"
    PHASE_ARGS=(--resolution=512 --max-train-steps="${P3_STEPS}"
                --learning-rate="${LEARNING_RATE:-5e-5}"
                --data-dir="${DATA_SFT}" --caption-columns="${CAPTION_COLUMNS_SFT}"
                --init-ckpt="${INIT}")
    ;;
  *) echo "PHASE must be 1, 2, or 3 (got '${PHASE}')" >&2; exit 1;;
esac

if [ "${PHASE}" != "1" ] && [ ! -f "${INIT}" ]; then
    echo "PHASE=${PHASE} warm-starts from ${INIT}, which does not exist." >&2
    echo "Finish the previous phase first, or pass INIT_CKPT=<path>." >&2
    exit 1
fi

[ -f "${TOKENIZER_PATH}" ] || { echo "ERROR: tokenizer not found: ${TOKENIZER_PATH} (see docs/JOINT.md)" >&2; exit 1; }
[ -d "${TEXT_VAE_PATH}" ] || { echo "ERROR: text VAE dir not found: ${TEXT_VAE_PATH} (see docs/JOINT.md)" >&2; exit 1; }

cat <<EOF
--------------------------------------------------------------------
 Joint prior-only stage, phase ${PHASE}
   exp-name        : ${EXP}
   model           : ${MODEL}
   text VAE        : ${RUN} (${TEXT_VAE_CKPT:-${TEXT_VAE_PATH}})
   image VAE       : ${IMAGE_VAE} (${IMAGE_VAE_CKPT:-${FLUX_VAE}})
--------------------------------------------------------------------
EOF
[ "${PHASE}" != "1" ] && echo ">>> warm-start from: ${INIT}"

torchrun "${TORCHRUN_LAUNCH[@]}" --nproc_per_node="${NPROC}" Prior_only/train_joint_frozen.py \
    --exp-name="${EXP}" \
    --output-dir="${OUTPUT_DIR}" \
    --report-to="${REPORT_TO}" \
    --wandb-project="${WANDB_PROJECT}" \
    --model="${MODEL}" \
    "${IMAGE_VAE_ARGS[@]}" \
    --text-vae-path="${TEXT_VAE_PATH}" \
    "${TEXT_VAE_ARGS[@]}" \
    --tokenizer-path="${TOKENIZER_PATH}" \
    --text-block-size="${TEXT_BLOCK_SIZE:-16}" \
    --max-text-tokens="${MAX_TEXT_TOKENS:-256}" \
    --t2i-prob="${T2I_PROB:-0.5}" \
    --cond-dropout="${COND_DROPOUT:-0.1}" \
    --use-repa \
    --enc-type="${ENC_TYPE:-dinov2-vit-b}" \
    --proj-coeff="${PROJ_COEFF:-0.5}" \
    --encoder-depth="${ENCODER_DEPTH:-8}" \
    "${PHASE_ARGS[@]}" \
    --batch-size="${BATCH_SIZE}" \
    --parquet-rows-per-file=128 \
    --resume-step="${RESUME_STEP:-0}" \
    --checkpointing-steps="${CHECKPOINTING_STEPS}" \
    --sampling-steps="${SAMPLING_STEPS}" \
    --log-every="${LOG_EVERY}" \
    --num-workers=8 \
    --seed=0 \
    --allow-tf32
