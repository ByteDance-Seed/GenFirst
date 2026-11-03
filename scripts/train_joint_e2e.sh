#!/usr/bin/env bash
# End-to-end stage (joint image+text): frozen image VAE + end-to-end-trained Cola
# text VAE + block-causal joint MMDiT, mixing t2i / i2t per sample.
#
# This stage exists to *reshape the text VAE*; its DiT is thrown away. Extract the
# reshaped VAEs afterwards and train the model that actually gets evaluated with
# Prior_only/scripts/train_joint_prior_only.sh:
#
#   python save_joint_vae_weights.py --joint-ckpt exps/<EXP>/checkpoints/0300000.pt \
#       --name e2e-joint-l2 --save-dir extracted_vae
#
# Two phases, as in the image-only and T2I recipes: the prior weight starts high so
# the generative objective shapes the text latent space, then drops so that text
# reconstruction regains relative weight.
#
#   PHASE=1   0 -> 200k steps      lambda_prior = lambda_ent = 0.1
#   PHASE=2   200k -> 300k         lambda_prior = lambda_ent = 0.01
#
# Phase 2 is a true continuation: it resumes optimizer state from the same run, so
# EXP must not change between the two invocations.
#
# Prerequisites (see docs/JOINT.md):
#   * the Cola DLM release, for the pretrained text VAE and its tokenizer
#   * an image VAE: either a FLUX.1 VAE, or one reshaped by the T2I end-to-end
#     stage (scripts/train_t2i_e2e.sh) together with its -latents-stats.pt
#
# Usage:
#   PHASE=1 DATA=/path/to/parquet_roots bash scripts/train_joint_e2e.sh
#   PHASE=2 DATA=/path/to/parquet_roots bash scripts/train_joint_e2e.sh
#   IMAGE_VAE_MODE=e2e-recon PHASE=1 DATA=... bash scripts/train_joint_e2e.sh

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/dist_launch.sh

PHASE="${PHASE:-1}"
NPROC="${NPROC:-8}"
BATCH_SIZE="${BATCH_SIZE:-256}"
OUTPUT_DIR="${OUTPUT_DIR:-exps}"
MODEL="${MODEL:-JointMMDiT-L/2}"
RESOLUTION="${RESOLUTION:-256}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-10000}"
LOG_EVERY="${LOG_EVERY:-100}"
WANDB_PROJECT="${WANDB_PROJECT:-genfirst}"
REPORT_TO="${REPORT_TO:-wandb}"
LOSS_CFG="${LOSS_CFG:-configs/l1_lpips_entropy_gan.yaml}"

# ---------------------------------------------------------------- text VAE
# Pretrained Cola text VAE + its tokenizer. See docs/JOINT.md for the download.
COLA_ROOT="${COLA_ROOT:-${PRETRAINED_DIR:-pretrained}/cola-dlm}"
TEXT_VAE_PATH="${TEXT_VAE_PATH:-${COLA_ROOT}/cola_dlm/cola_vae}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${COLA_ROOT}/tokenizer.json}"

# ---------------------------------------------------------------- image VAE
# `frozen` keeps the image branch fixed and only runs its encoder, which is the
# cheap default. `e2e-recon` / `e2e-gan` also train the image VAE and roughly
# double step time and peak memory.
IMAGE_VAE_MODE="${IMAGE_VAE_MODE:-frozen}"
IMAGE_VAE_CKPT="${IMAGE_VAE_CKPT:-}"
IMAGE_LATENTS_STATS="${IMAGE_LATENTS_STATS:-}"
FLUX_VAE="${FLUX_VAE:-${PRETRAINED_DIR:-pretrained}/flux-vae}"

if [ -n "${IMAGE_VAE_CKPT}" ]; then
    # A VAE reshaped by the T2I end-to-end stage: its latent distribution is not
    # FLUX's any more, so the measured statistics have to travel with it.
    IMAGE_LATENTS_STATS="${IMAGE_LATENTS_STATS:-${IMAGE_VAE_CKPT%.pt}-latents-stats.pt}"
    IMAGE_VAE_ARGS=(--image-vae-ckpt="${IMAGE_VAE_CKPT}"
                    --image-latents-stats="${IMAGE_LATENTS_STATS}")
    [ -f "${IMAGE_VAE_CKPT}" ] || { echo "ERROR: image VAE not found: ${IMAGE_VAE_CKPT}" >&2; exit 1; }
    [ -f "${IMAGE_LATENTS_STATS}" ] || { echo "ERROR: latent stats not found: ${IMAGE_LATENTS_STATS}" >&2; exit 1; }
else
    # FLUX's own scalar shift/scale only seeds the statistics, so the DiT's BN has
    # to measure the real ones (implied anyway once the image VAE trains).
    IMAGE_VAE_ARGS=(--image-vae-pretrained="${FLUX_VAE}" --track-latent-stats)
    [ -d "${FLUX_VAE}" ] || { echo "ERROR: FLUX VAE dir not found: ${FLUX_VAE}" >&2; exit 1; }
fi

DATA="${DATA:?set DATA to comma-separated parquet roots (see docs/T2I.md)}"
CAPTION_COLUMNS="${CAPTION_COLUMNS:-caption,prompt,text}"

# max-train-steps is absolute, so phase 2's 300000 means "100k more", matching the
# image-only recipe's convention.
case "${PHASE}" in
    1)
        MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-200000}"
        PRIOR_WEIGHT="${PRIOR_WEIGHT:-0.1}"
        RESUME_STEP="${RESUME_STEP:-0}"
        ;;
    2)
        MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-300000}"
        PRIOR_WEIGHT="${PRIOR_WEIGHT:-0.01}"
        RESUME_STEP="${RESUME_STEP:-200000}"
        ;;
    *)
        echo "ERROR: unknown PHASE '${PHASE}'. Supported: 1, 2." >&2
        exit 1
        ;;
esac

# lambda_prior on each VAE is set equal to the entropy weight opposing it, the
# pairing the image-only recipe holds at every phase. All four therefore move
# together across phases.
ENTROPY_WEIGHT="${ENTROPY_WEIGHT:-${PRIOR_WEIGHT}}"
TEXT_VAE_COEFF="${TEXT_VAE_COEFF:-${ENTROPY_WEIGHT}}"
IMAGE_VAE_COEFF="${IMAGE_VAE_COEFF:-${ENTROPY_WEIGHT}}"

EXP="${EXP:-joint-e2e-${RESOLUTION}-$(echo "${MODEL}" | tr '/' '-')}"

if [ "${PHASE}" = "2" ] && [ "${RESUME_STEP}" != "0" ]; then
    PREV="${OUTPUT_DIR}/${EXP}/checkpoints/$(printf '%07d' "${RESUME_STEP}").pt"
    [ -f "${PREV}" ] || {
        echo "ERROR: PHASE=2 resumes ${PREV}, which does not exist." >&2
        echo "       Finish phase 1 first, or set RESUME_STEP=<a step you have>." >&2
        exit 1
    }
fi

[ -f "${TOKENIZER_PATH}" ] || { echo "ERROR: tokenizer not found: ${TOKENIZER_PATH} (see docs/JOINT.md)" >&2; exit 1; }
[ -d "${TEXT_VAE_PATH}" ] || { echo "ERROR: text VAE dir not found: ${TEXT_VAE_PATH} (see docs/JOINT.md)" >&2; exit 1; }
[ -f "${LOSS_CFG}" ] || { echo "ERROR: loss config not found: ${LOSS_CFG}" >&2; exit 1; }

cat <<EOF
--------------------------------------------------------------------
 Joint end-to-end stage, phase ${PHASE}
   exp-name        : ${EXP}
   model           : ${MODEL} @ ${RESOLUTION}
   image VAE       : ${IMAGE_VAE_MODE} (${IMAGE_VAE_CKPT:-${FLUX_VAE}})
   text VAE        : ${TEXT_VAE_PATH}
   lambda_prior    : ${PRIOR_WEIGHT}   lambda_ent: ${ENTROPY_WEIGHT}
   vae coeffs      : text ${TEXT_VAE_COEFF}   image ${IMAGE_VAE_COEFF}
   resume-step     : ${RESUME_STEP}
   max-train-steps : ${MAX_TRAIN_STEPS} (absolute)
   data            : ${DATA}
--------------------------------------------------------------------
EOF

torchrun "${TORCHRUN_LAUNCH[@]}" --nproc_per_node="${NPROC}" train_joint_e2e.py \
    --exp-name="${EXP}" \
    --output-dir="${OUTPUT_DIR}" \
    --report-to="${REPORT_TO}" \
    --wandb-project="${WANDB_PROJECT}" \
    --model="${MODEL}" \
    --resolution="${RESOLUTION}" \
    --data-dir="${DATA}" \
    --caption-columns="${CAPTION_COLUMNS}" \
    --parquet-rows-per-file=128 \
    --image-vae-mode="${IMAGE_VAE_MODE}" \
    "${IMAGE_VAE_ARGS[@]}" \
    --loss-cfg-path="${LOSS_CFG}" \
    --text-vae-path="${TEXT_VAE_PATH}" \
    --tokenizer-path="${TOKENIZER_PATH}" \
    --text-block-size="${TEXT_BLOCK_SIZE:-16}" \
    --max-text-tokens="${MAX_TEXT_TOKENS:-256}" \
    --t2i-prob="${T2I_PROB:-0.5}" \
    --cond-dropout="${COND_DROPOUT:-0.1}" \
    --recon-weight="${RECON_WEIGHT:-1.0}" \
    --kl-weight="${KL_WEIGHT:-1e-6}" \
    --entropy-weight="${ENTROPY_WEIGHT}" \
    --prior-weight="${PRIOR_WEIGHT}" \
    --latent-reg-weight="${ENTROPY_WEIGHT}" \
    --vae-diffusion-coeff="${TEXT_VAE_COEFF}" \
    --image-vae-diffusion-coeff="${IMAGE_VAE_COEFF}" \
    --use-repa \
    --enc-type="${ENC_TYPE:-dinov2-vit-b}" \
    --proj-coeff="${PROJ_COEFF:-0.5}" \
    --encoder-depth="${ENCODER_DEPTH:-8}" \
    --vae-align-proj-coeff="${VAE_ALIGN_PROJ_COEFF:-1.5}" \
    --batch-size="${BATCH_SIZE}" \
    --learning-rate="${LEARNING_RATE:-1e-4}" \
    --text-vae-lr="${TEXT_VAE_LR:-1e-5}" \
    --bn-momentum=0.1 \
    --mixed-precision="${MIXED_PRECISION:-bf16}" \
    --resume-step="${RESUME_STEP}" \
    --max-train-steps="${MAX_TRAIN_STEPS}" \
    --checkpointing-steps="${CHECKPOINTING_STEPS}" \
    --log-every="${LOG_EVERY}" \
    --num-workers=8 \
    --seed=0 \
    --allow-tf32
