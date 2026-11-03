#!/usr/bin/env bash
# End-to-end stage: train the VAE together with a SiT prior.
#
# The SiT denoising loss back-propagates into the VAE encoder
# (--vae-diffusion-grad, scaled by --vae-diffusion-coeff = lambda_prior), while
# the KL term is replaced by posterior-entropy maximization
# (--latent-reg entropy, weight --latent-reg-weight = lambda_ent) to stop the
# latent from collapsing. A small 1e-6 KL is kept on top to bound the latent
# scale, which SiT needs because it perturbs latents with unit-variance noise.
#
# GenFirst is the schedule over those two weights, kept equal throughout:
#
#   PHASE=1   0 -> 400k steps    lambda_prior = lambda_ent = 0.1    (80 epochs)
#   PHASE=2   400k -> 600k       lambda_prior = lambda_ent = 0.01   (+40 epochs)
#
# Phase 1 forms a generation-friendly latent geometry under weak reconstruction
# pressure; phase 2 lowers the prior weight so reconstruction recovers visual
# detail without leaving that geometry.
#
# The end-to-end stage is 256 only. 512 reuses this VAE frozen and trains a new
# SiT on 512 latents, which is entirely the prior-only stage's job; see
# Prior_only/scripts/run_prior_only_pipeline.sh.
#
# Usage:
#   bash scripts/train_e2e.sh                      # f8d4  + SiT-XL/2 @256, phase 1
#   PHASE=2 bash scripts/train_e2e.sh              # continue to 600k
#   VAE=f16d32 bash scripts/train_e2e.sh           # f16d32 + SiT-XL/1
#
# Every setting below can be overridden from the environment.

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/dist_launch.sh

PHASE="${PHASE:-1}"
VAE="${VAE:-f8d4}"
RESOLUTION="${RESOLUTION:-256}"
NPROC="${NPROC:-8}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DATA_DIR="${DATA_DIR:-data}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-val_data}"
OUTPUT_DIR="${OUTPUT_DIR:-exps}"
PRETRAINED_DIR="${PRETRAINED_DIR:-pretrained}"
LOSS_CFG="${LOSS_CFG:-configs/l1_lpips_entropy_gan.yaml}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-50000}"
WANDB_PROJECT="${WANDB_PROJECT:-genfirst}"
REPORT_TO="${REPORT_TO:-wandb}"

# The patch size is paired with the VAE downsample factor so that both
# configurations produce a 16x16 token grid at 256px and stay comparable in
# compute. Mixing them trains fine but scores worse.
case "${VAE}" in
  f8d4)
    MODEL="${MODEL:-SiT-XL/2}"
    VAE_CKPT="${VAE_CKPT:-${PRETRAINED_DIR}/sdvae/sdvae-f8d4.pt}"
    DISC_CKPT="${DISC_CKPT:-${PRETRAINED_DIR}/sdvae/sdvae-f8d4-discriminator-ckpt.pt}"
    ;;
  f16d32)
    MODEL="${MODEL:-SiT-XL/1}"
    VAE_CKPT="${VAE_CKPT:-${PRETRAINED_DIR}/vavae/vavae-f16d32.pt}"
    DISC_CKPT="${DISC_CKPT:-${PRETRAINED_DIR}/vavae/vavae-f16d32-discriminator-ckpt.pt}"
    ;;
  *)
    echo "ERROR: unknown VAE '${VAE}'. Supported: f8d4, f16d32." >&2
    exit 1
    ;;
esac

TAG="${TAG:-e2e-${VAE}-$(echo "${MODEL}" | tr -d '/-' | tr '[:upper:]' '[:lower:]')-${RESOLUTION}}"

# --max-train-steps is an ABSOLUTE global-step budget, not a number of
# additional steps. Phase 2 resumes at 400k and must therefore ask for 600000;
# asking for 200000 makes the run exit immediately.
RESUME_ARGS=()
case "${PHASE}" in
  1)
    PRIOR_WEIGHT="${PRIOR_WEIGHT:-0.1}"
    MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-400000}"
    EXP_NAME="${EXP_NAME:-${TAG}-phase1}"
    ;;
  2)
    PRIOR_WEIGHT="${PRIOR_WEIGHT:-0.01}"
    MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-600000}"
    EXP_NAME="${EXP_NAME:-${TAG}-phase2}"
    RESUME_STEP="${RESUME_STEP:-400000}"
    PREV_EXP="${PREV_EXP:-${OUTPUT_DIR}/${TAG}-phase1}"
    # Phase 2 keeps optimizer, EMA and discriminator state, so the weight change
    # lands on a warm run rather than restarting the optimization.
    RESUME_ARGS=(--resume-step="${RESUME_STEP}" --continue-train-exp-dir="${PREV_EXP}")
    ;;
  *)
    echo "ERROR: unknown PHASE '${PHASE}'. Supported: 1, 2." >&2
    echo "       512 is not an end-to-end phase: the 256 VAE is reused frozen and" >&2
    echo "       only a new SiT is trained, in the prior-only stage." >&2
    exit 1
    ;;
esac

# lambda_ent = lambda_prior throughout, as in the paper.
ENTROPY_WEIGHT="${ENTROPY_WEIGHT:-${PRIOR_WEIGHT}}"

for f in "${VAE_CKPT}" "${DISC_CKPT}" "${LOSS_CFG}"; do
  [ -f "$f" ] || { echo "ERROR: missing required file: $f" >&2; exit 1; }
done
[ -d "${DATA_DIR}" ] || { echo "ERROR: missing data dir: ${DATA_DIR} (see docs/DATA.md)" >&2; exit 1; }

cat <<EOF
--------------------------------------------------------------------
 end-to-end stage, phase ${PHASE}
   vae / sit       : ${VAE} / ${MODEL}
   resolution      : ${RESOLUTION}
   lambda_prior    : ${PRIOR_WEIGHT}   (--vae-diffusion-coeff)
   lambda_ent      : ${ENTROPY_WEIGHT}   (--latent-reg-weight)
   max-train-steps : ${MAX_TRAIN_STEPS} (absolute)
   exp-name        : ${EXP_NAME}
--------------------------------------------------------------------
EOF

torchrun "${TORCHRUN_LAUNCH[@]}" --nproc_per_node="${NPROC}" train_e2e.py \
    --max-train-steps="${MAX_TRAIN_STEPS}" \
    --report-to="${REPORT_TO}" \
    --wandb-project="${WANDB_PROJECT}" \
    --allow-tf32 \
    --mixed-precision="fp16" \
    --seed=0 \
    --resolution="${RESOLUTION}" \
    --data-dir="${DATA_DIR}" \
    --eval-data-dir="${EVAL_DATA_DIR}" \
    --output-dir="${OUTPUT_DIR}" \
    --batch-size="${BATCH_SIZE}" \
    --eval-batch-size=128 \
    --num-vis-samples=8 \
    --path-type="linear" \
    --prediction="v" \
    --weighting="uniform" \
    --model="${MODEL}" \
    --checkpointing-steps="${CHECKPOINTING_STEPS}" \
    --loss-cfg-path="${LOSS_CFG}" \
    --vae="${VAE}" \
    --vae-ckpt="${VAE_CKPT}" \
    --disc-pretrained-ckpt="${DISC_CKPT}" \
    --enc-type="dinov2-vit-b" \
    --proj-coeff=0.5 \
    --encoder-depth=8 \
    --vae-align-proj-coeff=1.5 \
    --bn-momentum=0.1 \
    --vae-diffusion-grad \
    --vae-diffusion-coeff="${PRIOR_WEIGHT}" \
    --latent-reg=entropy \
    --latent-reg-weight="${ENTROPY_WEIGHT}" \
    --entropy-aux-kl-weight=1e-6 \
    --no-eval-at-start \
    --eval-smoke-samples=64 \
    --exp-name="${EXP_NAME}" \
    "${RESUME_ARGS[@]}"
