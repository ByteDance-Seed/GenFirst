#!/usr/bin/env bash
# End-to-end stage (T2I): FLUX.1 VAE + text-conditional MMDiT on parquet data.
#
# GenFirst schedule (weights-only warm start between phases via --init-ckpt):
#   PHASE=1 -> 256, 400K, lambda_prior = lambda_ent = 0.1
#   PHASE=2 -> 256, 200K, lambda_prior = lambda_ent = 0.01  (warm-start phase 1)
#
# The T2I end-to-end stage stays at 256; 512 is handled by the prior-only stage
# (Prior_only/scripts/train_t2i_prior_only.sh PHASE=2). Extract the VAE from the
# phase-2 checkpoint.
#
# The prior here is MMDiT-L/2, smaller than the MMDiT-XL/2 the prior-only stage
# trains. This prior only exists to shape the latent space and is then discarded,
# so paying for XL capacity twice buys nothing.
#
# Usage:
#   PHASE=1 DATA=/path/to/parquet_roots bash scripts/train_t2i_e2e.sh
#   PHASE=2 bash scripts/train_t2i_e2e.sh

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/dist_launch.sh

PHASE="${PHASE:-1}"
NPROC="${NPROC:-8}"
BATCH_SIZE="${BATCH_SIZE:-256}"
OUTPUT_DIR="${OUTPUT_DIR:-exps}"
LOSS_CFG="${LOSS_CFG:-configs/l1_lpips_entropy_gan.yaml}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-50000}"
WANDB_PROJECT="${WANDB_PROJECT:-genfirst}"
REPORT_TO="${REPORT_TO:-wandb}"

VAE_ARCH="${VAE_ARCH:-flux}"               # flux (FLUX.1, 16ch) | flux2 (FLUX.2, 32ch)
FLUX_VAE="${FLUX_VAE:-${PRETRAINED_DIR:-pretrained}/flux-vae}"
FLUX2_VAE="${FLUX2_VAE:-${PRETRAINED_DIR:-pretrained}/flux2-vae}"
case "${VAE_ARCH}" in
  flux)  VAE_PRETRAINED="${FLUX_VAE}" ;;
  flux2) VAE_PRETRAINED="${FLUX2_VAE}" ;;
  *)     echo "VAE_ARCH must be flux or flux2 (got '${VAE_ARCH}')" >&2; exit 1;;
esac

DATA="${DATA:?set DATA to comma-separated parquet roots (see docs/T2I.md)}"
CAPTION_COLUMNS="${CAPTION_COLUMNS:-caption,prompt,text}"

BASE="t2i-e2e-${VAE_ARCH}-vae-mmdit-l2"
S1_STEPS="${S1_STEPS:-400000}"
S2_STEPS="${S2_STEPS:-200000}"
EXP1="${BASE}-phase1-256-c0.1"
EXP2="${BASE}-phase2-256-c0.01"
ckpt_of() { printf "%s/%s/checkpoints/%07d.pt" "${OUTPUT_DIR}" "$1" "$2"; }

case "${PHASE}" in
  1)
    EXP="${EXP1}"
    PRIOR_WEIGHT="${PRIOR_WEIGHT:-0.1}"
    PHASE_ARGS=(--resolution=256 --max-train-steps="${S1_STEPS}"
                --batch-size="${BATCH_SIZE}"
                --vae-diffusion-coeff="${PRIOR_WEIGHT}" --latent-reg-weight="${PRIOR_WEIGHT}")
    ;;
  2)
    EXP="${EXP2}"
    PRIOR_WEIGHT="${PRIOR_WEIGHT:-0.01}"
    INIT="${INIT_CKPT:-$(ckpt_of "${EXP1}" "${S1_STEPS}")}"
    PHASE_ARGS=(--resolution=256 --max-train-steps="${S2_STEPS}"
                --batch-size="${BATCH_SIZE}"
                --vae-diffusion-coeff="${PRIOR_WEIGHT}" --latent-reg-weight="${PRIOR_WEIGHT}"
                --init-ckpt="${INIT}")
    ;;
  *)
    echo "PHASE must be 1 or 2 (got '${PHASE}'); 512 is handled by the prior-only stage" >&2
    exit 1;;
esac

ENTROPY_WEIGHT="${ENTROPY_WEIGHT:-${PRIOR_WEIGHT}}"

[ -e "${VAE_PRETRAINED}" ] || { echo "ERROR: VAE pretrained path not found: ${VAE_PRETRAINED}" >&2; exit 1; }
[ -f "${LOSS_CFG}" ] || { echo "ERROR: loss config not found: ${LOSS_CFG}" >&2; exit 1; }

cat <<EOF
--------------------------------------------------------------------
 T2I end-to-end stage, phase ${PHASE}
   exp-name        : ${EXP}
   vae_arch        : ${VAE_ARCH}
   lambda_prior    : ${PRIOR_WEIGHT}
   lambda_ent      : ${ENTROPY_WEIGHT}
   data            : ${DATA}
--------------------------------------------------------------------
EOF
[ "${PHASE}" != "1" ] && echo ">>> warm-start from: ${INIT}"

torchrun "${TORCHRUN_LAUNCH[@]}" --nproc_per_node="${NPROC}" train_e2e.py \
    --report-to="${REPORT_TO}" \
    --wandb-project="${WANDB_PROJECT}" \
    --allow-tf32 \
    --mixed-precision="fp16" \
    --seed=0 \
    --conditioning="text" \
    --data-dir="${DATA}" \
    --caption-columns="${CAPTION_COLUMNS}" \
    --parquet-rows-per-file=128 \
    --output-dir="${OUTPUT_DIR}" \
    --num-vis-samples=8 \
    --path-type="linear" \
    --prediction="v" \
    --weighting="uniform" \
    --model="MMDiT-L/2" \
    --text-encoder="clip" \
    --cfg-prob=0.1 \
    --checkpointing-steps="${CHECKPOINTING_STEPS}" \
    --sampling-steps=5000 \
    --no-compile \
    --loss-cfg-path="${LOSS_CFG}" \
    --vae="${VAE_ARCH}" \
    --vae-pretrained-path="${VAE_PRETRAINED}" \
    --vae-ckpt="" \
    --enc-type="dinov2-vit-b" \
    --proj-coeff=0.5 \
    --encoder-depth=8 \
    --vae-align-proj-coeff=1.5 \
    --bn-momentum=0.1 \
    --vae-diffusion-grad \
    --latent-reg=entropy \
    --entropy-aux-kl-weight=1e-6 \
    --eval-steps=0 \
    --no-eval-at-start \
    --exp-name="${EXP}" \
    "${PHASE_ARGS[@]}"
