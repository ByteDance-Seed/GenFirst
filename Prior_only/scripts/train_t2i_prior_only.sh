#!/usr/bin/env bash
# Prior-only stage (T2I): frozen FLUX VAE + text-conditional MMDiT
# (Prior_only/train_t2i_frozen.py).
#
#   RUN=e2e  -> frozen end-to-end VAE extracted from the end-to-end stage
#   RUN=orig -> frozen original FLUX.1 VAE (baseline)
#   EXP_BASE    -> experiment-name prefix, to keep runs on different VAEs apart
#
#   PHASE=1 -> 256, 200K steps, global batch 1536
#   PHASE=2 -> 512,  80K steps, global batch 512, warm-start from phase 1
#   PHASE=3 -> 512 finetune, 40K steps, global batch 512, warm-start from phase 2
#              (optional high-quality captions)
#
# These are the settings the released T2I checkpoints were trained with. Phase 1's
# batch needs a lot of GPUs; set BATCH_SIZE to override every phase at once, or
# S1_BATCH / S2_BATCH / S3_BATCH to override one.
#
# Usage:
#   RUN=e2e E2E_VAE_CKPT=/path/to/e2e-flux-vae.pt PHASE=1 \
#     DATA=/path/to/parquet bash scripts/train_t2i_prior_only.sh

set -euo pipefail
cd "$(dirname "$0")/.."
source ../scripts/dist_launch.sh

RUN="${RUN:-e2e}"
PHASE="${PHASE:-1}"
NPROC="${NPROC:-8}"
# Unset by default so each phase can pick its own; set it to override all three.
BATCH_SIZE="${BATCH_SIZE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-exps}"
WANDB_PROJECT="${WANDB_PROJECT:-genfirst}"
REPORT_TO="${REPORT_TO:-wandb}"

VAE_ARCH="${VAE_ARCH:-flux}"               # flux (FLUX.1, 16ch) | flux2 (FLUX.2, 32ch) | qwenvit (Qwen3-VL ViT AE, 16ch)
FLUX_VAE="${FLUX_VAE:-../pretrained/flux-vae}"
FLUX2_VAE="${FLUX2_VAE:-../pretrained/flux2-vae}"
QWENVIT_PATH="${QWENVIT_PATH:-Qwen/Qwen3-VL-2B-Instruct}"
case "${VAE_ARCH}" in
  flux)    ORIG_VAE="${FLUX_VAE}" ;;
  flux2)   ORIG_VAE="${FLUX2_VAE}" ;;
  qwenvit) ORIG_VAE="" ;;
  *)       echo "VAE_ARCH must be flux, flux2, or qwenvit (got '${VAE_ARCH}')" >&2; exit 1;;
esac

EXTRACT_ROOT="${EXTRACT_ROOT:-../pretrained/extracted_vae}"
DATA="${DATA:?set DATA to comma-separated parquet roots}"
CAPTION_COLUMNS="${CAPTION_COLUMNS:-caption,prompt,text}"

if [ "${RUN}" = "e2e" ] || [ "${VAE_ARCH}" = "qwenvit" ]; then
    E2E_VAE_CKPT="${E2E_VAE_CKPT:-${EXTRACT_ROOT}/e2e-${VAE_ARCH}-vae/e2e-${VAE_ARCH}-vae.pt}"
    DEFAULT_STATS="${E2E_VAE_CKPT%.safetensors}"
    DEFAULT_STATS="${DEFAULT_STATS%.pt}-latents-stats.pt"
    STATS="${STATS:-${DEFAULT_STATS}}"
    VAE_ARGS=(--vae-arch "${VAE_ARCH}" --vae-ckpt "${E2E_VAE_CKPT}" --latents-stats "${STATS}")
    if [ "${VAE_ARCH}" = "qwenvit" ]; then
        VAE_ARGS+=(--vae-vit-path "${QWENVIT_PATH}")
    fi
    BASE="${EXP_BASE:-t2i-frozen-e2e-${VAE_ARCH}-vae-mmdit}"
else
    # No STATS: the trainer falls back to FLUX.1's constant shift/scale or FLUX.2's
    # embedded BN stats, which is the right default. Set STATS to use per-channel
    # stats from make_flux_latents_stats.py --from-data instead.
    VAE_ARGS=(--vae-arch "${VAE_ARCH}" --vae-ckpt "${ORIG_VAE}")
    if [ -n "${STATS:-}" ]; then
        VAE_ARGS+=(--latents-stats "${STATS}")
    fi
    BASE="${EXP_BASE:-t2i-frozen-orig-${VAE_ARCH}-vae-mmdit}"
fi

S1_STEPS="${S1_STEPS:-200000}"
S2_STEPS="${S2_STEPS:-80000}"
S3_STEPS="${S3_STEPS:-40000}"
S1_BATCH="${BATCH_SIZE:-${S1_BATCH:-1536}}"
S2_BATCH="${BATCH_SIZE:-${S2_BATCH:-512}}"
S3_BATCH="${BATCH_SIZE:-${S3_BATCH:-512}}"
EXP1="${BASE}-phase1-256"
EXP2="${BASE}-phase2-512"
EXP3="${BASE}-phase3-512-ft"
ckpt_of() { printf "%s/%s/checkpoints/%07d.pt" "${OUTPUT_DIR}" "$1" "$2"; }

case "${PHASE}" in
  1)
    EXP="${EXP1}"
    PHASE_ARGS=(--resolution=256 --max-train-steps="${S1_STEPS}"
                --batch-size="${S1_BATCH}" --learning-rate="${LR:-1e-4}"
                --data-dir="${DATA}" --caption-columns="${CAPTION_COLUMNS}")
    ;;
  2)
    EXP="${EXP2}"
    INIT="${INIT_CKPT:-$(ckpt_of "${EXP1}" "${S1_STEPS}")}"
    PHASE_ARGS=(--resolution=512 --max-train-steps="${S2_STEPS}"
                --batch-size="${S2_BATCH}" --learning-rate="${LR:-1e-4}"
                --data-dir="${DATA}" --caption-columns="${CAPTION_COLUMNS}"
                --init-ckpt="${INIT}")
    ;;
  3)
    EXP="${EXP3}"
    INIT="${INIT_CKPT:-$(ckpt_of "${EXP2}" "${S2_STEPS}")}"
    DATA_FT="${DATA_FT:-${DATA}}"
    CAPS_FT="${CAPS_FT:-${CAPTION_COLUMNS}}"
    PHASE_ARGS=(--resolution=512 --max-train-steps="${S3_STEPS}"
                --batch-size="${S3_BATCH}" --learning-rate="${LR:-5e-5}"
                --data-dir="${DATA_FT}" --caption-columns="${CAPS_FT}"
                --init-ckpt="${INIT}")
    ;;
  *)
    echo "PHASE must be 1, 2, or 3 (got '${PHASE}')" >&2; exit 1;;
esac

echo ">>> RUN=${RUN} PHASE=${PHASE} EXP=${EXP}"
[ "${PHASE}" != "1" ] && echo ">>> warm-start from: ${INIT}"

torchrun "${TORCHRUN_LAUNCH[@]}" --nproc_per_node="${NPROC}" train_t2i_frozen.py \
    --report-to="${REPORT_TO}" \
    --wandb-project="${WANDB_PROJECT}" \
    --allow-tf32 \
    --mixed-precision="fp16" \
    --seed=0 \
    --parquet-rows-per-file=128 \
    --output-dir="${OUTPUT_DIR}" \
    --num-vis-samples=8 \
    --path-type="linear" \
    --prediction="v" \
    --weighting="uniform" \
    --depth=28 \
    --hidden-size=1152 \
    --num-heads=16 \
    --text-encoder="clip" \
    --cfg-prob=0.1 \
    --checkpointing-steps="${CKPT_STEPS:-20000}" \
    --sampling-steps=5000 \
    --use-repa \
    --enc-type="dinov2-vit-b" \
    --proj-coeff=0.5 \
    --encoder-depth=8 \
    "${VAE_ARGS[@]}" \
    "${PHASE_ARGS[@]}" \
    --exp-name="${EXP}"
