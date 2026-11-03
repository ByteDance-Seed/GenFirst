#!/usr/bin/env bash
# Three-stage end-to-end training of the FLUX.2 VAE (32ch, f8) with an
# unconditional SiT-XL/2 prior on aspect-ratio-bucketed parquet data:
#
#   STAGE=1  low-res e2e   : side 256, general pretraining mix, L1+LPIPS (no GAN),
#                            diffusion->VAE 0.1, entropy 0.1, aux_kl 1e-6, from FLUX.2
#   STAGE=2  high-res e2e  : side 512, high-res/OCR mix, GAN from 40k,
#                            diffusion->VAE 0.01, entropy 0.01, aux_kl 2.5e-8, from stage 1
#   STAGE=3  decoder-GAN   : side 512, high-res/OCR mix, encoder frozen,
#                            L1+LPIPS+GAN (from 1k), from stage 2
# "side" = bucket area side: images keep their aspect ratio (1:2 .. 2:1),
# H*W ~= side^2.
#
# Usage:
#   STAGE=1 bash scripts/train_e2e_vae.sh
#   STAGE=2 bash scripts/train_e2e_vae.sh          # inits from stage-1 final ckpt
#   STAGE=3 bash scripts/train_e2e_vae.sh          # inits from stage-2 ckpt
# Overrides: OUTPUT_DIR, BATCH, STEPS, INIT_CKPT, TAG, VAE_GRAD_CKPT=0|1,
#   NUM_WORKERS, REPORT_TO, FILELIST_DIR, FLUX2_VAE, DATA_S1, DATA_HI, EXTRA_ARGS.

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/dist_launch.sh

STAGE="${STAGE:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-exps}"
TAG="${TAG:-auxkl2p5e8}"
BASE="e2e-vae-flux2-sitxl2${TAG:+-${TAG}}"
EXP1="e2e-vae-flux2-sitxl2-s1-256"
EXP2="${BASE}-s2-512"
EXP3="${BASE}-s3-512-dec"
S1_STEPS="${S1_STEPS:-100000}"
S2_STEPS="${S2_STEPS:-100000}"
S2_INIT_STEP="${S2_INIT_STEP:-70000}"
S3_STEPS="${S3_STEPS:-20000}"

FLUX2_VAE="${FLUX2_VAE:-${PRETRAINED_DIR:-pretrained}/flux2-vae}"
FILELIST_DIR="${FILELIST_DIR:-data/vae_filelists}"
DATA_S1="${DATA_S1:-configs/vae_data/stage1_mix.json}"
DATA_HI="${DATA_HI:-configs/vae_data/stage23_highres.json}"
ckpt_of() { printf "%s/%s/checkpoints/%07d.pt" "${OUTPUT_DIR}" "$1" "$2"; }

case "${STAGE}" in
    1)
        EXP="${EXP1}"; DATA="${DATA_S1}"; SIDE=256; STEPS="${STEPS:-${S1_STEPS}}"; BATCH="${BATCH:-1024}"
        INIT=""; GC_ABOVE=16
        AUX_KL="${AUX_KL:-1e-6}"
        STAGE_ARGS=(--mode e2e --vae-pretrained-path="${FLUX2_VAE}"
                    --vae-diffusion-coeff=0.1 --latent-reg-weight=0.1
                    --entropy-aux-kl-weight="${AUX_KL}"
                    --disc-start=1000000000 --vae-learning-rate=1e-4 --lr-schedule=constant)
        ;;
    2)
        EXP="${EXP2}"; DATA="${DATA_HI}"; SIDE=512; STEPS="${STEPS:-${S2_STEPS}}"; BATCH="${BATCH:-256}"
        INIT="${INIT_CKPT:-$(ckpt_of "${EXP1}" "${S1_STEPS}")}"; GC_ABOVE=4
        AUX_KL="${AUX_KL:-2.5e-8}"
        STAGE_ARGS=(--mode e2e --vae-diffusion-coeff=0.01 --latent-reg-weight=0.01
                    --entropy-aux-kl-weight="${AUX_KL}"
                    --disc-start=40000 --vae-learning-rate=5e-5 --lr-schedule=cosine)
        ;;
    3)
        EXP="${EXP3}"; DATA="${DATA_HI}"; SIDE=512; STEPS="${STEPS:-${S3_STEPS}}"; BATCH="${BATCH:-512}"
        INIT="${INIT_CKPT:-$(ckpt_of "${EXP2}" "${S2_INIT_STEP}")}"; GC_ABOVE=8
        STAGE_ARGS=(--mode decoder --disc-start=1000 --vae-learning-rate=5e-5 --lr-schedule=cosine)
        ;;
    *) echo "STAGE must be 1, 2 or 3 (got '${STAGE}')" >&2; exit 1;;
esac

if [ -n "${INIT}" ]; then
    [ -f "${INIT}" ] || { echo "init checkpoint not found: ${INIT} (set INIT_CKPT or OUTPUT_DIR)" >&2; exit 1; }
    STAGE_ARGS+=(--init-ckpt="${INIT}")
fi

NPROC="${NPROC:-8}"
NNODES="${NNODES:-1}"
WORLD=$((NNODES * NPROC))
if [ $((BATCH % WORLD)) -ne 0 ]; then
    echo "BATCH=${BATCH} is not divisible by ${WORLD} GPUs" >&2; exit 1
fi
PER_GPU=$((BATCH / WORLD))
VAE_GRAD_CKPT="${VAE_GRAD_CKPT:-$([ "${PER_GPU}" -gt "${GC_ABOVE}" ] && echo 1 || echo 0)}"
[ "${VAE_GRAD_CKPT}" = "1" ] && STAGE_ARGS+=(--vae-grad-ckpt)

python scripts/build_vae_filelists.py --data-json "${DATA}" --out-dir "${FILELIST_DIR}"

echo "=========================================================="
echo " STAGE ${STAGE}: ${EXP}   side=${SIDE} steps=${STEPS}"
echo " batch ${BATCH} = ${PER_GPU}/GPU x ${WORLD}  vae_grad_ckpt=${VAE_GRAD_CKPT}"
echo " data ${DATA}   init ${INIT:-${FLUX2_VAE}}"
echo " output ${OUTPUT_DIR}/${EXP}"
echo "=========================================================="

# shellcheck disable=SC2086
torchrun "${TORCHRUN_LAUNCH[@]}" --nproc_per_node="${NPROC}" train_e2e_vae.py \
    --exp-name="${EXP}" --output-dir="${OUTPUT_DIR}" --report-to="${REPORT_TO:-wandb}" \
    --wandb-project="${WANDB_PROJECT:-genfirst}" \
    --data-json="${DATA}" --filelist-dir="${FILELIST_DIR}" --side="${SIDE}" \
    --batch-size="${BATCH}" --max-train-steps="${STEPS}" --num-workers="${NUM_WORKERS:-8}" \
    --vae=flux2 --model=SiT-XL/2 --enc-type=dinov2-vit-b --encoder-depth=8 \
    --proj-coeff=0.5 --vae-align-proj-coeff=1.5 --latent-reg=entropy \
    --disc-weight=0.1 --checkpointing-steps="${CKPT_STEPS:-10000}" --sampling-steps=5000 \
    "${STAGE_ARGS[@]}" ${EXTRA_ARGS:-}
