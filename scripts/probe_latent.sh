#!/usr/bin/env bash
# ImageNet linear probe on a latent space (and, optionally, on a trained
# projector's output). Answers what caption NLL cannot: is the information
# linearly accessible semantics, or a private code only the LLM decodes?
#
# It is also the one understanding read-out that applies to every configuration,
# including runs trained with no understanding loss at all -- so it is what makes
# a baseline VAE and an --und-coeff run directly comparable.
#
# What to read (val top-1 %):
#   pixel/*      the floor. A linear map on downsampled RGB is already far above
#                chance, so no latent number means anything without this.
#   latent/*     the space the diffusion model consumes.
#   projector/*  the space the LLM consumes. With the VAE frozen the projector
#                adds no information, so any gain over latent/* is pure gain in
#                linear accessibility -- i.e. real semantic decoding.
#   vit/pool1    the ceiling: Qwen3-VL's own ViT features, globally pooled.
#
# pool<g> averages the feature map to g x g (conventional, but coarser grids also
# shrink the dimensionality). smooth<g> averages to g x g and upsamples back, so
# the dimensionality is fixed and only spatial frequency changes: semantics
# survives it, a high-frequency private code does not.
#
# Data layout:
#   TRAIN_DIR  ImageNet c2i parquet shards (see docs/DATA.md)
#   LABEL_MAP  one class name per line, canonical 0-999 order
#   VAL_DIR    ImageNet val as <root>/<synset>/*.JPEG
#
# Usage:
#   TRAIN_DIR=... LABEL_MAP=... VAL_DIR=... bash scripts/probe_latent.sh
#   # an end-to-end tuned VAE instead of the pretrained one:
#   VAE_CKPT=exps/<run>/checkpoints/0400000.pt EXP=probe-e2e-flux \
#       TRAIN_DIR=... LABEL_MAP=... VAL_DIR=... bash scripts/probe_latent.sh
#   # + projector and ViT ceiling (needs an --und-type nll checkpoint):
#   MODE=full PROJ_CKPT=exps/<run>/checkpoints/0400000.pt \
#       TRAIN_DIR=... LABEL_MAP=... VAL_DIR=... bash scripts/probe_latent.sh
#   # re-solve on cached features, no re-extraction:
#   SKIP_EXTRACT=1 EXP=probe-e2e-flux ... bash scripts/probe_latent.sh
#
# MODE=full loads Qwen3-VL and so needs transformers >= 4.57 (see docs/UND.md).

set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${MODE:-latent}"
case "${MODE}" in
    latent) TARGETS="latent,pixel" ;;
    full)   TARGETS="latent,pixel,projector,vit" ;;
    *) echo "MODE must be 'latent' or 'full' (got '${MODE}')" >&2; exit 1 ;;
esac
if [ "${MODE}" = "full" ] && [ -z "${PROJ_CKPT:-}" ]; then
    echo "MODE=full needs PROJ_CKPT=<a train_e2e.py checkpoint trained with --und-type nll>" >&2
    exit 1
fi

TRAIN_DIR="${TRAIN_DIR:?set TRAIN_DIR to the ImageNet c2i parquet root (see docs/DATA.md)}"
LABEL_MAP="${LABEL_MAP:?set LABEL_MAP to a file with one class name per line, 0-999 order}"
VAL_DIR="${VAL_DIR:?set VAL_DIR to the ImageNet val root (<root>/<synset>/*.JPEG)}"
FLUX_VAE="${FLUX_VAE:-${PRETRAINED_DIR:-pretrained}/flux-vae}"
VLM="${VLM:-Qwen/Qwen3-VL-2B-Instruct}"
NPROC="${NPROC:-8}"
EXP="${EXP:-probe-flux-latent}"

[ -d "${FLUX_VAE}" ] || { echo "ERROR: FLUX VAE dir not found: ${FLUX_VAE}" >&2; exit 1; }

# 100k train / 10k val is the reported protocol; absolute numbers move with it, so
# change it only if you are not comparing against the paper's table. n-train should
# in any case stay ~2x the largest probe dimension (16384 for a 16ch 32x32 latent),
# otherwise ridge is fitting an underdetermined system.
ARGS=(
    --imagenet-train-dir="${TRAIN_DIR}"
    --label-map="${LABEL_MAP}"
    --imagenet-val-dir="${VAL_DIR}"
    --n-train="${N_TRAIN:-100000}"
    --n-val="${N_VAL:-10000}"
    --resolution=256
    --batch-size="${BATCH:-64}"
    --num-workers="${WORKERS:-8}"
    --targets="${TARGETS}"
    --vae="${VAE:-flux}"
    --vae-pretrained-path="${FLUX_VAE}"
    --vae-ckpt="${VAE_CKPT:-}"
    --vlm-path="${VLM}"
    --projector-ckpt="${PROJ_CKPT:-}"
    --pool-grids="${POOL_GRIDS:-full,4,2,1}"
    --smooth-grids="${SMOOTH_GRIDS:-4,2}"
    --exp-name="${EXP}"
)

# The solve is single-process; extraction is the only distributed part.
if [ -n "${SKIP_EXTRACT:-}" ]; then
    python probe_latent.py --skip-extract "${ARGS[@]}"
else
    torchrun --standalone --nproc_per_node="${NPROC}" probe_latent.py "${ARGS[@]}"
fi
