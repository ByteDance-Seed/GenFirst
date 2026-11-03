#!/usr/bin/env bash
# MAE-protocol ImageNet linear probe (probe_linear_mae.py) on a frozen encoder.
#
# This is the number that can sit next to published linear-probe results: full
# 1.28M-image IN1K train split, LARS, 90 epochs, RandomResizedCrop, BN+Linear
# head. The ridge probe in scripts/probe_latent.sh stays useful for fast
# feature-space ranking, but its absolute values are not comparable to MAE's.
#
# Data layout (original JPEGs, ImageFolder):
#   IN_ROOT/train/<synset>/*.JPEG
#   IN_ROOT/val/<synset>/*.JPEG
#
# Usage:
#   # the run: VAE_CKPT is a train_e2e.py checkpoint or an extracted VAE
#   IN_ROOT=/data/imagenet VAE_CKPT=exps/<run>/checkpoints/0100000.pt \
#       EXP=mae-probe-siglip bash scripts/probe_linear_mae.sh
#
#   # no checkpoint = the stock Qwen3-VL tower, the reference for what
#   # end-to-end training cost the encoder (no latent tap: it is untrained)
#   IN_ROOT=/data/imagenet EXP=mae-probe-frozen bash scripts/probe_linear_mae.sh
#
#   # measure throughput and project the full run, then exit
#   BENCH=30 IN_ROOT=... VAE_CKPT=... bash scripts/probe_linear_mae.sh
#
#   # optional: copy ImageNet onto local disk first (~150 GB). Worth it when the
#   # source is network storage -- the encoder can consume images faster than a
#   # FUSE mount serves them, and a 90-epoch run reads the split 90 times.
#   STAGE=1 SRC_ROOT=<network>/imagenet LOCAL_ROOT=/data/imagenet \
#       STAGE_ONLY=1 bash scripts/probe_linear_mae.sh
#
# Staging must happen in the same job as the run wherever local disk does not
# survive between jobs: use STAGE=1 without STAGE_ONLY in the job command.
#
# A qwenvit encoder needs transformers >= 4.57 for Qwen3-VL (see docs/UND.md).

set -euo pipefail
cd "$(dirname "$0")/.."

# Local copy if one was staged, the source otherwise.
SRC_ROOT="${SRC_ROOT:-${IN_ROOT:-}}"
LOCAL_ROOT="${LOCAL_ROOT:-${IN_ROOT:-}}"
if [ -z "${SRC_ROOT}" ]; then
    echo "set IN_ROOT to the ImageNet root (<root>/{train,val}/<synset>/*.JPEG)" >&2
    exit 1
fi

if [ -n "${STAGE:-}" ]; then
    [ "${LOCAL_ROOT}" != "${SRC_ROOT}" ] || {
        echo "STAGE=1 needs LOCAL_ROOT different from SRC_ROOT" >&2; exit 1; }
    for split in val train; do
        mkdir -p "${LOCAL_ROOT}/${split}"
        # One cp per class dir, 32 at a time: a single-threaded recursive copy of
        # 1.28M files over a network mount takes many hours.
        ls "${SRC_ROOT}/${split}" | xargs -P "${STAGE_JOBS:-32}" -I{} \
            cp -rn "${SRC_ROOT}/${split}/{}" "${LOCAL_ROOT}/${split}/"
        # Count images, not class dirs: an interrupted copy leaves all 1000 dirs
        # in place and only partly filled, and the probe's labels are positions in
        # the sorted synset list, so a short split silently renames every class.
        case "${split}" in train) want=1281167 ;; val) want=50000 ;; esac
        got=$(find "${LOCAL_ROOT}/${split}" -type f \
              \( -iname '*.jpeg' -o -iname '*.jpg' -o -iname '*.png' \) | wc -l)
        echo "staged ${split}: $(ls "${LOCAL_ROOT}/${split}" | wc -l) class dirs, ${got}/${want} images"
        if [ "${got}" -lt "${want}" ]; then
            echo "  INCOMPLETE -- re-run this STAGE step, cp -rn resumes where it stopped" >&2
        fi
    done
    du -sh "${LOCAL_ROOT}"
    if [ -n "${STAGE_ONLY:-}" ]; then exit 0; fi
fi

if [ -d "${LOCAL_ROOT}/train" ] && [ -d "${LOCAL_ROOT}/val" ]; then
    ROOT="${LOCAL_ROOT}"
else
    ROOT="${SRC_ROOT}"
fi
[ -d "${ROOT}/train" ] && [ -d "${ROOT}/val" ] || {
    echo "ERROR: ${ROOT} has no train/ and val/ subdirectories" >&2; exit 1; }

VAE="${VAE:-qwenvit}"
CKPT="${VAE_CKPT:-}"
if [ -n "${CKPT}" ] && [ ! -f "${CKPT}" ]; then
    echo "ERROR: checkpoint not found: ${CKPT}" >&2; exit 1
fi
if [ -n "${CKPT}" ]; then
    # latent/* is what the diffusion model consumes, so it belongs in the table.
    VARIANTS="${VARIANTS:-latent/full,trunk/pool4,trunk/pool1,merged/pool1}"
else
    # No trained bottleneck on the stock tower, so no latent tap.
    VARIANTS="${VARIANTS:-trunk/pool4,trunk/pool1,merged/pool1}"
fi

NPROC="${NPROC:-8}"
EXP="${EXP:-mae-probe}"
echo ">>> ${EXP}: ${VAE} ${CKPT:-(stock tower)} | data ${ROOT} | ${NPROC} gpus"

ARGS=(
    --imagenet-train-dir="${ROOT}/train"
    --imagenet-val-dir="${ROOT}/val"
    --vae="${VAE}"
    --vae-ckpt="${CKPT}"
    --vae-vit-path="${VLM:-Qwen/Qwen3-VL-2B-Instruct}"
    --variants="${VARIANTS}"
    --resolution="${RES:-256}"
    --epochs="${EPOCHS:-90}"
    --warmup-epochs="${WARMUP:--1}"
    --batch-size="${BATCH:-512}"
    --eff-batch="${EFF_BATCH:-16384}"
    --blr="${BLR:-0.1}"
    --num-workers="${WORKERS:-12}"
    --eval-every="${EVAL_EVERY:-10}"
    --exp-name="${EXP}"
    # Node-local on purpose: it caches local file paths, while exps/ is often
    # shared storage.
    --index-dir="${INDEX_DIR:-.cache/imagenet-index}"
)
if [ -n "${BENCH:-}" ]; then ARGS+=(--bench-steps="${BENCH}"); fi

# Multi-node: swap --standalone for your launcher's rendezvous arguments (see
# README). One single-node job per setting is usually better -- the settings are
# independent, and every node otherwise stages its own copy of ImageNet inside
# torchrun's rendezvous timeout.
torchrun --standalone --nproc_per_node="${NPROC}" probe_linear_mae.py "${ARGS[@]}"
