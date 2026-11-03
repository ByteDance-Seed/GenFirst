#!/usr/bin/env bash
# Image-to-text evaluation of a joint MMDiT: CIDEr / BLEU-4 / METEOR / ROUGE-L /
# SPICE / CLIPScore on COCO.
#
# Samples captions once for the union of COCO val2017 and the Karpathy test split
# (9407 unique images), then scores that single pass on both splits:
#
#   val2017        5000 images, disjoint from train2017 and so from the training
#                  mix. The clean number.
#   karpathy-test  the 5000 images the captioning literature reports on, but 4407
#                  of them live in train2017, so these carry image leakage and
#                  exist for comparability with published results.
#
# Usage:
#   CKPT=exps/<run>/checkpoints/0080000.pt COCO_ROOT=/data/coco \
#       KARPATHY_JSON=/data/coco_karpathy/dataset_coco.json \
#       OUT=outputs/i2t ARM=e2etext bash scripts/eval_caption.sh
#
#   # a frozen-stage run whose recorded image VAE path has moved
#   IMAGE_VAE_CKPT=/path/to/extracted_vae/<name>/<name>.pt ... bash scripts/eval_caption.sh
#
#   # score an existing pass again, e.g. after adding SPICE's JVM
#   SKIP_GEN=1 ... bash scripts/eval_caption.sh
#
# Scoring needs pycocoevalcap and a JVM on PATH (PTBTokenizer, METEOR and SPICE
# shell out to bundled jars). See docs/JOINT.md.

set -u
cd "$(dirname "$0")/.."

COCO_ROOT="${COCO_ROOT:?set COCO_ROOT to a COCO root with annotations/, train2017/, val2017/}"
KARPATHY_JSON="${KARPATHY_JSON:-}"
OUT="${OUT:-outputs/i2t}"
ARM="${ARM:-i2t}"
PRED="${PRED:-${OUT}/${ARM}.jsonl}"

# Decoding. The paper's setting is EMA weights, cfg 1.5, 20 Euler steps per block
# and 4 blocks -- 4 x block_size=16 is 64 tokens, ample for a COCO caption.
GPUS="${GPUS:-8}"
BATCH="${BATCH:-64}"
CFG="${CFG:-1.5}"
STEPS="${STEPS:-20}"
BLOCKS="${BLOCKS:-4}"
LIMIT="${LIMIT:-0}"

mkdir -p "$OUT"
export TOKENIZERS_PARALLELISM=false

if [ "${SKIP_GEN:-0}" = "1" ]; then
    [ -f "$PRED" ] || { echo "ERROR: SKIP_GEN=1 but $PRED does not exist" >&2; exit 1; }
    echo ">>> [1/2] skipped, scoring the $(wc -l < "$PRED") captions in $PRED"
else
    CKPT="${CKPT:?set CKPT to a train_joint_e2e.py checkpoint}"
    [ -f "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT" >&2; exit 1; }
    ARGS=(
        --ckpt "$CKPT" --use-ema --out "$PRED"
        --coco-root "$COCO_ROOT" --split "${SPLIT:-union}"
        --n-blocks "$BLOCKS" --i2t-steps "$STEPS" --cfg-scale "$CFG"
        --batch-size "$BATCH" --allow-tf32
    )
    [ -n "$KARPATHY_JSON" ] && ARGS+=(--karpathy-json "$KARPATHY_JSON")
    [ "$LIMIT" != "0" ] && ARGS+=(--limit "$LIMIT")
    # Only a run whose image VAE is not the one recorded in the checkpoint needs
    # this; load_everything fails loudly rather than encoding with random weights.
    [ -n "${IMAGE_VAE_CKPT:-}" ] && ARGS+=(--image-vae-ckpt "$IMAGE_VAE_CKPT")
    [ -n "${IMAGE_VAE_PRETRAINED:-}" ] && ARGS+=(--image-vae-pretrained "$IMAGE_VAE_PRETRAINED")
    [ -n "${TEXT_VAE_PATH:-}" ] && ARGS+=(--text-vae-path "$TEXT_VAE_PATH")
    [ -n "${TOKENIZER_PATH:-}" ] && ARGS+=(--tokenizer-path "$TOKENIZER_PATH")

    echo ">>> [1/2] generating captions"
    torchrun --standalone --nproc_per_node="$GPUS" eval_caption.py "${ARGS[@]}"
    # torchrun's own summary hides the traceback and returns before the file is
    # there, so check the artefact rather than falling through to score nothing.
    [ -f "$PRED" ] || { echo "ERROR: generation produced no $PRED" >&2; exit 1; }
fi

echo ">>> [2/2] scoring"
SPLITS="${SPLITS:-val2017 karpathy-test}"
for split in $SPLITS; do
    if [ "$split" = "karpathy-test" ] && [ -z "$KARPATHY_JSON" ]; then
        echo "  skipping karpathy-test, KARPATHY_JSON is not set"
        continue
    fi
    SARGS=(--pred "$PRED" --split "$split" --coco-root "$COCO_ROOT"
           --out "${OUT}/${ARM}.${split}.json")
    [ -n "$KARPATHY_JSON" ] && SARGS+=(--karpathy-json "$KARPATHY_JSON")
    [ -n "${SKIP_SPICE:-}" ] && SARGS+=(--skip-spice)
    [ -n "${SKIP_CLIPSCORE:-}" ] && SARGS+=(--skip-clipscore)
    python score_caption.py "${SARGS[@]}"
done
