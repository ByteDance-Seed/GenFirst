#!/usr/bin/env bash
# End-to-end GenEval: generate images from a checkpoint, then score them.
#
# GenEval measures compositional correctness (object presence, counts, colours,
# relative position) with an object detector, not a distribution distance.
#
# Prerequisites: bash eval_bench/setup_eval_env.sh
#
# Usage:
#   CKPT=exps/<run>/checkpoints/0200000.pt bash eval_bench/scripts/eval_geneval.sh
#   SMOKE=1 CKPT=... bash eval_bench/scripts/eval_geneval.sh   # 12 prompts only
#   SKIP_GEN=1 OUT=<dir> bash eval_bench/scripts/eval_geneval.sh  # score only
#
# SKIP_GEN=1 scores images that are already in OUT, which is how a generator
# other than run_gen_multi.sh gets scored -- the joint image-text model writes
# this same layout (see docs/JOINT.md).
#
# Every setting can be overridden from the environment.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(dirname "$HERE")"
cd "$BENCH_DIR"

SKIP_GEN="${SKIP_GEN:-0}"
if [ "$SKIP_GEN" = "1" ]; then
  CKPT="${CKPT:-(pre-generated)}"
else
  CKPT="${CKPT:?set CKPT=path to a train_t2i_frozen.py checkpoint}"
  [ -f "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT" >&2; exit 1; }
fi

NGPU="${NGPU:-8}"
NPP="${NPP:-4}"
# The paper's text-to-image inference settings (table 16).
STEPS="${STEPS:-50}"
CFG="${CFG:-7.0}"
GLOW="${GLOW:-0.35}"
GHIGH="${GHIGH:-1.0}"
BS="${BS:-16}"

DETECTOR_CKPT="${DETECTOR_CKPT:-detector/mask2former.pth}"
OBJECT_NAMES="${OBJECT_NAMES:-geneval/evaluation/object_names.txt}"
CLIP_PATH="${CLIP_PATH:-openai/clip-vit-large-patch14}"

# mmdet ships the Mask2Former config; resolve it from the installed package so
# the user does not have to locate it by hand.
if [ -z "${MODEL_CONFIG:-}" ]; then
  MODEL_CONFIG="$(python - <<'PY'
import os
try:
    import mmdet
    root = os.path.dirname(mmdet.__file__)
    cand = os.path.join(root, ".mim", "configs", "mask2former",
                        "mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py")
    print(cand if os.path.isfile(cand) else "")
except Exception:
    print("")
PY
)"
fi
if [ -z "${MODEL_CONFIG}" ] || [ ! -f "${MODEL_CONFIG}" ]; then
  echo "ERROR: could not locate the Mask2Former mmdet config." >&2
  echo "       Set MODEL_CONFIG=/path/to/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py" >&2
  exit 1
fi

if [ "${SMOKE:-0}" = "1" ]; then
  META="${META:-smoke_metadata.jsonl}"
  OUT="${OUT:-outputs/geneval_smoke}"
  NPP="${NPP_SMOKE:-1}"
else
  META="${META:-geneval/prompts/evaluation_metadata.jsonl}"
  OUT="${OUT:-outputs/geneval}"
fi

for f in "${META}" "${DETECTOR_CKPT}" "${OBJECT_NAMES}"; do
  [ -f "$f" ] || { echo "ERROR: missing required asset: $f (run setup_eval_env.sh)" >&2; exit 1; }
done

cat <<EOF
--------------------------------------------------------------------
 GenEval
   ckpt        : ${CKPT}
   prompts     : ${META}
   out         : ${OUT}
   n/prompt    : ${NPP}   steps: ${STEPS}   cfg: ${CFG} on [${GLOW}, ${GHIGH}]
--------------------------------------------------------------------
EOF

if [ "$SKIP_GEN" = "1" ]; then
  echo ">>> [1/2] skipped, scoring images already in ${OUT}"
  [ -d "$OUT" ] || { echo "ERROR: SKIP_GEN=1 but ${OUT} does not exist" >&2; exit 1; }
else
  echo ">>> [1/2] generating images"
  BENCH=geneval CKPT="$CKPT" OUT="$OUT" META="$META" \
    NGPU="$NGPU" NPP="$NPP" STEPS="$STEPS" CFG="$CFG" BS="$BS" \
    GLOW="$GLOW" GHIGH="$GHIGH" \
    bash "${BENCH_DIR}/run_gen_multi.sh"
fi

echo ">>> [2/2] scoring"
python score_geneval.py "$OUT" \
    --model-config "$MODEL_CONFIG" \
    --model-ckpt "$DETECTOR_CKPT" \
    --object-names "$OBJECT_NAMES" \
    --clip-path "$CLIP_PATH" \
    --outfile "${OUT}/results.jsonl"

echo ">>> results written to ${OUT}/results.jsonl"
