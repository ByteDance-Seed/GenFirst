#!/usr/bin/env bash
# End-to-end DPG-Bench: generate images, stitch into 2x2 grids, then score.
#
# DPG-Bench measures dense prompt following on long, multi-clause prompts using
# a VQA model. The official scorer expects a 2x2 grid of 4 samples per prompt,
# which is why the stitch step exists. Grid filenames must be <item_id>.png to
# match the item_id column of dpg_bench.csv.
#
# Prerequisites: bash eval_bench/setup_eval_env.sh
#
# Usage:
#   CKPT=exps/<run>/checkpoints/0200000.pt bash eval_bench/scripts/eval_dpg.sh
#   SKIP_GEN=1 SKIP_STITCH=1 GRID_DIR=<dir> OUT=<dir> bash .../eval_dpg.sh
#
# SKIP_GEN=1 scores images another generator produced. Add SKIP_STITCH=1 when
# GRID_DIR already holds <item_id>.png 2x2 grids rather than per-sample images:
# the joint image-text model writes grids directly (see docs/JOINT.md).
#
# RESOLUTION is the size of a SINGLE sample (the stitched grid is 2x), and must
# match the resolution the checkpoint was trained at.

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
# The paper's text-to-image inference settings (table 16).
STEPS="${STEPS:-50}"
CFG="${CFG:-7.0}"
GLOW="${GLOW:-0.35}"
GHIGH="${GHIGH:-1.0}"
BS="${BS:-16}"
# The scorer requires exactly 4 samples per prompt (2x2 grid).
NPP=4

PROMPTS="${PROMPTS:-ELLA/dpg_bench/prompts}"
DPG_CSV="${DPG_CSV:-ELLA/dpg_bench/dpg_bench.csv}"
SCORER="${SCORER:-ELLA/dpg_bench/compute_dpg_bench.py}"
OUT="${OUT:-outputs/dpg}"
RAW_DIR="${RAW_DIR:-${OUT}/raw}"
GRID_DIR="${GRID_DIR:-${OUT}/grids}"

for f in "${DPG_CSV}" "${SCORER}"; do
  [ -f "$f" ] || { echo "ERROR: missing required asset: $f (run setup_eval_env.sh)" >&2; exit 1; }
done
[ -d "${PROMPTS}" ] || { echo "ERROR: missing DPG prompt dir: ${PROMPTS}" >&2; exit 1; }

cat <<EOF
--------------------------------------------------------------------
 DPG-Bench
   ckpt        : ${CKPT}
   prompts     : ${PROMPTS}
   out         : ${OUT}
   n/prompt    : ${NPP} (fixed)   steps: ${STEPS}   cfg: ${CFG} on [${GLOW}, ${GHIGH}]
--------------------------------------------------------------------
EOF

if [ "$SKIP_GEN" = "1" ]; then
  echo ">>> [1/3] skipped, scoring images already on disk"
else
  echo ">>> [1/3] generating images"
  BENCH=dpg CKPT="$CKPT" OUT="$RAW_DIR" PROMPTS="$PROMPTS" \
    NGPU="$NGPU" NPP="$NPP" STEPS="$STEPS" CFG="$CFG" BS="$BS" \
    GLOW="$GLOW" GHIGH="$GHIGH" \
    bash "${BENCH_DIR}/run_gen_multi.sh"
fi

if [ "${SKIP_STITCH:-0}" = "1" ]; then
  echo ">>> [2/3] skipped, ${GRID_DIR} already holds 2x2 grids"
  [ -d "$GRID_DIR" ] || { echo "ERROR: SKIP_STITCH=1 but ${GRID_DIR} does not exist" >&2; exit 1; }
else
  echo ">>> [2/3] stitching 2x2 grids"
  python stitch_dpg.py --src "$RAW_DIR" --dst "$GRID_DIR"
fi

# The official scorer is an accelerate script; --resolution is the size of ONE
# quadrant (the stitched grid is 2x that). It shards images across processes.
echo ">>> [3/3] scoring with the official DPG-Bench scorer"
mkdir -p "$OUT"
accelerate launch --num_machines 1 --num_processes "$NGPU" --multi_gpu \
    --mixed_precision fp16 --main_process_port "${PORT:-29500}" \
    "$(realpath "$SCORER")" \
    --image-root-path "$(realpath "$GRID_DIR")" \
    --resolution "${RESOLUTION:-256}" \
    --pic-num 4 \
    --vqa-model mplug \
    --csv "$(realpath "$DPG_CSV")" \
    --res-path "$(realpath "$OUT")/dpg_results.txt"

echo ">>> results written to ${OUT}/dpg_results.txt"
echo "    (the final 'DPG-Bench score' line is the headline number)"
