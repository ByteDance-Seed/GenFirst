#!/usr/bin/env bash
# Multi-GPU image generation for a benchmark prompt set, one shard per GPU.
#
# Usage:
#   BENCH=geneval CKPT=... OUT=... META=...    bash run_gen_multi.sh
#   BENCH=dpg     CKPT=... OUT=... PROMPTS=... bash run_gen_multi.sh
set -uo pipefail
export PYTHONUNBUFFERED=1

BENCH="${BENCH:?set BENCH=geneval|dpg}"
CKPT="${CKPT:?set CKPT=path to a train_t2i_frozen.py checkpoint}"
OUT="${OUT:?set OUT=output image dir}"
NGPU="${NGPU:-8}"
NPP="${NPP:-4}"
STEPS="${STEPS:-50}"
# The paper's text-to-image inference settings (table 16).
CFG="${CFG:-7.0}"
GLOW="${GLOW:-0.35}"
GHIGH="${GHIGH:-1.0}"
BS="${BS:-16}"
LOG_DIR="${LOG_DIR:-${OUT}/gen_logs}"
PRIOR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../Prior_only" && pwd)"

MODEL_ARGS=()
[ -n "${VAE_ARCH:-}" ] && MODEL_ARGS+=(--vae-arch "$VAE_ARCH")
[ -n "${VAE_CKPT:-}" ] && MODEL_ARGS+=(--vae-ckpt "$VAE_CKPT")
[ -n "${VAE_VIT_PATH:-}" ] && MODEL_ARGS+=(--vae-vit-path "$VAE_VIT_PATH")
[ -n "${LATENTS_STATS:-}" ] && MODEL_ARGS+=(--latents-stats "$LATENTS_STATS")
[ -n "${TEXT_ENCODER_NAME:-}" ] && MODEL_ARGS+=(--text-encoder-name "$TEXT_ENCODER_NAME")

if [ "$BENCH" = "geneval" ]; then
    SRC_ARG="--metadata ${META:?set META=evaluation_metadata.jsonl}"
else
    SRC_ARG="--prompt-dir ${PROMPTS:?set PROMPTS=DPG prompts dir}"
fi

mkdir -p "$LOG_DIR"
pids=()
for i in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$i python -u "$PRIOR_DIR/gen_for_bench.py" \
        --ckpt "$CKPT" --out "$OUT" --bench "$BENCH" $SRC_ARG \
        --n-per-prompt "$NPP" --num-steps "$STEPS" --cfg-scale "$CFG" \
        --guidance-low "$GLOW" --guidance-high "$GHIGH" \
        --batch-size "$BS" --shard-id "$i" --num-shards "$NGPU" \
        "${MODEL_ARGS[@]}" \
        > "${LOG_DIR}/gen_${BENCH}_shard${i}.log" 2>&1 &
    pids+=($!)
done
echo ">>> launched ${NGPU} shards (logs in ${LOG_DIR}): ${pids[*]}"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo ">>> all shards done (fail=$fail)"
exit $fail
