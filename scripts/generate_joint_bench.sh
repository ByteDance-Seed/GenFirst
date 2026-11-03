#!/usr/bin/env bash
# Generate the GenEval or DPG-Bench prompt sets from a joint MMDiT checkpoint
# (t2i direction only), in the layout each harness scores.
#
# Scoring is a separate step: set the harnesses up with eval_bench/setup_eval_env.sh
# and score with eval_bench/scripts/eval_geneval.sh / eval_dpg.sh. See
# docs/T2I_EVAL.md.
#
# Sampling is sharded over ranks by prompt and skips prompts already on disk, so
# an interrupted run can just be relaunched with the same arguments.
#
# Usage:
#   CKPT=Prior_only/exps/<run>/checkpoints/0200000.pt BENCH=geneval \
#       GENEVAL_METADATA=<bench>/geneval/prompts/evaluation_metadata.jsonl \
#       bash scripts/generate_joint_bench.sh
#   CKPT=... BENCH=dpg DPG_CSV=<bench>/dpg_bench.csv bash scripts/generate_joint_bench.sh

set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="${CKPT:?set CKPT to a joint checkpoint (.pt)}"
[ -f "${CKPT}" ] || { echo "ERROR: checkpoint not found: ${CKPT}" >&2; exit 1; }

BENCH="${BENCH:-geneval}"
NPROC="${NPROC:-8}"
RUN_NAME="$(basename "$(dirname "$(dirname "${CKPT}")")")"
OUT_DIR="${OUT_DIR:-evals/${RUN_NAME}/${BENCH}}"

# Keep apostrophes out of the messages below: inside ${var:?...} an unpaired
# quote starts quoting and swallows the rest of the case body.
case "${BENCH}" in
  geneval)
    GENEVAL_METADATA="${GENEVAL_METADATA:?set GENEVAL_METADATA to the GenEval evaluation_metadata.jsonl}"
    [ -f "${GENEVAL_METADATA}" ] || { echo "ERROR: not found: ${GENEVAL_METADATA}" >&2; exit 1; }
    BENCH_ARGS=(--geneval-metadata="${GENEVAL_METADATA}")
    N_SAMPLES="${N_SAMPLES:-4}"
    ;;
  dpg)
    DPG_CSV="${DPG_CSV:?set DPG_CSV to the DPG-Bench dpg_bench.csv}"
    [ -f "${DPG_CSV}" ] || { echo "ERROR: not found: ${DPG_CSV}" >&2; exit 1; }
    BENCH_ARGS=(--dpg-csv="${DPG_CSV}")
    # The scorer crops each 2x2 grid back into four tiles, so this must be 4.
    N_SAMPLES=4
    ;;
  *) echo "BENCH must be geneval or dpg (got '${BENCH}')" >&2; exit 1;;
esac

EMA_ARGS=()
case "${USE_EMA:-1}" in 1|on|true|yes) EMA_ARGS=(--use-ema);; esac
OVERRIDE_ARGS=()
[ -n "${IMAGE_VAE_CKPT:-}" ] && OVERRIDE_ARGS+=(--image-vae-ckpt="${IMAGE_VAE_CKPT}")
[ -n "${IMAGE_VAE_PRETRAINED:-}" ] && OVERRIDE_ARGS+=(--image-vae-pretrained="${IMAGE_VAE_PRETRAINED}")
[ -n "${TEXT_VAE_PATH:-}" ] && OVERRIDE_ARGS+=(--text-vae-path="${TEXT_VAE_PATH}")
[ -n "${TOKENIZER_PATH:-}" ] && OVERRIDE_ARGS+=(--tokenizer-path="${TOKENIZER_PATH}")

echo ">>> ${BENCH} from ${CKPT}"
echo ">>> writing to ${OUT_DIR}"

torchrun --standalone --nproc_per_node="${NPROC}" generate_joint_bench.py \
    --ckpt="${CKPT}" \
    --bench="${BENCH}" \
    --out-dir="${OUT_DIR}" \
    "${BENCH_ARGS[@]}" \
    "${OVERRIDE_ARGS[@]}" \
    --n-samples="${N_SAMPLES}" \
    --prompts-per-batch="${PROMPTS_PER_BATCH:-2}" \
    --decode-chunk="${DECODE_CHUNK:-4}" \
    --t2i-steps="${T2I_STEPS:-50}" \
    --cfg-scale="${CFG_SCALE:-4.0}" \
    --seed="${SEED:-0}" \
    --limit="${LIMIT:-0}" \
    "${EMA_ARGS[@]}" \
    --allow-tf32
