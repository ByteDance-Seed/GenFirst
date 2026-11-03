#!/usr/bin/env bash
# End-to-end stage with Qwen3-VL's vision tower as the VAE encoder.
#
# Why: an ImageNet ridge linear probe reads 67.1% off this tower's features and
# 3.1% off a FLUX latent, and training a projector on top of a frozen FLUX latent
# closes almost none of that gap. Shaping a reconstruction-only code into something
# a language model can read is a problem of *creating* semantics; starting from the
# tower makes it a problem of *preserving* semantics through a bottleneck.
#
# Latent geometry is chosen to change nothing downstream. The tower is patch-16 and
# each token is bottlenecked into a 2x2 block, so at 256px the latent is 16x32x32 --
# the same shape as FLUX's, and the same rate: 16*32*32 = 16384 either way. Each ViT
# token spends 64 numbers on the 16x16x3 = 768 pixels it covers, a 12x compression,
# exactly FLUX's 192 -> 16. So MMDiT-L/2, its latent-normalising BN and the loss
# weights are all the baseline's, and the comparison is apples to apples. LATENT_CH
# is left as a knob, but 64 would be 4x FLUX's rate rather than a match for it.
#
# THREE phases, because this encoder arrives pretrained but with NO decoder:
#
#   PHASE=0 -> 256,  30K, encoder FROZEN, decoder + bottleneck only, reconstruction
#              only (--no-gen-loss, no REPA, weak KL), GAN from 10K.
#   PHASE=1 -> 256, 400K, encoder UNFROZEN at ENC_LR, joint reconstruction +
#              denoising + REPA, lambda 0.1, warm-started from phase 0.
#   PHASE=2 -> 256, 200K, same joint recipe, lambda 0.01, warm-started from phase 1.
#
# Phase 0 has no counterpart in the conv-VAE recipe, where encoder and decoder are
# trained together from scratch. Here the decoder starts from nothing against an
# encoder that is already good, so it is fitted first; letting the denoising and
# GAN gradients in before reconstruction works at all just drags the tower out of
# the geometry it came with.
#
# Phase 0 uses a WEAK KL, not the entropy regulariser, and this is the one thing
# that is easy to get wrong. Entropy maximisation exists to stop the *denoising*
# gradient from collapsing the latent; with --no-gen-loss there is no such gradient,
# so at weight 0.1 it is free to inflate the posterior instead. Measured: variance
# 0.0025 -> 2.29 within 1000 steps and pinned there, leaving ~70% of the latent's
# variance as sampling noise; the bottleneck then spent 100K steps raising the
# signal just to out-shout it and reconstruction saturated at 20K anyway
# (PSNR 16.6, SSIM 0.374, LPIPS 0.336 -- far off a usable AE). train_e2e.py now
# refuses the combination rather than letting it run.
#
# Phase 0 also runs with --vae-align-proj-coeff=0 on purpose. REPA's target is a
# projection inside the MMDiT, and with --no-gen-loss the MMDiT is never stepped,
# so those projectors would stay at initialisation and the "alignment" would be to
# a random projection of DINOv2. The MMDiT still runs its cheap align-only forward,
# which is what keeps its BatchNorm tracking the new latent's statistics so phase 1
# inherits sane normalisation.
#
# UND selects the understanding objective (phases 1 and 2 only). UND=off is the
# control arm: the same phase with the same warm start, differing only in this loss,
# so the pair is a clean ablation of "does an understanding prior help".
#   off       -- reconstruction + denoising + REPA only
#   siglip    -- sigmoid pairwise contrastive + caption-token cls on a weak pool
#   nll       -- caption NLL under Qwen's frozen LLM, read off the *latent* through
#                the tower's own frozen merger
#   nll-trunk -- the same NLL read off the *trunk*: understanding and the bottleneck
#                become sibling branches instead of being nested. No temperature, no
#                projector, no pooling head, so there is no trainable parameter to
#                move except the trunk itself, and it starts at the base model's own
#                PMI instead of climbing from zero.
#
# Needs transformers >= 4.57 for Qwen3-VL; see docs/UND.md.
#
# Usage:
#   PHASE=0 DATA=/path/to/parquet_roots bash scripts/train_qwenvit_e2e.sh
#   PHASE=1 DATA=... bash scripts/train_qwenvit_e2e.sh                  # control arm
#   PHASE=1 UND=nll UND_BATCH=8 DATA=... bash scripts/train_qwenvit_e2e.sh
#   PHASE=0 BATCH_SIZE=64 DATA=... bash scripts/train_qwenvit_e2e.sh    # local smoke
#
# See docs/UND.md for the read-outs and for how to calibrate ENC_LR against
# und/pmi_frac.

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/dist_launch.sh

PHASE="${PHASE:-0}"
NPROC="${NPROC:-8}"
BATCH_SIZE="${BATCH_SIZE:-256}"
OUTPUT_DIR="${OUTPUT_DIR:-exps}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-50000}"
WANDB_PROJECT="${WANDB_PROJECT:-genfirst}"
REPORT_TO="${REPORT_TO:-wandb}"
MODEL="${MODEL:-MMDiT-L/2}"

VIT="${VIT:-Qwen/Qwen3-VL-2B-Instruct}"
LATENT_CH="${LATENT_CH:-16}"
DATA="${DATA:?set DATA to comma-separated parquet roots (see docs/T2I.md)}"
CAPTION_COLUMNS="${CAPTION_COLUMNS:-caption,prompt,text}"

BASE="qwenvit-e2e-ae-mmdit-l2-t2i"
# 30K, not 100K: reconstruction saturates around 20K here. Re-measure the
# saturation point before spending more.
S0_STEPS="${S0_STEPS:-30000}"
S1_STEPS="${S1_STEPS:-400000}"
S2_STEPS="${S2_STEPS:-200000}"
# TAG keeps a new attempt out of an old run's directory.
TAG="${TAG:-}"

# Qwen3-VL is only in transformers >= 4.57. Fail here with something actionable
# rather than 8 ranks deep in an interleaved torchrun traceback.
python - <<'PY' || exit 1
import sys
try:
    import transformers
    from transformers import Qwen3VLForConditionalGeneration  # noqa: F401
except Exception:
    import transformers
    sys.exit(
        f"ERROR: --vae qwenvit needs Qwen3-VL, which requires transformers>=4.57 "
        f"(found {getattr(transformers, '__version__', 'none')}).\n"
        f"       pip install -U 'transformers>=4.57' 'tokenizers>=0.22'\n"
        f"       See docs/UND.md; consider a separate environment, since this also "
        f"upgrades tokenizers."
    )
PY

# The pretrained tower gets its own, much smaller rate, and this is the single most
# important setting in the joint phases. Adam moves every coordinate by about lr in
# the early steps whatever its gradient is, so at the decoder's 1e-4 the tower
# leaves the geometry Qwen's frozen merger and language model were trained to read
# almost immediately: measured, caption PMI falls from 1.13 to 0.005 nats/token by
# step 6, and lowering the understanding coefficient 100x barely delays it, because
# the drift is driven by the reconstruction and denoising gradients rather than the
# understanding one. At 1e-5 PMI holds at 0.95-1.35 for at least 100 steps, and at
# 1e-6 for 30. 1e-5 is the default because it buys 10x more encoder adaptation at no
# measured cost to readability. und/pmi_frac is the instrument: if it decays, lower
# this.
VAE_LR="${VAE_LR:-1e-4}"
ENC_LR="${ENC_LR:-1e-5}"

UND="${UND:-off}"
# Phase 2 drops every prior weight on the 0.1 -> 0.01 schedule, the understanding
# term included. Override with UND_COEFF= for a different value.
if [ "${PHASE}" = "2" ]; then
    UND_COEFF="${UND_COEFF:-0.01}"
else
    UND_COEFF="${UND_COEFF:-0.1}"
fi
UND_WARMUP="${UND_WARMUP:-2000}"

UND_ARGS=()
SUFFIX=""
if [ "${PHASE}" != "0" ]; then
  case "${UND}" in
    off) ;;
    siglip)
        SUFFIX="-siglip"
        UND_ARGS=(--und-type=siglip --und-coeff="${UND_COEFF}"
                  --und-warmup-steps="${UND_WARMUP}"
                  --und-head-mode=attnpool --und-cls-coeff=1.0
                  --und-text-projection)
        ;;
    nll)
        SUFFIX="-nll"
        # The 2B forward *and* backward is why this needs a sub-batch: the joint
        # phase already sits near 57GB/GPU before the language model is loaded.
        UND_ARGS=(--und-type=nll --und-coeff="${UND_COEFF}"
                  --und-warmup-steps="${UND_WARMUP}"
                  --und-via-merger --und-projector-mode=linear
                  --und-batch-size="${UND_BATCH:-8}"
                  --und-pmi-mode=monitor)
        ;;
    nll-trunk)
        SUFFIX="-nlltrunk"
        # No warm-up and no projector: there is no head to warm up, and at step 0
        # the trunk is still the pretrained one, which makes this the one gradient
        # in the system that is trustworthy immediately.
        UND_ARGS=(--und-type=nll --und-tap=trunk --und-coeff="${UND_COEFF}"
                  --und-deepstack
                  --und-batch-size="${UND_BATCH:-8}"
                  --und-pmi-mode=monitor
                  --und-pmi-ceiling="${UND_PMI_CEILING:-0.94}")
        ;;
    *)
        echo "UND must be off, siglip, nll or nll-trunk (got '${UND}')" >&2; exit 1;;
  esac
elif [ "${UND}" != "off" ]; then
    echo "ERROR: UND=${UND} but phase 0 has no denoising loss and no MMDiT step," >&2
    echo "       so there is nothing for an understanding prior to ride along with." >&2
    echo "       Run phase 0 with UND=off, then set UND on phase 1." >&2
    exit 1
fi

EXP0="${BASE}-phase0-256-frozen-enc${TAG}"
EXP1="${BASE}-phase1-256-joint${SUFFIX}${TAG}"
EXP2="${BASE}-phase2-256-joint${SUFFIX}${TAG}"
ckpt_of() { printf "%s/%s/checkpoints/%07d.pt" "${OUTPUT_DIR}" "$1" "$2"; }

require_ckpt() {
    [ -f "$1" ] || { echo "ERROR: checkpoint not found: $1" >&2; echo "       $2" >&2; exit 1; }
}

case "${PHASE}" in
  0)
    EXP="${EXP0}"
    PRIOR_WEIGHT="n/a (reconstruction only)"
    LOSS_CFG="${LOSS_CFG:-configs/l1_lpips_kl_gan_disc10k.yaml}"
    PHASE_ARGS=(--resolution=256 --max-train-steps="${S0_STEPS}"
                --batch-size="${BATCH_SIZE}"
                --vae-freeze-encoder
                --no-gen-loss
                --vae-align-proj-coeff=0
                --latent-reg=kl --latent-reg-weight=1e-6)
    ;;
  1)
    EXP="${EXP1}"
    PRIOR_WEIGHT="${PRIOR_WEIGHT:-0.1}"
    LOSS_CFG="${LOSS_CFG:-configs/l1_lpips_entropy_gan.yaml}"
    INIT="${INIT_CKPT:-$(ckpt_of "${EXP0}" "${S0_STEPS}")}"
    require_ckpt "${INIT}" "Run PHASE=0 first, or pass INIT_CKPT=<path>."
    PHASE_ARGS=(--resolution=256 --max-train-steps="${S1_STEPS}"
                --batch-size="${BATCH_SIZE}"
                --no-vae-freeze-encoder
                --vae-align-proj-coeff=1.5
                --vae-diffusion-grad
                --vae-diffusion-coeff="${PRIOR_WEIGHT}"
                --latent-reg=entropy --latent-reg-weight="${PRIOR_WEIGHT}"
                --entropy-aux-kl-weight=1e-6
                --init-ckpt="${INIT}")
    ;;
  2)
    EXP="${EXP2}"
    PRIOR_WEIGHT="${PRIOR_WEIGHT:-0.01}"
    LOSS_CFG="${LOSS_CFG:-configs/l1_lpips_entropy_gan.yaml}"
    INIT="${INIT_CKPT:-$(ckpt_of "${EXP1}" "${S1_STEPS}")}"
    require_ckpt "${INIT}" "Phase 1 may have stopped at a different step; pass S1_STEPS= or INIT_CKPT=<path>."
    PHASE_ARGS=(--resolution=256 --max-train-steps="${S2_STEPS}"
                --batch-size="${BATCH_SIZE}"
                --no-vae-freeze-encoder
                --vae-align-proj-coeff=1.5
                --vae-diffusion-grad
                --vae-diffusion-coeff="${PRIOR_WEIGHT}"
                --latent-reg=entropy --latent-reg-weight="${PRIOR_WEIGHT}"
                --entropy-aux-kl-weight=1e-6
                --init-ckpt="${INIT}")
    ;;
  *)
    echo "PHASE must be 0, 1 or 2 (got '${PHASE}')" >&2
    exit 1;;
esac

[ -f "${LOSS_CFG}" ] || { echo "ERROR: loss config not found: ${LOSS_CFG}" >&2; exit 1; }

cat <<EOF
--------------------------------------------------------------------
 Qwen3-VL-encoder end-to-end stage, phase ${PHASE}
   exp-name        : ${EXP}
   encoder         : ${VIT} $([ "${PHASE}" = "0" ] && echo "(FROZEN)" || echo "(unfrozen, lr ${ENC_LR})")
   latent          : ${LATENT_CH}x32x32 (rate $((LATENT_CH * 32 * 32)); FLUX is 16384)
   loss            : $([ "${PHASE}" = "0" ] && echo "reconstruction only, weak KL" || echo "joint, lambda=${PRIOR_WEIGHT}")
   understanding   : $([ "${PHASE}" = "0" ] && echo "n/a" || echo "${UND}$([ "${UND}" != "off" ] && echo " coeff=${UND_COEFF}")")
   loss-cfg        : ${LOSS_CFG}
   batch           : global=${BATCH_SIZE} -> per-gpu=$(( BATCH_SIZE / NPROC ))
   data            : ${DATA}
--------------------------------------------------------------------
EOF
[ "${PHASE}" != "0" ] && echo ">>> warm-start from: ${INIT}"

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
    --model="${MODEL}" \
    --text-encoder="clip" \
    --cfg-prob=0.1 \
    --checkpointing-steps="${CHECKPOINTING_STEPS}" \
    --sampling-steps=5000 \
    --no-compile \
    --loss-cfg-path="${LOSS_CFG}" \
    --vae="qwenvit" \
    --vae-vit-path="${VIT}" \
    --latent-channels="${LATENT_CH}" \
    --vae-ckpt="" \
    --vae-learning-rate="${VAE_LR}" \
    --vae-encoder-lr="${ENC_LR}" \
    --vae-decoder-width="${DEC_WIDTH:-1024}" \
    --vae-decoder-depth="${DEC_DEPTH:-12}" \
    --enc-type="dinov2-vit-b" \
    --proj-coeff=0.5 \
    --encoder-depth=8 \
    --bn-momentum=0.1 \
    --eval-steps=0 \
    --no-eval-at-start \
    --exp-name="${EXP}" \
    "${PHASE_ARGS[@]}" \
    "${UND_ARGS[@]}"
