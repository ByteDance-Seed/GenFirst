#!/usr/bin/env bash
# =============================================================================
# One-time setup for the T2I compositional benchmarks (GenEval + DPG-Bench).
#
#   GenEval : mmengine + mmcv (source build) + mmdet, a Mask2Former Swin-S COCO
#             detector checkpoint, and CLIP for the colour/attribute classifier.
#   DPG     : modelscope + the mPLUG VQA model, with modelscope patched so it
#             imports without fairseq.
#
# Both stacks are heavy and version-sensitive. They do NOT coexist well with the
# training environment, so installing them into a separate conda/venv is
# recommended. torch/torchvision are assumed to be already installed and are
# deliberately left untouched: mmcv is compiled against whatever torch is here.
#
# Idempotent: safe to re-run, skips anything already installed or downloaded.
#
# Usage:
#   bash setup_eval_env.sh
#   SKIP_MMCV_BUILD=1 bash setup_eval_env.sh   # skip the slow mmcv source build
#   SKIP_DOWNLOADS=1  bash setup_eval_env.sh   # skip model/asset downloads
#
# Set TORCH_CUDA_ARCH_LIST to your GPU's compute capability before running:
#   V100=7.0  A100=8.0  A10=8.6  H100=9.0
# =============================================================================
set -uo pipefail

# mmcv compiles CUDA ops, so it needs the arch list and a CUDA toolkit whose
# version matches the one torch was built against.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-32}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"
PY="${PY:-python}"

pipq() { $PY -m pip install -q "$@"; }

echo "############################################################"
echo "# T2I eval env setup   (dir=${HERE})"
$PY -c "import torch;print('# torch',torch.__version__,'cuda',torch.version.cuda)"
echo "# TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}  CUDA_HOME=${CUDA_HOME}"
echo "############################################################"

echo ">>> [1/6] small deps"
pipq \
    pandas \
    unicodedata2 zhconv ftfy sacremoses jieba \
    simplejson sortedcontainers addict pycocotools \
    cython

# -----------------------------------------------------------------------------
# GenEval detector stack. Prebuilt mmcv wheels are built against CUDA 11 and
# fail with "libcudart.so.11.0" on a cu12 torch, so mmcv is built from source.
# mmengine is pinned to 0.10.7 to avoid an "Adafactor already registered" clash
# with torch >= 2.5.
# -----------------------------------------------------------------------------
echo ">>> [2/6] GenEval detector stack (mmengine/mmcv/mmdet)"
pipq "mmengine==0.10.7"

if [ "${SKIP_MMCV_BUILD:-0}" = "1" ]; then
    echo "    SKIP_MMCV_BUILD=1 -> not (re)building mmcv"
elif $PY -c "import mmcv,torch; assert mmcv.__version__=='2.1.0'; import mmcv.ops" 2>/dev/null; then
    echo "    mmcv 2.1.0 (with ops) already importable -> skip build"
else
    echo "    building mmcv 2.1.0 from source (this takes several minutes)..."
    $PY -m pip uninstall -y mmcv mmcv-full 2>/dev/null || true
    MMCV_WITH_OPS=1 $PY -m pip install --no-binary mmcv "mmcv==2.1.0" -v \
        || { echo "!! mmcv source build FAILED (check gcc/nvcc + CUDA_HOME)"; exit 1; }
fi

pipq "mmdet==3.3.0"

$PY - <<'PY'
import mmengine, mmcv, mmdet
from mmdet.apis import init_detector  # noqa
print(f"    mmengine {mmengine.__version__} | mmcv {mmcv.__version__} | mmdet {mmdet.__version__}  OK")
PY

# -----------------------------------------------------------------------------
# DPG scoring stack. modelscope's VQA pipeline eagerly imports OFA/MMSpeech,
# which pull in fairseq; fairseq does not build against recent torch/Python, so
# the patch script makes those non-essential imports optional.
# -----------------------------------------------------------------------------
echo ">>> [3/6] DPG scoring stack (modelscope + datasets)"
pipq "modelscope==1.38.1" "datasets==5.0.0"
$PY "${HERE}/patch_modelscope_no_fairseq.py"

echo ">>> [4/6] benchmark prompt assets"
if [ ! -f geneval/prompts/evaluation_metadata.jsonl ]; then
    echo "    cloning GenEval repo (prompts + object_names)"
    git clone --depth 1 https://github.com/djghosh13/geneval.git geneval
else
    echo "    geneval prompts present"
fi
if [ ! -f ELLA/dpg_bench/dpg_bench.csv ]; then
    echo "    cloning ELLA repo (DPG-Bench prompts + csv + scorer)"
    git clone --depth 1 https://github.com/TencentQQGYLab/ELLA.git ELLA
else
    echo "    ELLA dpg_bench assets present"
fi

if [ "${SKIP_DOWNLOADS:-0}" = "1" ]; then
    echo ">>> [5/6] SKIP_DOWNLOADS=1 -> skipping model weights"
else
echo ">>> [5/6] model weights"

# Mask2Former Swin-S COCO checkpoint for the GenEval object detector (~430 MB).
mkdir -p detector
M2F=detector/mask2former.pth
M2F_URL="https://download.openmmlab.com/mmdetection/v3.0/mask2former/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_20220504_001756-743b7d99.pth"
if [ -s "$M2F" ] && $PY -c "import torch;torch.load('$M2F',map_location='cpu')" 2>/dev/null; then
    echo "    Mask2Former ckpt present & valid"
else
    echo "    downloading Mask2Former ckpt"
    wget -q --show-progress -O "$M2F" "$M2F_URL" \
        || { echo "!! Mask2Former download failed"; exit 1; }
fi

# CLIP for the GenEval colour/attribute classifier (~1.7 GB, cached by HF).
echo "    caching CLIP openai/clip-vit-large-patch14"
$PY - <<'PY'
from transformers import CLIPModel, CLIPProcessor
CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
print("    CLIP cached OK")
PY

# mPLUG VQA model for DPG-Bench (~3 GB from ModelScope).
echo "    downloading mPLUG VQA model"
$PY - <<'PY'
from modelscope.hub.snapshot_download import snapshot_download
d = snapshot_download("damo/mplug_visual-question-answering_coco_large_en")
print("    mPLUG at", d)
PY
fi

echo ">>> [6/6] verify mPLUG pipeline imports and runs without fairseq"
$PY - <<'PY'
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from PIL import Image
import numpy as np
p = pipeline(Tasks.visual_question_answering,
             model="damo/mplug_visual-question-answering_coco_large_en", device="gpu")
img = Image.fromarray(np.uint8(np.random.rand(256, 256, 3) * 255))
print("    VQA smoke:", p({"image": img, "question": "Is there a cat?"}))
print("    mPLUG OK")
PY

echo "############################################################"
echo "# DONE. Eval env ready."
echo "#   GenEval   : bash scripts/eval_geneval.sh"
echo "#   DPG-Bench : bash scripts/eval_dpg.sh"
echo "############################################################"
