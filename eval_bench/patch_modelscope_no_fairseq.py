# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

"""Patch modelscope so the mPLUG VQA pipeline imports WITHOUT fairseq.

DPG-Bench only uses the mPLUG VQA model, but modelscope's VQA pipeline and its
multi-modal preprocessor eagerly import the OFA / MMSpeech code paths, which pull
in `fairseq`. fairseq fails to build on torch>=2.5 / py3.11, so instead of
installing it we make those non-essential imports optional (try/except).

This script is idempotent: it looks for a marker string and skips files that are
already patched. Run it after `pip install modelscope`.

    python patch_modelscope_no_fairseq.py
"""

import os
import re


def modelscope_root():
    import modelscope
    return os.path.dirname(modelscope.__file__)


# (relative path, old snippet, new snippet, marker to detect prior patch)
PATCHES = [
    (
        "pipelines/multi_modal/visual_question_answering_pipeline.py",
        "from modelscope.models.multi_modal import MPlugForAllTasks, OfaForAllTasks",
        (
            "from modelscope.models.multi_modal import MPlugForAllTasks\n"
            "try:\n"
            "    from modelscope.models.multi_modal import OfaForAllTasks\n"
            "except Exception:  # Ofa needs fairseq (unavailable); DPG only uses MPlug\n"
            "    class OfaForAllTasks:  # type: ignore\n"
            "        pass"
        ),
        "Ofa needs fairseq",
    ),
    (
        "pipelines/multi_modal/visual_question_answering_pipeline.py",
        "from modelscope.preprocessors import (MPlugPreprocessor, OfaPreprocessor,\n"
        "                                      Preprocessor)",
        (
            "from modelscope.preprocessors import MPlugPreprocessor, Preprocessor\n"
            "try:\n"
            "    from modelscope.preprocessors import OfaPreprocessor\n"
            "except Exception:  # Ofa needs fairseq (unavailable); DPG only uses MPlug\n"
            "    class OfaPreprocessor:  # type: ignore\n"
            "        pass"
        ),
        "class OfaPreprocessor:  # type: ignore",
    ),
    (
        "preprocessors/ofa/__init__.py",
        "from .asr import OfaASRPreprocessor",
        (
            "try:\n"
            "    from .asr import OfaASRPreprocessor\n"
            "except Exception:  # asr needs fairseq (unavailable); not needed for VQA/DPG\n"
            "    OfaASRPreprocessor = None"
        ),
        "asr needs fairseq",
    ),
    (
        "models/multi_modal/ofa/__init__.py",
        "from .modeling_mmspeech import MMSpeechModel\n"
        "from .modeling_ofa import OFADecoder, OFAEncoder, OFAModel, OFAPreTrainedModel",
        (
            "try:\n"
            "    from .modeling_mmspeech import MMSpeechModel\n"
            "except Exception:  # mmspeech (fairseq) not needed for VQA/DPG\n"
            "    MMSpeechModel = None\n"
            "try:\n"
            "    from .modeling_ofa import OFADecoder, OFAEncoder, OFAModel, OFAPreTrainedModel\n"
            "except Exception:  # ofa model (fairseq) not needed for VQA/DPG\n"
            "    OFADecoder = OFAEncoder = OFAModel = OFAPreTrainedModel = None"
        ),
        "not needed for VQA/DPG",
    ),
]


def apply():
    root = modelscope_root()
    print(f">>> modelscope at {root}")
    for rel, old, new, marker in PATCHES:
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            print(f"  [skip] missing file: {rel}")
            continue
        with open(path, "r") as f:
            src = f.read()
        if marker in src:
            print(f"  [ok]   already patched: {rel}")
            continue
        if old not in src:
            # tolerate whitespace variation in the two-line preprocessor import
            norm = re.sub(r"\s+", " ", old)
            hit = re.search(re.escape(norm).replace(r"\ ", r"\s+"), src)
            if hit:
                src = src[:hit.start()] + new + src[hit.end():]
            else:
                print(f"  [WARN] snippet not found, manual check needed: {rel}")
                continue
        else:
            src = src.replace(old, new, 1)
        with open(path, "w") as f:
            f.write(src)
        print(f"  [patched] {rel}")


if __name__ == "__main__":
    apply()
    # sanity: importing the VQA pipeline module must not raise
    try:
        import importlib
        importlib.import_module(
            "modelscope.pipelines.multi_modal.visual_question_answering_pipeline")
        print(">>> import check OK (no fairseq needed)")
    except Exception as e:
        print(f">>> import check FAILED: {type(e).__name__}: {e}")
        raise
