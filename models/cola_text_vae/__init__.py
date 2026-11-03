# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Vendored from Cola DLM, Apache-2.0. See NOTICE and the sibling modules,
# which carry the full Apache header.

"""Vendored text-VAE modules from Cola DLM (Apache-2.0).

Copied verbatim from the ``cola_dlm`` package of
https://github.com/ByteDance-Seed/Cola-DLM so that training does not depend on
that checkout being present, which matters for multi-node launches. Only the VAE
path is vendored; the Cola DiT prior is not used. Pretrained weights come from
https://huggingface.co/ByteDance-Seed/Cola-DLM -- see ``docs/JOINT.md``.
"""

from .attention_utils import (
    create_2l_block_causal_mask,
    create_na_block_causal_mask,
    cu_seqlens,
    get_seqlen,
    max_seqlen,
)
from .configuration_cola_vae import ColaTextVAEConfig
from .modeling_cola_vae import (
    ColaTextVAEModel,
    DiagonalGaussianDistribution,
    TextVAEEncoderOutput,
)

__all__ = [
    "ColaTextVAEConfig",
    "ColaTextVAEModel",
    "DiagonalGaussianDistribution",
    "TextVAEEncoderOutput",
    "create_2l_block_causal_mask",
    "create_na_block_causal_mask",
    "cu_seqlens",
    "get_seqlen",
    "max_seqlen",
]
