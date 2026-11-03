# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers import PretrainedConfig


class ColaTextVAEConfig(PretrainedConfig):
    """Configuration for :class:`ColaTextVAEModel`.

    Parameterizes the **Text VAE** of Cola DLM, which provides *both*
    the inference encoder ``q_phi(z_0 | x)`` and the conditional
    decoder ``p_theta(x | z_0)`` of the joint factorization

        p(x, z_0) = p_theta(x | z_0) * p_psi(z_0)

    (Eq. 2.1.1 of the paper *Continuous Latent Diffusion Language
    Model*, arXiv:2605.06548). Stage-1 pretraining fits this module
    with the objective ``L_VAE`` of Eq. 2.2.1; in Stage 2 the same
    encoder/decoder are jointly trained with the DiT prior.

    Key knobs (in paper notation):

    * ``vocab_size``: tokenizer vocabulary (OLMo 2 BPE in the released
      checkpoint).
    * ``encoder_num_blocks`` / ``decoder_num_blocks`` / ``dim`` /
      ``ffn_dim`` / ``num_heads``: shape of the transformer trunk
      backing ``q_phi`` and ``p_theta``.
    * ``latent_dim``: dimension ``d`` of the continuous latent
      ``z_0 ∈ R^d`` (Eq. 2.1.1). Must agree with
      :class:`ColaDiTConfig.txt_in_channels` / ``txt_out_channels``.
    * ``patch_size``: 1-D patchification factor between the token axis
      and the latent axis (``n_i = L_i / patch_size``).
    * ``block_causal`` / ``block_size``: per-sample factor that defines
      how the latent sequence is partitioned into ``B`` blocks
      ``z_0 = (z_0^(1), ..., z_0^(B))`` (Eq. 2.1.4). The same
      ``block_size`` is used by the DiT prior to enforce the visible
      set ``V_b`` (Eq. 2.2.3); both VAE encoder and decoder respect
      this factorization to prevent information leakage and to keep
      streaming generation well-defined.
    * ``rope_theta`` / ``rope_full_precision``: RoPE positional
      encoding configuration.
    * ``use_variation``: whether to materialize ``q_phi`` as a Gaussian
      posterior (mean + log-variance) versus a deterministic encoder.
    * ``scaling_factor`` / ``shifting_factor``: per-channel
      normalization applied to VAE latents at the boundary with the
      DiT prior, ``z_0 ← (z_0 - shifting_factor) * scaling_factor``.
      They control the geometry of the latent space over which
      ``p_psi`` is learned.

    Defaults match the released ~500M-parameter Text VAE from the
    paper's RQ4 scaling curve (4 encoder + 4 decoder blocks,
    ``dim=1536``, ``ffn_dim=6144``, ``latent_dim=16``).
    """

    model_type = "cola_text_vae"

    def __init__(
        self,
        vocab_size: int = 100278,
        encoder_num_blocks: int = 4,
        decoder_num_blocks: int = 4,
        dim: int = 1536,
        ffn_dim: int = 6144,
        latent_dim: int = 16,
        patch_size: int = 1,
        num_heads: int = 12,
        shared_heads_kv: int = 1,
        encoder_last_ln: bool = True,
        layer_norm_type: str = "layer_norm",
        layer_norm_eps: float = 1e-6,
        layer_norm_affine: bool = True,
        post_norm: bool = True,
        qk_bias: bool = False,
        qk_norm: bool = True,
        qk_norm_affine: bool = True,
        clip_qkv: float = None,
        rope_theta: int = 500000,
        rope_full_precision: bool = True,
        bias: bool = True,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        act: str = "swiglu",
        causal: bool = False,
        block_causal: bool = True,
        block_size: int = 4,
        init_fn: str = "normal",
        init_std: float = 0.02,
        init_cutoff_factor: float = 3,
        use_variation: bool = True,
        use_emb: bool = True,
        scaling_factor: float = 1.0,
        shifting_factor: float = 0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.encoder_num_blocks = encoder_num_blocks
        self.decoder_num_blocks = decoder_num_blocks
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.shared_heads_kv = shared_heads_kv
        self.encoder_last_ln = encoder_last_ln
        self.layer_norm_type = layer_norm_type
        self.layer_norm_eps = layer_norm_eps
        self.layer_norm_affine = layer_norm_affine
        self.post_norm = post_norm
        self.qk_bias = qk_bias
        self.qk_norm = qk_norm
        self.qk_norm_affine = qk_norm_affine
        self.clip_qkv = clip_qkv
        self.rope_theta = rope_theta
        self.rope_full_precision = rope_full_precision
        self.bias = bias
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.act = act
        self.causal = causal
        self.block_causal = block_causal
        self.block_size = block_size
        self.init_fn = init_fn
        self.init_std = init_std
        self.init_cutoff_factor = init_cutoff_factor
        self.use_variation = use_variation
        self.use_emb = use_emb
        self.scaling_factor = scaling_factor
        self.shifting_factor = shifting_factor
        super().__init__(**kwargs)
