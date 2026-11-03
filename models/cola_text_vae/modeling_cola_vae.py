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

"""
Cola Text VAE — HuggingFace Transformers compatible, NA (no-padding) inference.

This module implements the **Text VAE** of Cola DLM, which provides
*both* the inference encoder ``q_phi(z_0 | x)`` and the conditional
decoder ``p_theta(x | z_0)`` of the joint factorization

    p(x, z_0) = p_theta(x | z_0) * p_psi(z_0)

(Eq. 2.1.1 of the paper *Continuous Latent Diffusion Language Model*,
arXiv:2605.06548). The encoder is *not* part of the generative model —
it is used at training time for variational inference and at inference
time to encode the prefix into clean latent conditions
``z^pre ~ q_phi(z^pre | x^pre)`` (Eq. 2.2.4).

In the Stage-1 pretraining pipeline of the paper, this module is
trained with (Eq. 2.2.1)::

    L_VAE = - E_{q_phi(z_0|x)}[log p_theta(x | z_0)]
            + beta * KL(q_phi(z_0 | x) || p_base(z_0))
            + lambda_mask * L_mask,

where ``L_mask`` is a BERT-style masking loss that prevents the encoder
from collapsing semantically while the decoder merely memorizes surface
text. In Stage 2, the same encoder/decoder are jointly trained with the
DiT prior under a reference-encoder KL regularizer that suppresses
latent drift; see ``docs/architecture.md`` §9 for the explicit
objectives. Training code is not included in this open-source release.

The public API is a list-in / list-out variant:

* :meth:`ColaTextVAEModel.encode` accepts ``List[LongTensor(L_i,)]`` —
  one ``input_ids`` tensor per sample, each of length already divisible
  by ``patch_size * block_size`` (the per-sample pad is kept because
  ``nn.Conv1d`` cannot handle variable-length inputs; everything else
  happens in flattened NA form).
* :meth:`ColaTextVAEModel.decode` works in flattened NA form with
  per-sample ``txt_shape`` / ``txt_q_shape`` describing K and Q
  lengths, and realizes ``hat x^res ~ p_theta(x^res | z^pre, hat z_0^(1:B))``
  (Eq. 2.2.6) one block at a time.
"""

import math
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from transformers import AutoConfig, AutoModel, PreTrainedModel

from .attention_utils import create_na_block_causal_mask
from .configuration_cola_vae import ColaTextVAEConfig

# ---------------------------------------------------------------------------
# DiagonalGaussianDistribution
# ---------------------------------------------------------------------------


class DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        assert parameters.ndim in (2, 3)
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        sample = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        return self.mean + self.std * sample

    def mode(self) -> torch.Tensor:
        return self.mean


# ---------------------------------------------------------------------------
# Encoder / Decoder output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TextVAEEncoderOutput:
    # ``latents_list[i]`` has shape ``(n_i, latent_dim)``.
    latents_list: list[torch.Tensor]
    # ``latent_dists[i]`` is the posterior over sample ``i`` (None when
    # ``use_variation=False``).
    latent_dists: Optional[list[DiagonalGaussianDistribution]] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class BufferCache(dict, MutableMapping[str, torch.Tensor]):
    pass


def _build_na_positions(txt_shape: torch.LongTensor) -> torch.Tensor:
    """``txt_shape: (B, 1)`` → ``positions: (1, L_total)`` where each
    sample's positions restart from 0."""
    parts = [torch.arange(int(l), device=txt_shape.device) for l in txt_shape.flatten()]
    return torch.cat(parts).unsqueeze(0)


def _build_na_q_positions(txt_shape: torch.LongTensor, txt_q_shape: torch.LongTensor) -> torch.Tensor:
    """Per-sample Q positions aligned to the TAIL of K within each sample."""
    parts = []
    for k_len, q_len in zip(txt_shape.flatten().tolist(), txt_q_shape.flatten().tolist()):
        parts.append(torch.arange(k_len - q_len, k_len, device=txt_shape.device))
    return torch.cat(parts).unsqueeze(0)


# ---------------------------------------------------------------------------
# VAE Rotary Embedding (NA-capable)
# ---------------------------------------------------------------------------


class VAERotaryEmbedding(nn.Module):
    """Sin/cos RoPE with an explicit ``positions`` argument so the same
    layer can handle both padded-batch training and NA inference."""

    def __init__(self, rope_theta: int, head_dim: int, full_precision: bool = True, cache: BufferCache = None):
        super().__init__()
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.full_precision = full_precision
        self.__cache = cache if cache is not None else BufferCache()

    def get_rotary_embedding(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            (pos_sin := self.__cache.get("rope_pos_sin")) is not None
            and (pos_cos := self.__cache.get("rope_pos_cos")) is not None
            and pos_sin.shape[-2] >= seq_len
            and pos_cos.shape[-2] >= seq_len
        ):
            if pos_sin.device != device:
                pos_sin = pos_sin.to(device)
                self.__cache["rope_pos_sin"] = pos_sin
            if pos_cos.device != device:
                pos_cos = pos_cos.to(device)
                self.__cache["rope_pos_cos"] = pos_cos
            return pos_sin[:, :, :seq_len, :], pos_cos[:, :, :seq_len, :]

        with torch.autocast(device.type, enabled=False):
            dim = self.head_dim
            inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim))
            seq = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = torch.einsum("i , j -> i j", seq, inv_freq)
            positions = torch.cat((freqs, freqs), dim=-1)
            pos_sin = positions.sin()[None, None, :, :]
            pos_cos = positions.cos()[None, None, :, :]
        self.__cache["rope_pos_sin"] = pos_sin
        self.__cache["rope_pos_cos"] = pos_cos
        return pos_sin, pos_cos

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        B, nh, T, hs = x.size()
        x = x.view(B, nh, T, 2, hs // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, pos_sin, pos_cos, t):
        return ((t * pos_cos) + (self.rotate_half(t) * pos_sin)).to(t.dtype)

    def forward(
        self,
        q: torch.Tensor,  # (1, h, L_q, c)
        k: torch.Tensor,  # (1, h, L_k, c)
        q_positions: torch.Tensor,  # (1, L_q) absolute positions for Q
        k_positions: torch.Tensor,  # (1, L_k) absolute positions for K
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_ = q.float() if self.full_precision else q
        k_ = k.float() if self.full_precision else k

        with torch.autocast(q.device.type, enabled=False):
            max_pos = int(max(int(q_positions.max().item()), int(k_positions.max().item()))) + 1
            all_pos_sin, all_pos_cos = self.get_rotary_embedding(max_pos, q_.device)
            all_pos_sin = all_pos_sin.type_as(q_)
            all_pos_cos = all_pos_cos.type_as(q_)

            lookup_sin = all_pos_sin.squeeze(0).squeeze(0)  # (max_pos, c)
            lookup_cos = all_pos_cos.squeeze(0).squeeze(0)

            pos_sin_q = lookup_sin[q_positions.squeeze(0)].unsqueeze(0).unsqueeze(0)
            pos_cos_q = lookup_cos[q_positions.squeeze(0)].unsqueeze(0).unsqueeze(0)
            pos_sin_k = lookup_sin[k_positions.squeeze(0)].unsqueeze(0).unsqueeze(0)
            pos_cos_k = lookup_cos[k_positions.squeeze(0)].unsqueeze(0).unsqueeze(0)

            q_ = self.apply_rotary_pos_emb(pos_sin_q, pos_cos_q, q_)
            k_ = self.apply_rotary_pos_emb(pos_sin_k, pos_cos_k, k_)

        return q_.type_as(q), k_.type_as(k)


# ---------------------------------------------------------------------------
# SwiGLU / norm builder
# ---------------------------------------------------------------------------


class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

    @property
    def output_multiplier(self) -> float:
        return 0.5


def build_norm_layer(layer_norm_type: str, dim: int, eps: float = 1e-5, elementwise_affine: bool = True):
    if layer_norm_type == "layer_norm":
        return nn.LayerNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
    raise NotImplementedError(f"Unsupported layer_norm_type: {layer_norm_type}")


def init_normal(module, std: float, init_cutoff_factor: Optional[float] = None):
    if init_cutoff_factor is not None:
        cutoff_value = init_cutoff_factor * std
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-cutoff_value, b=cutoff_value)
    else:
        nn.init.normal_(module.weight, mean=0.0, std=std)
    if isinstance(module, nn.Linear) and module.bias is not None:
        nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# TextVAEBlock — transformer block for encoder/decoder (NA form)
# ---------------------------------------------------------------------------


class TextVAEBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        shared_heads_kv: int = 1,
        qk_bias: bool = False,
        clip_qkv: Optional[float] = None,
        qk_norm: bool = False,
        qk_norm_affine: bool = True,
        post_norm: bool = False,
        layer_norm_type: str = "layer_norm",
        layer_norm_eps: float = 1e-6,
        layer_norm_affine: bool = True,
        rope_theta: int = 500000,
        rope_full_precision: bool = True,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        bias: bool = False,
        act: str = "swiglu",
        causal: bool = False,
        init_fn: str = "normal",
        init_std: float = 0.02,
        init_cutoff_factor: Optional[float] = None,
    ):
        super().__init__()
        assert dim % num_heads == 0
        assert num_heads % shared_heads_kv == 0
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.clip_qkv = clip_qkv
        self.post_norm = post_norm
        self.attn_dropout = attn_dropout
        self.causal = causal
        self.init_fn = init_fn
        self.init_std = init_std
        self.init_cutoff_factor = init_cutoff_factor

        self.__cache = BufferCache()

        self.norm_attn = build_norm_layer(layer_norm_type, dim, layer_norm_eps, layer_norm_affine)
        self.fused_dims = (dim, dim // shared_heads_kv, dim // shared_heads_kv)
        self.qkv_proj = nn.Linear(dim, sum(self.fused_dims), bias=qk_bias)
        self.scale = 1 / math.sqrt(self.head_dim)

        self.q_norm = None
        self.k_norm = None
        if qk_norm:
            self.q_norm = build_norm_layer(layer_norm_type, dim, layer_norm_eps, qk_norm_affine)
            self.k_norm = build_norm_layer(layer_norm_type, dim // shared_heads_kv, layer_norm_eps, qk_norm_affine)

        self.rope = VAERotaryEmbedding(
            rope_theta=rope_theta,
            head_dim=self.head_dim,
            full_precision=rope_full_precision,
            cache=self.__cache,
        )
        self.attn_out_proj = nn.Linear(dim, dim, bias=bias)
        self.dropout_layer = nn.Dropout(dropout)

        self.norm_ffn = build_norm_layer(layer_norm_type, dim, layer_norm_eps)
        self.ffn_proj = nn.Linear(dim, ffn_dim, bias=bias)
        self.ffn_act = self._build_act(act)
        self.ffn_out = nn.Linear(
            int(getattr(self.ffn_act, "output_multiplier", 1.0) * ffn_dim),
            dim,
            bias=bias,
        )

        # Per-sample KV cache (one ``(l_i_cum, d)`` tensor per sample). None
        # means "fresh cache".
        self._k_cache: Optional[list[torch.Tensor]] = None
        self._v_cache: Optional[list[torch.Tensor]] = None
        self.reset_parameters()

    def _build_act(self, act):
        if act == "silu":
            return nn.SiLU()
        if act == "gelu":
            return nn.GELU()
        if act == "swiglu":
            return SwiGLU()
        raise NotImplementedError(f"Unsupported activation: {act}")

    def reset_parameters(self):
        def _init_layer_norm(m):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)

        self.norm_attn.apply(_init_layer_norm)
        self.norm_ffn.apply(_init_layer_norm)

        if self.init_fn == "normal":
            std = self.init_std
            cutoff_factor = self.init_cutoff_factor
        elif self.init_fn == "mitchell":
            std = 1.0 / math.sqrt(self.dim)
            cutoff_factor = self.init_cutoff_factor or 3.0
        elif self.init_fn == "full_megatron":
            std = self.init_std
            cutoff_factor = self.init_cutoff_factor or 3.0
        else:
            raise NotImplementedError(self.init_fn)

        init_normal(self.qkv_proj, std, cutoff_factor)
        init_normal(self.attn_out_proj, std, cutoff_factor)
        init_normal(self.ffn_proj, std, cutoff_factor)
        init_normal(self.ffn_out, std, cutoff_factor)

    def set_kv_cache(self, flag: bool) -> None:
        self._k_cache = None
        self._v_cache = None

    def slow_attn(self, q, k, v, attn_mask=None, dropout_p=0.0):
        attn = q.mul(self.scale) @ k.transpose(-2, -1)
        with torch.autocast(device_type="cuda" if q.is_cuda else q.device.type, dtype=torch.bfloat16):
            if attn_mask is not None:
                attn.add_(attn_mask.to(attn.dtype))
        attn_weight = attn.softmax(dim=-1)
        if dropout_p > 0:
            attn_weight = F.dropout(attn_weight, p=dropout_p, inplace=True)
        return attn_weight @ v

    def forward(
        self,
        x: torch.Tensor,  # (1, L_q_total, d)
        txt_shape: torch.LongTensor,  # (B, 1) K length per sample
        txt_q_shape: torch.LongTensor,  # (B, 1) Q length per sample
        attn_block_mask: torch.Tensor,  # (1, 1, L_q, L_k)
        update_kv: bool = False,
    ) -> torch.Tensor:
        if self.post_norm:
            h = x
        else:
            h = self.norm_attn(x)

        qkv = self.qkv_proj(h)
        if self.clip_qkv is not None:
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
        q, k, v = qkv.split(self.fused_dims, dim=-1)

        # -- KV cache bookkeeping (per-sample) -----------------------------------
        q_lens = txt_q_shape.flatten().tolist()
        new_ks = list(k.squeeze(0).split(q_lens, dim=0))
        new_vs = list(v.squeeze(0).split(q_lens, dim=0))

        if update_kv:
            if self._k_cache is None:
                self._k_cache = [tk.clone() for tk in new_ks]
                self._v_cache = [tv.clone() for tv in new_vs]
            else:
                self._k_cache = [torch.cat([c, n], dim=0) for c, n in zip(self._k_cache, new_ks)]
                self._v_cache = [torch.cat([c, n], dim=0) for c, n in zip(self._v_cache, new_vs)]
            full_k = torch.cat(self._k_cache, dim=0).unsqueeze(0)
            full_v = torch.cat(self._v_cache, dim=0).unsqueeze(0)
        elif self._k_cache is not None:
            full_k = torch.cat([torch.cat([c, n], dim=0) for c, n in zip(self._k_cache, new_ks)], dim=0).unsqueeze(0)
            full_v = torch.cat([torch.cat([c, n], dim=0) for c, n in zip(self._v_cache, new_vs)], dim=0).unsqueeze(0)
        else:
            full_k = k
            full_v = v

        # -- qk_norm (optional) --------------------------------------------------
        dtype = q.dtype
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q).to(dtype=dtype)
            full_k = self.k_norm(full_k).to(dtype=dtype)

        q = rearrange(q, "b l (h c) -> b h l c", c=self.head_dim)
        full_k = rearrange(full_k, "b l (h c) -> b h l c", c=self.head_dim)
        full_v = rearrange(full_v, "b l (h c) -> b h l c", c=self.head_dim)

        # -- RoPE (NA positions) -------------------------------------------------
        q_positions = _build_na_q_positions(txt_shape, txt_q_shape)
        k_positions = _build_na_positions(txt_shape)
        q, full_k = self.rope(q, full_k, q_positions=q_positions, k_positions=k_positions)

        # -- Attention -----------------------------------------------------------
        o = self.slow_attn(
            q, full_k, full_v, attn_mask=attn_block_mask, dropout_p=self.attn_dropout if self.training else 0.0
        )
        o = rearrange(o, "b h l c -> b l (h c)")
        attn = self.attn_out_proj(o)

        if self.post_norm:
            x = self.norm_attn(x)
        x = x + self.dropout_layer(attn)

        residual = x
        if not self.post_norm:
            x = self.norm_ffn(x)
        x = self.ffn_proj(x)
        x = self.ffn_act(x)
        x = self.ffn_out(x)
        if self.post_norm:
            x = self.norm_ffn(x)
        x = self.dropout_layer(x)
        x = residual + x
        return x


# ---------------------------------------------------------------------------
# Main Model: ColaTextVAEModel
# ---------------------------------------------------------------------------


class ColaTextVAEModel(PreTrainedModel):
    config_class = ColaTextVAEConfig
    base_model_prefix = "text_vae"

    def __init__(self, config: ColaTextVAEConfig):
        super().__init__(config)
        self.patch_size = config.patch_size
        self.vocab_size = config.vocab_size
        self.latent_dim = config.latent_dim
        self.use_variation = config.use_variation
        self.encoder_last_ln = config.encoder_last_ln
        self.use_emb = config.use_emb
        self.block_size = config.block_size
        self.block_causal = config.block_causal
        self.scaling_factor = config.scaling_factor
        self.shifting_factor = config.shifting_factor

        block_kwargs = dict(
            dim=config.dim,
            ffn_dim=config.ffn_dim,
            num_heads=config.num_heads,
            shared_heads_kv=config.shared_heads_kv,
            qk_bias=config.qk_bias,
            clip_qkv=config.clip_qkv,
            qk_norm=config.qk_norm,
            qk_norm_affine=config.qk_norm_affine,
            post_norm=config.post_norm,
            layer_norm_type=config.layer_norm_type,
            layer_norm_eps=config.layer_norm_eps,
            layer_norm_affine=config.layer_norm_affine,
            rope_theta=config.rope_theta,
            rope_full_precision=config.rope_full_precision,
            dropout=config.dropout,
            attn_dropout=config.attn_dropout,
            bias=config.bias,
            act=config.act,
            causal=config.causal,
            init_fn=config.init_fn,
            init_std=config.init_std,
            init_cutoff_factor=config.init_cutoff_factor,
        )

        encoder_dict = dict(
            wte=nn.Embedding(config.vocab_size + 1, config.dim),
            patch_embedder=nn.Conv1d(config.dim, config.dim, kernel_size=config.patch_size, stride=config.patch_size),
            blocks=nn.ModuleList([TextVAEBlock(**block_kwargs) for _ in range(config.encoder_num_blocks)]),
        )

        if config.use_variation:
            if config.encoder_last_ln:
                final_norm = build_norm_layer(config.layer_norm_type, config.latent_dim, config.layer_norm_eps, False)
            else:
                final_norm = nn.Identity()
            encoder_dict["final_layer"] = nn.Linear(config.dim, config.latent_dim * 2, bias=config.bias)
            encoder_dict["final_norm"] = final_norm
        else:
            encoder_dict["final_layer"] = nn.Linear(config.dim, config.latent_dim, bias=config.bias)
            encoder_dict["final_norm"] = build_norm_layer(
                config.layer_norm_type,
                config.latent_dim,
                config.layer_norm_eps,
                False,
            )

        self.encoder = nn.ModuleDict(encoder_dict)

        self.decoder = nn.ModuleDict(
            dict(
                in_layer=nn.Linear(config.latent_dim, config.dim, bias=config.bias),
                blocks=nn.ModuleList([TextVAEBlock(**block_kwargs) for _ in range(config.decoder_num_blocks)]),
                unpatch_layer=nn.Linear(config.dim, config.patch_size * config.dim),
                final_norm=build_norm_layer(config.layer_norm_type, config.dim, config.layer_norm_eps),
                final_layer=nn.Linear(config.dim, config.vocab_size, bias=config.bias),
            )
        )

        self.post_init()

    def _init_weights(self, module):
        std = self.config.init_std
        cutoff = self.config.init_cutoff_factor
        if isinstance(module, (nn.Linear, nn.Embedding)):
            init_normal(module, std, cutoff)
        if isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # ---------------------------------------------------------------
    # NA encode
    # ---------------------------------------------------------------

    def _encode_patch_per_sample(self, input_ids_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """Run embedding + patch ``Conv1d`` one sample at a time.

        ``nn.Conv1d`` requires a fixed length within a batch; looping
        over the (small) Python list is cheap compared to the
        downstream transformer blocks and lets the subsequent path stay
        pure NA.
        """
        out_list: list[torch.Tensor] = []
        for ids in input_ids_list:
            x = self.encoder.wte(ids.unsqueeze(0))  # (1, L_i, d)
            x = x.permute(0, 2, 1)  # (1, d, L_i)
            x = self.encoder.patch_embedder(x)  # (1, d, n_i)
            x = x.permute(0, 2, 1).squeeze(0)  # (n_i, d)
            out_list.append(x)
        return out_list

    def encode(self, input_ids_list: list[torch.Tensor]) -> TextVAEEncoderOutput:
        """Encode per-sample ``input_ids`` into per-sample latents.

        Realizes the inference encoder ``q_phi(z_0 | x)`` of Eq. 2.1.1
        of the paper. At inference time the same call also produces the
        prefix latent ``z^pre ~ q_phi(z^pre | x^pre)`` of Eq. 2.2.4
        (set ``deterministic=True`` on the returned distribution to use
        the posterior mode as in :func:`generate_task_repaint_inference`).

        Each ``input_ids_list[i]`` must have length divisible by
        ``patch_size * block_size`` (the per-sample block alignment pad
        still applies because of the patch ``Conv1d``).
        """
        per_sample = self._encode_patch_per_sample(input_ids_list)  # List[(n_i, d)]
        txt_shape = torch.tensor(
            [[x.shape[0]] for x in per_sample],
            dtype=torch.long,
            device=per_sample[0].device,
        )
        x = torch.cat(per_sample, dim=0).unsqueeze(0)  # (1, L_total, d)

        attn_mask = None
        if self.block_causal:
            attn_mask = create_na_block_causal_mask(
                txt_shape=txt_shape,
                txt_q_shape=txt_shape,
                block_size=self.block_size,
                dtype=torch.bfloat16 if x.is_cuda else x.dtype,
                device=x.device,
            )

        for block in self.encoder.blocks:
            x = block(
                x,
                txt_shape=txt_shape,
                txt_q_shape=txt_shape,
                attn_block_mask=attn_mask,
                update_kv=False,
            )

        x = self.encoder.final_layer(x)
        if self.encoder_last_ln and self.use_variation:
            mean, logvar = torch.chunk(x, 2, dim=-1)
            mean = self.encoder.final_norm(mean)
            x = torch.cat((mean, logvar), dim=-1)
        else:
            x = self.encoder.final_norm(x)

        # Split back to per-sample latents.
        latents_flat = x.squeeze(0)  # (L_total, latent_dim_out)
        split_sizes = txt_shape.flatten().tolist()
        per_sample_latents = list(latents_flat.split(split_sizes, dim=0))

        latent_dists: Optional[list[DiagonalGaussianDistribution]] = None
        if self.use_variation:
            latent_dists = [DiagonalGaussianDistribution(p) for p in per_sample_latents]
            latents_mode = [d.mode() for d in latent_dists]  # mean only
        else:
            latents_mode = per_sample_latents

        return TextVAEEncoderOutput(latents_list=latents_mode, latent_dists=latent_dists)

    # ---------------------------------------------------------------
    # NA decode
    # ---------------------------------------------------------------

    def decode(
        self,
        z: torch.Tensor,  # (L_q_total, latent_dim)
        txt_shape: torch.LongTensor,  # (B, 1) K length per sample
        txt_q_shape: torch.LongTensor,  # (B, 1) Q length per sample
        update_kv: bool = False,
    ) -> torch.Tensor:
        """Decode flattened latents into vocabulary logits.

        Realizes the conditional decoder ``p_theta(x | z_0)`` of
        Eq. 2.1.1. During block-wise inference, repeated calls realize
        ``hat x^res ~ p_theta(x^res | z^pre, hat z_0^(1:B))`` (Eq. 2.2.6)
        one block at a time, with the per-sample prefix latent
        ``z^pre`` and previously committed blocks ``hat z_0^(<b)``
        carried implicitly through the per-layer KV cache.

        Returns ``(1, L_q_total * patch_size, vocab)`` so the caller
        can reshape / split per sample.
        """
        z = self.decoder.in_layer(z)
        z = z.unsqueeze(0)  # (1, L_q_total, d)

        attn_mask = None
        if self.block_causal:
            attn_mask = create_na_block_causal_mask(
                txt_shape=txt_shape,
                txt_q_shape=txt_q_shape,
                block_size=self.block_size,
                dtype=torch.bfloat16 if z.is_cuda else z.dtype,
                device=z.device,
            )

        for block in self.decoder.blocks:
            z = block(
                z,
                txt_shape=txt_shape,
                txt_q_shape=txt_q_shape,
                attn_block_mask=attn_mask,
                update_kv=update_kv,
            )

        z = self.decoder.unpatch_layer(z)  # (1, L_q, d*patch_size)
        z = rearrange(z, "b l (c ps) -> b (l ps) c", ps=self.patch_size)
        z = self.decoder.final_norm(z)
        z = self.decoder.final_layer(z)
        return z

    # ---------------------------------------------------------------
    # KV cache control
    # ---------------------------------------------------------------

    def set_kv_cache(self, flag: bool) -> None:
        for block in self.encoder.blocks:
            block.set_kv_cache(flag)
        for block in self.decoder.blocks:
            block.set_kv_cache(flag)

    # ---------------------------------------------------------------
    # HF-style forward — NA mode does not use this for the usual
    # encode/decode flow, but PreTrainedModel requires a ``forward``.
    # ---------------------------------------------------------------

    def forward(self, input_ids_list: list[torch.Tensor]):
        enc = self.encode(input_ids_list)
        latents_cat = torch.cat(enc.latents_list, dim=0)
        txt_shape = torch.tensor(
            [[l.shape[0]] for l in enc.latents_list],
            dtype=torch.long,
            device=latents_cat.device,
        )
        preds = self.decode(latents_cat, txt_shape=txt_shape, txt_q_shape=txt_shape)
        return enc, preds


# Tolerate the reference `cola_dlm` package having registered the same
# model_type earlier in the process.
try:
    AutoConfig.register("cola_text_vae", ColaTextVAEConfig)
    AutoModel.register(ColaTextVAEConfig, ColaTextVAEModel)
except ValueError:
    pass
