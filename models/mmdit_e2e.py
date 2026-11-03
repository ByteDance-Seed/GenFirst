# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

"""MMDiT (SD3/Flux-style dual-stream) adapted to the end-to-end training interface.

This mirrors the interface of ``models/sit.py::SiT`` so it can be dropped into
``train_e2e.py``'s end-to-end VAE+DiT loop, but replaces class-label adaLN
conditioning with **text conditioning**:

  * the caption token sequence (``context``) is the text stream in the joint
    (image+text) attention of every MMDiT block;
  * a pooled text vector is added to the timestep embedding to drive adaLN.

Like ``SiT`` it keeps:
  * a (affine-free) BatchNorm2d over the latent channels for online latent
    normalization (``init_bn`` / ``extract_latents_stats``);
  * REPA projectors on the *image* tokens at ``encoder_depth`` (``proj_loss``);
  * an internal v-prediction diffusion loss over a linear/cosine interpolant.

forward returns the same dict keys used by the training loop:
``denoising_loss``, ``proj_loss``, ``time_input``, ``noises`` (+ ``model_output``).
"""

import math

import numpy as np
import torch
import torch.nn as nn


def mean_flat(x):
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def modulate(x, shift, scale):
    if shift is None:
        shift = torch.zeros_like(scale)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )


def attention(q, k, v, heads):
    b, _, dim = q.shape
    dim_head = dim // heads
    q, k, v = map(lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2), (q, k, v))
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    return out.transpose(1, 2).reshape(b, -1, heads * dim_head)


# ----------------------------- embedders ------------------------------------
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(self.mlp[0].weight.dtype)
        return self.mlp(t_freq)


class VectorEmbedder(nn.Module):
    """Embeds a pooled text vector into the model hidden size."""

    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, x):
        return self.mlp(x)


# ----------------------------- blocks ---------------------------------------
class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, pre_only=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.pre_only = pre_only
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # A pre_only stream never runs post_attention, so it must NOT own a proj
        # (an unused parameter would break DDP without find_unused_parameters).
        if not pre_only:
            self.proj = nn.Linear(dim, dim)

    def pre_attention(self, x):
        qkv = self.qkv(x)
        B, L, _ = qkv.shape
        qkv = qkv.reshape(B, L, 3, -1).movedim(2, 0)
        return qkv[0], qkv[1], qkv[2]

    def post_attention(self, x):
        assert not self.pre_only
        return self.proj(x)


class DismantledBlock(nn.Module):
    """A single-stream adaLN DiT block, split into pre/post attention so two
    streams can share a joint attention (MMDiT)."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, qkv_bias=True, pre_only=False):
        super().__init__()
        self.pre_only = pre_only
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(hidden_size, num_heads, qkv_bias=qkv_bias, pre_only=pre_only)
        if not pre_only:
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            mlp_hidden = int(hidden_size * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, mlp_hidden),
                nn.GELU(approximate="tanh"),
                nn.Linear(mlp_hidden, hidden_size),
            )
        n_mods = 2 if pre_only else 6
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, n_mods * hidden_size, bias=True)
        )

    def pre_attention(self, x, c):
        if self.pre_only:
            shift_msa, scale_msa = self.adaLN_modulation(c).chunk(2, dim=1)
            qkv = self.attn.pre_attention(modulate(self.norm1(x), shift_msa, scale_msa))
            return qkv, None
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        qkv = self.attn.pre_attention(modulate(self.norm1(x), shift_msa, scale_msa))
        return qkv, (x, gate_msa, shift_mlp, scale_mlp, gate_mlp)

    def post_attention(self, attn, x, gate_msa, shift_mlp, scale_mlp, gate_mlp):
        x = x + gate_msa.unsqueeze(1) * self.attn.post_attention(attn)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class JointBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, qkv_bias=True, pre_only_context=False):
        super().__init__()
        self.context_block = DismantledBlock(hidden_size, num_heads, mlp_ratio, qkv_bias, pre_only=pre_only_context)
        self.x_block = DismantledBlock(hidden_size, num_heads, mlp_ratio, qkv_bias, pre_only=False)

    def forward(self, context, x, c):
        ctx_qkv, ctx_inter = self.context_block.pre_attention(context, c)
        x_qkv, x_inter = self.x_block.pre_attention(x, c)

        q = torch.cat((ctx_qkv[0], x_qkv[0]), dim=1)
        k = torch.cat((ctx_qkv[1], x_qkv[1]), dim=1)
        v = torch.cat((ctx_qkv[2], x_qkv[2]), dim=1)
        attn = attention(q, k, v, self.x_block.attn.num_heads)
        ctx_len = ctx_qkv[0].shape[1]
        ctx_attn, x_attn = attn[:, :ctx_len], attn[:, ctx_len:]

        if not self.context_block.pre_only:
            context = self.context_block.post_attention(ctx_attn, *ctx_inter)
        else:
            context = None
        x = self.x_block.post_attention(x_attn, *x_inter)
        return context, x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


# ----------------------------- pos embed ------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    emb_h = _1d_sincos(embed_dim // 2, grid[0])
    emb_w = _1d_sincos(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _1d_sincos(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0)
    omega = 1.0 / 10000**omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


# ----------------------------- model ----------------------------------------
class MMDiTE2E(nn.Module):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=16,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        context_dim=768,
        pooled_dim=768,
        caption_dropout_prob=0.1,
        z_dims=(768,),
        encoder_depth=8,
        projector_dim=2048,
        bn_momentum=0.1,
        path_type="linear",
        **block_kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.encoder_depth = encoder_depth
        self.caption_dropout_prob = caption_dropout_prob
        self.path_type = path_type
        self.z_dims = list(z_dims)

        grid = input_size // patch_size
        self.num_patches = grid * grid

        self.x_embedder = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = VectorEmbedder(pooled_dim, hidden_size)
        self.context_embedder = nn.Linear(context_dim, hidden_size)

        self.register_buffer("pos_embed", torch.zeros(1, self.num_patches, hidden_size))

        self.joint_blocks = nn.ModuleList([
            JointBlock(hidden_size, num_heads, mlp_ratio, qkv_bias=True, pre_only_context=(i == depth - 1))
            for i in range(depth)
        ])
        self.projectors = nn.ModuleList([
            build_mlp(hidden_size, projector_dim, z_dim) for z_dim in z_dims
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        # affine-free latent BN (same as SiT) for online latent normalization
        self.bn = torch.nn.BatchNorm2d(
            in_channels, eps=1e-4, momentum=bn_momentum, affine=False, track_running_stats=True
        )
        self.bn.reset_running_stats()

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic_init)

        pos = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))

        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.joint_blocks:
            nn.init.constant_(block.context_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.context_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.x_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.x_block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    # --- latent BN helpers (identical semantics to SiT) ---
    def init_bn(self, latents_scale, latents_bias):
        self.bn.running_mean = latents_bias
        self.bn.running_var = (1.0 / latents_scale).pow(2)

    def extract_latents_stats(self):
        return dict(
            latents_scale=self.bn.running_var.rsqrt(),
            latents_bias=self.bn.running_mean,
        )

    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(shape=(x.shape[0], c, h * p, w * p))

    def interpolant(self, t, path_type):
        if path_type == "linear":
            return 1 - t, t, -1, 1
        elif path_type == "cosine":
            return (
                torch.cos(t * np.pi / 2), torch.sin(t * np.pi / 2),
                -np.pi / 2 * torch.sin(t * np.pi / 2), np.pi / 2 * torch.cos(t * np.pi / 2),
            )
        raise NotImplementedError(path_type)

    def _embed_conditioning(self, time_input, context, pooled, train):
        """Return (c_mod, context_tokens). Applies caption dropout for CFG."""
        if train and self.caption_dropout_prob > 0:
            keep = (torch.rand(context.shape[0], device=context.device) >= self.caption_dropout_prob)
            keep = keep.to(context.dtype)
            context = context * keep.view(-1, 1, 1)
            pooled = pooled * keep.view(-1, 1)
        t_emb = self.t_embedder(time_input.flatten())
        c_mod = t_emb + self.y_embedder(pooled)
        context_tokens = self.context_embedder(context)
        return c_mod, context_tokens

    def _run_blocks(self, x, c_mod, context_tokens, align_only):
        B, L, D = x.shape
        zs_tilde = None
        for i, block in enumerate(self.joint_blocks):
            context_tokens, x = block(context_tokens, x, c_mod)
            if (i + 1) == self.encoder_depth:
                zs_tilde = [proj(x.reshape(-1, D)).reshape(B, L, -1) for proj in self.projectors]
                if align_only:
                    return x, zs_tilde
        x = self.final_layer(x, c_mod)
        return x, zs_tilde

    def forward(self, x, context, pooled, zs, loss_kwargs, time_input=None, noises=None):
        """x: (N,C,H,W) UNNORMALIZED latent; context: (N,L,context_dim);
        pooled: (N,pooled_dim); zs: list of (N,L',C') dinov2 features for REPA."""
        normalized_x = self.bn(x)

        if time_input is None:
            if loss_kwargs["weighting"] == "uniform":
                time_input = torch.rand((normalized_x.shape[0], 1, 1, 1))
            elif loss_kwargs["weighting"] == "lognormal":
                rnd = torch.randn((normalized_x.shape[0], 1, 1, 1))
                sigma = rnd.exp()
                if loss_kwargs["path_type"] == "linear":
                    time_input = sigma / (1 + sigma)
                else:
                    time_input = 2 / np.pi * torch.atan(sigma)
            else:
                raise NotImplementedError(loss_kwargs["weighting"])
        time_input = time_input.to(device=normalized_x.device, dtype=normalized_x.dtype)

        if noises is None:
            noises = torch.randn_like(normalized_x)
        else:
            noises = noises.to(device=normalized_x.device, dtype=normalized_x.dtype)

        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input, loss_kwargs["path_type"])
        model_input = alpha_t * normalized_x + sigma_t * noises
        if loss_kwargs["prediction"] == "v":
            model_target = d_alpha_t * normalized_x + d_sigma_t * noises
        else:
            raise NotImplementedError()

        x_tok = self.x_embedder(model_input).flatten(2).transpose(1, 2) + self.pos_embed
        c_mod, context_tokens = self._embed_conditioning(time_input, context, pooled, self.training)

        out, zs_tilde = self._run_blocks(x_tok, c_mod, context_tokens, loss_kwargs["align_only"])

        denoising_loss = None
        if not loss_kwargs["align_only"]:
            out = self.unpatchify(out)
            denoising_loss = mean_flat((out - model_target) ** 2)

        proj_loss = torch.tensor(0.0, device=x.device)
        bsz = zs[0].shape[0]
        for z, z_tilde in zip(zs, zs_tilde):
            for z_j, z_tilde_j in zip(z, z_tilde):
                z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1)
                z_j = torch.nn.functional.normalize(z_j, dim=-1)
                proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
        proj_loss /= (len(zs) * bsz)

        return {
            "zs_tilde": zs_tilde,
            "model_output": out,
            "denoising_loss": denoising_loss,
            "proj_loss": proj_loss,
            "time_input": time_input,
            "noises": noises,
        }

    @torch.no_grad()
    def inference(self, x, t, context, pooled):
        """Single denoising step forward (normalized-latent space). x is the
        already-normalized noisy latent; returns predicted v (same space)."""
        x_tok = self.x_embedder(x).flatten(2).transpose(1, 2) + self.pos_embed
        t_emb = self.t_embedder(t)
        c_mod = t_emb + self.y_embedder(pooled)
        context_tokens = self.context_embedder(context)
        for block in self.joint_blocks:
            context_tokens, x_tok = block(context_tokens, x_tok, c_mod)
        x_tok = self.final_layer(x_tok, c_mod)
        return self.unpatchify(x_tok)


# ------------------------------ configs -------------------------------------
def MMDiT_XL_2(**kwargs):
    return MMDiTE2E(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def MMDiT_L_2(**kwargs):
    return MMDiTE2E(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def MMDiT_B_2(**kwargs):
    return MMDiTE2E(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


MMDiT_models = {
    "MMDiT-XL/2": MMDiT_XL_2,
    "MMDiT-L/2": MMDiT_L_2,
    "MMDiT-B/2": MMDiT_B_2,
}
