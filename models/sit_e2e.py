# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

# SiT for end-to-end VAE training on aspect-ratio-bucketed latents.
#
# Same math and parameter names as models/sit.py (so checkpoints are
# interchangeable), except:
#   * the 2D sin-cos position embedding is built per (h, w) token grid instead of
#     a fixed square buffer, so any latent shape divisible by patch_size works;
#   * unpatchify handles rectangular grids;
#   * attention / MLP / patch embed are local (no timm version dependence), with
#     optional activation checkpointing over the blocks.
# Unconditional training uses num_classes=1 with class_dropout_prob=0: the label
# embedding then degenerates to a single learned bias.

import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def mean_flat(x):
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, qk_norm=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


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
    def positional_embedding(t, dim, max_period=10000):
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
        t_freq = self.positional_embedding(t, dim=self.frequency_embedding_size).to(t.dtype)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def forward(self, labels, train, force_drop_ids=None):
        if (train and self.dropout_prob > 0) or (force_drop_ids is not None):
            if force_drop_ids is None:
                drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            else:
                drop_ids = force_drop_ids == 1
            labels = torch.where(drop_ids, self.num_classes, labels)
        return self.embedding_table(labels)


class SiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, qk_norm=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qk_norm=qk_norm)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(hidden_size, int(hidden_size * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


@lru_cache(maxsize=64)
def _sincos_2d(embed_dim, h, w):
    grid_h = np.arange(h, dtype=np.float32)
    grid_w = np.arange(w, dtype=np.float32)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0).reshape(2, 1, h, w)
    emb_h = _sincos_1d(embed_dim // 2, grid[0])
    emb_w = _sincos_1d(embed_dim // 2, grid[1])
    return torch.from_numpy(np.concatenate([emb_h, emb_w], axis=1)).float()


def _sincos_1d(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0)
    omega = 1.0 / 10000 ** omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


class SiT(nn.Module):
    def __init__(
        self,
        path_type="linear",
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        decoder_hidden_size=768,
        encoder_depth=8,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.0,
        num_classes=1,
        z_dims=[768],
        projector_dim=2048,
        bn_momentum=0.1,
        qk_norm=False,
        grad_ckpt=False,
        **unused,
    ):
        super().__init__()
        self.path_type = path_type
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.encoder_depth = encoder_depth
        self.grad_ckpt = grad_ckpt

        self.x_embedder = PatchEmbed(patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, qk_norm=qk_norm) for _ in range(depth)
        ])
        self.projectors = nn.ModuleList([build_mlp(hidden_size, projector_dim, z_dim) for z_dim in z_dims])
        self.final_layer = FinalLayer(decoder_hidden_size, patch_size, self.out_channels)
        # No affine params: an affine BN would let the diffusion loss rescale the latent for free.
        self.bn = nn.BatchNorm2d(in_channels, eps=1e-4, momentum=bn_momentum, affine=False, track_running_stats=True)
        self.bn.reset_running_stats()
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def init_bn(self, latents_scale, latents_bias):
        # latents_scale = 1 / std, latents_bias = mean
        self.bn.running_mean.copy_(latents_bias)
        self.bn.running_var.copy_((1.0 / latents_scale).pow(2))

    def extract_latents_stats(self):
        return dict(latents_scale=self.bn.running_var.rsqrt(), latents_bias=self.bn.running_mean)

    def _embed(self, x):
        p = self.patch_size
        h, w = x.shape[-2] // p, x.shape[-1] // p
        tokens = self.x_embedder(x)
        pos = _sincos_2d(self.hidden_size, h, w).to(device=tokens.device, dtype=tokens.dtype)
        return tokens + pos.unsqueeze(0), h, w

    def unpatchify(self, x, h, w):
        c, p = self.out_channels, self.patch_size
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def _run_block(self, block, x, c):
        if self.grad_ckpt and self.training and torch.is_grad_enabled():
            return checkpoint(block, x, c, use_reentrant=False)
        return block(x, c)

    @staticmethod
    def interpolant(t, path_type):
        if path_type == "linear":
            return 1 - t, t, -1, 1
        if path_type == "cosine":
            return (torch.cos(t * np.pi / 2), torch.sin(t * np.pi / 2),
                    -np.pi / 2 * torch.sin(t * np.pi / 2), np.pi / 2 * torch.cos(t * np.pi / 2))
        raise NotImplementedError(path_type)

    def forward(self, x, y, zs, loss_kwargs, time_input=None, noises=None):
        """x: unnormalized latent [N, C, H, W]; zs: list of [N, T, C'] REPA targets
        on the same (H/p) x (W/p) token grid."""
        normalized_x = self.bn(x)
        n = normalized_x.shape[0]
        if time_input is None:
            if loss_kwargs["weighting"] == "uniform":
                time_input = torch.rand((n, 1, 1, 1))
            elif loss_kwargs["weighting"] == "lognormal":
                sigma = torch.randn((n, 1, 1, 1)).exp()
                time_input = sigma / (1 + sigma) if loss_kwargs["path_type"] == "linear" else 2 / np.pi * torch.atan(sigma)
            else:
                raise NotImplementedError(loss_kwargs["weighting"])
        time_input = time_input.to(device=normalized_x.device, dtype=normalized_x.dtype)
        if noises is None:
            noises = torch.randn_like(normalized_x)
        else:
            noises = noises.to(device=normalized_x.device, dtype=normalized_x.dtype)

        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input, loss_kwargs["path_type"])
        model_input = alpha_t * normalized_x + sigma_t * noises
        assert loss_kwargs["prediction"] == "v"
        model_target = d_alpha_t * normalized_x + d_sigma_t * noises

        h_tok, h, w = self._embed(model_input)
        N, T, D = h_tok.shape
        c = self.t_embedder(time_input.flatten()) + self.y_embedder(y, self.training)

        out = h_tok
        zs_tilde = None
        for i, block in enumerate(self.blocks):
            out = self._run_block(block, out, c)
            if (i + 1) == self.encoder_depth:
                zs_tilde = [proj(out.reshape(-1, D)).reshape(N, T, -1) for proj in self.projectors]
                if loss_kwargs["align_only"]:
                    break

        denoising_loss = None
        model_output = None
        if not loss_kwargs["align_only"]:
            model_output = self.unpatchify(self.final_layer(out, c), h, w)
            denoising_loss = mean_flat((model_output - model_target) ** 2)

        proj_loss = torch.tensor(0.0, device=out.device)
        for z, z_tilde in zip(zs, zs_tilde):
            z_tilde = F.normalize(z_tilde, dim=-1)
            z = F.normalize(z, dim=-1)
            proj_loss = proj_loss + (-(z * z_tilde).sum(dim=-1)).mean(dim=-1).sum()
        proj_loss = proj_loss / (len(zs) * N)

        return {
            "zs_tilde": zs_tilde,
            "model_output": model_output,
            "denoising_loss": denoising_loss,
            "proj_loss": proj_loss,
            "time_input": time_input,
            "noises": noises,
        }

    @torch.no_grad()
    def inference(self, x, t, y):
        tok, h, w = self._embed(x)
        c = self.t_embedder(t) + self.y_embedder(y, False)
        for block in self.blocks:
            tok = block(tok, c)
        return self.unpatchify(self.final_layer(tok, c), h, w)


def SiT_XL_2(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def SiT_L_2(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def SiT_B_2(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def SiT_S_2(**kwargs):
    return SiT(depth=12, hidden_size=384, decoder_hidden_size=384, patch_size=2, num_heads=6, **kwargs)


SiT_models = {"SiT-XL/2": SiT_XL_2, "SiT-L/2": SiT_L_2, "SiT-B/2": SiT_B_2, "SiT-S/2": SiT_S_2}
