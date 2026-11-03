# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

"""Block-causal joint MMDiT over image latents and text latents.

One transformer models both modalities with separate per-modality parameters
(MMDiT style) and per-token adaLN conditioning, so each modality — and, within
text, each block — carries its own timestep.

Two directions are trained, sampled *per sample* so that one batch carries both:

* ``t2i``: text is the clean condition, the image latent is denoised.
* ``i2t``: the image latent is the clean condition, text is denoised under the
  2L layout ``[clean | noisy]``, which lets every text block receive a gradient
  in one forward while still conditioning on a *clean* history.

Training runs the 2L layout for both directions so they share one sequence
length; a t2i sample leaves the noisy slot masked out and unsupervised, which
costs it ``L`` dead positions. Mixing beats alternating whole batches because the
image VAE then receives its diffusion gradient on every step rather than half of
them, and the attention mask is per-sample anyway, so nothing else has to change.

The text stream is block-causal (bidirectional inside a block, causal across
blocks) in both directions, so the text representation is computed identically
whichever way the model is being used. Attention never crosses samples, and the
condition modality never attends to the generated one, so the branch being
denoised can never read the clean copy of itself.

Image latents are normalized by a BatchNorm whose running stats are loaded from
the frozen VAE's ``latents-stats`` and then held fixed; keeping the BN (rather
than normalizing outside) preserves compatibility with ``generate.py``, which
recovers the latent statistics from the checkpoint. Text latents get no BN: the
text VAE already applies a non-affine LayerNorm to the posterior mean, and a
second, drifting normalization would only fight it.

Passing ``z_dims`` adds REPA projection heads on the **image** stream, which is
the only one with an external encoder to align to. They read the hidden state
after joint block ``encoder_depth`` and are returned alongside the velocity
predictions; the alignment loss itself lives in the training script.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mmdit_e2e import TimestepEmbedder, get_2d_sincos_pos_embed, _1d_sincos


def modulate(x, shift, scale):
    """Per-token adaLN. ``x``/``shift``/``scale`` are all ``(B, S, D)``."""
    return x * (1 + scale) + shift


# --------------------------------------------------------------------------
# masks
# --------------------------------------------------------------------------

TASK_T2I, TASK_I2T = "t2i", "i2t"


def _as_direction(is_t2i, B, device):
    """Normalize a direction argument to a ``(B,)`` bool tensor.

    Directions are per-sample so that one batch can mix them, but the samplers
    drive a whole batch one way and are clearer passing ``"t2i"``.
    """
    if isinstance(is_t2i, str):
        assert is_t2i in (TASK_T2I, TASK_I2T), is_t2i
        is_t2i = is_t2i == TASK_T2I
    if not torch.is_tensor(is_t2i):
        return torch.full((B,), bool(is_t2i), dtype=torch.bool, device=device)
    return is_t2i.to(device=device, dtype=torch.bool)


def build_joint_mask(n_img, text_len, text_valid, block_size, is_t2i, use_2l, cond_drop=None):
    """Boolean attention mask, ``True`` where attention is allowed.

    Layout is ``[image (n_img) | text_clean (L) | text_noisy (L)]``, with the
    noisy copy present only when ``use_2l``. Returns ``(B, 1, S, S)`` so it
    broadcasts over heads in ``scaled_dot_product_attention``.

    ``is_t2i`` is a ``(B,)`` bool, so a single batch can carry both directions:
    the mask is already per-sample, which is what makes mixing free. Training
    always runs the 2L layout so that both directions share one sequence length;
    a t2i sample keeps its clean text in the clean slot exactly as it did when
    the whole batch shared a direction, and its noisy slot is fully isolated
    below and excluded from the loss. ``use_2l=False`` drops the noisy slot
    entirely and then requires every sample to be t2i -- that is the sampling
    fast path, where the direction is uniform by construction.

    The text rules for the 2L layout follow ``create_2l_block_causal_mask`` in
    the vendored Cola attention utils: a noisy query at block ``b`` sees clean
    blocks ``0..b-1`` and its own noisy block ``b``, and nothing else. Because
    noisy blocks never see each other, all blocks train independently in one
    forward, each on a clean history.

    ``cond_drop`` is a ``(B,)`` bool marking samples whose *condition modality*
    is hidden: the cross-modal block is cleared, so the generated modality sees
    only itself. This is the prefix-dropping form of condition dropout, and the
    branch it trains is what classifier-free guidance interpolates against. No
    null token is needed — the absence of the condition is the signal.
    """
    B = text_valid.shape[0]
    L = text_len
    device = text_valid.device
    n_txt = 2 * L if use_2l else L
    S = n_img + n_txt
    assert is_t2i.shape == (B,), is_t2i.shape
    assert use_2l or bool(is_t2i.all()), "the no-2L layout only serves t2i"

    blk = (torch.arange(L, device=device) // block_size)
    q_blk = torch.cat([blk, blk]) if use_2l else blk  # (n_txt,)
    is_noisy = torch.zeros(n_txt, dtype=torch.bool, device=device)
    if use_2l:
        is_noisy[L:] = True

    valid = torch.cat([text_valid, text_valid], dim=1) if use_2l else text_valid  # (B, n_txt)
    # A t2i sample is supervised on the image only, so its noisy slot is dead
    # weight: treating it as invalid isolates it as both query and key, and the
    # no-visible-key fallback below parks it on its own diagonal.
    valid = valid & ~(is_t2i.view(B, 1) & is_noisy.view(1, n_txt))

    mask = torch.zeros(B, S, S, dtype=torch.bool, device=device)
    img, txt = slice(0, n_img), slice(n_img, S)

    # Image queries: bidirectional over image.
    mask[:, img, img] = True

    qb, kb = q_blk.unsqueeze(1), q_blk.unsqueeze(0)  # (n_txt, 1), (1, n_txt)
    qn, kn = is_noisy.unsqueeze(1), is_noisy.unsqueeze(0)
    # Same three rules as the vendored 2L mask; without the noisy copy this
    # reduces to plain block-causal attention. A t2i sample keeps only the first
    # rule, its noisy slot having been marked invalid, which is exactly the
    # block-causal text attention it had in the no-2L layout.
    t2t = (
        ((~qn) & (~kn) & (qb >= kb))       # clean -> clean, block-causal
        | (qn & (~kn) & (qb > kb))         # noisy -> clean history, strict
        | (qn & kn & (qb == kb))           # noisy -> own noisy block
    )
    # Drop attention to (and from) padding positions.
    t2t = t2t.unsqueeze(0) & valid.unsqueeze(1) & valid.unsqueeze(2)
    mask[:, txt, txt] = t2t

    keep = torch.ones(B, dtype=torch.bool, device=device) if cond_drop is None else ~cond_drop
    # Text is the clean condition for t2i: the image reads it, it does not read
    # back. Image is the clean condition for i2t, the other way round. Both
    # blocks are written and then gated by the direction, so a mixed batch needs
    # no separate forward per direction.
    cross = (keep & is_t2i).view(B, 1, 1)
    mask[:, img, txt] = valid.unsqueeze(1).expand(-1, n_img, -1) & cross
    cross = (keep & ~is_t2i).view(B, 1, 1)
    mask[:, txt, img] = valid.unsqueeze(2).expand(-1, -1, n_img) & cross

    # A query with no visible key would make softmax produce NaN. Padding
    # queries are excluded from every loss, so give them their own diagonal.
    eye = torch.eye(S, dtype=torch.bool, device=device).unsqueeze(0)
    mask = mask | (~mask.any(dim=-1, keepdim=True) & eye)
    return mask.unsqueeze(1)


# --------------------------------------------------------------------------
# blocks
# --------------------------------------------------------------------------


class StreamBlock(nn.Module):
    """One modality's parameters, split around a shared joint attention."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def pre_attention(self, x, c):
        mods = self.adaLN_modulation(c).chunk(6, dim=-1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods
        qkv = self.qkv(modulate(self.norm1(x), shift_msa, scale_msa))
        B, L, _ = qkv.shape
        qkv = qkv.reshape(B, L, 3, -1).movedim(2, 0)
        return (qkv[0], qkv[1], qkv[2]), (x, gate_msa, shift_mlp, scale_mlp, gate_mlp)

    def post_attention(self, attn, x, gate_msa, shift_mlp, scale_mlp, gate_mlp):
        x = x + gate_msa * self.proj(attn)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class JointBlock(nn.Module):
    """Image and text streams joined by one masked attention."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.img_block = StreamBlock(hidden_size, num_heads, mlp_ratio, qkv_bias)
        self.txt_block = StreamBlock(hidden_size, num_heads, mlp_ratio, qkv_bias)

    def forward(self, img, txt, c_img, c_txt, attn_mask):
        img_qkv, img_inter = self.img_block.pre_attention(img, c_img)
        txt_qkv, txt_inter = self.txt_block.pre_attention(txt, c_txt)

        q, k, v = (torch.cat([a, b], dim=1) for a, b in zip(img_qkv, txt_qkv))
        B, S, _ = q.shape
        q, k, v = (
            t.view(B, S, self.num_heads, self.head_dim).transpose(1, 2) for t in (q, k, v)
        )
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, S, -1)

        n_img = img.shape[1]
        img = self.img_block.post_attention(out[:, :n_img], *img_inter)
        txt = self.txt_block.post_attention(out[:, n_img:], *txt_inter)
        return img, txt


def build_projector(hidden_size, projector_dim, z_dim):
    """REPA projection head, same shape as ``Prior_only/models/mmdit.py:build_mlp``."""
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------


class BlockCausalJointMMDiT(nn.Module):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        image_channels=16,
        text_channels=16,
        text_block_size=16,
        max_text_tokens=512,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        path_type="linear",
        bn_momentum=0.0,
        cond_dropout=0.0,
        z_dims=(),
        encoder_depth=8,
        projector_dim=2048,
        detach_condition=False,
    ):
        super().__init__()
        self.cond_dropout = cond_dropout
        # When set, a modality that only conditions carries no gradient at all:
        # the clean text slot is detached for every sample rather than just for
        # i2t ones. The caller has to cut the image side the same way, since the
        # image latent arrives from outside.
        self.detach_condition = detach_condition
        self.patch_size = patch_size
        self.image_channels = image_channels
        self.text_channels = text_channels
        self.text_block_size = text_block_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.path_type = path_type

        self.grid = input_size // patch_size
        self.n_img = self.grid * self.grid
        self.img_token_dim = image_channels * patch_size * patch_size

        self.img_embedder = nn.Linear(self.img_token_dim, hidden_size)
        self.txt_embedder = nn.Linear(text_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        # Two directions and three token roles (image, clean text, noisy text).
        self.task_embed = nn.Parameter(torch.zeros(2, hidden_size))
        self.type_embed = nn.Parameter(torch.zeros(3, hidden_size))

        self.register_buffer("img_pos_embed", torch.zeros(1, self.n_img, hidden_size))
        self.register_buffer("txt_pos_embed", torch.zeros(1, max_text_tokens, hidden_size))

        self.blocks = nn.ModuleList(
            [JointBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.img_final = FinalLayer(hidden_size, self.img_token_dim)
        self.txt_final = FinalLayer(hidden_size, text_channels)

        # REPA alignment, image stream only: there is no frozen encoder that
        # produces a per-token target for text.
        self.encoder_depth = encoder_depth
        self.projectors = nn.ModuleList(
            [build_projector(hidden_size, projector_dim, z_dim) for z_dim in z_dims]
        )

        # Holds the image VAE's latent statistics. With a frozen VAE only the
        # buffers are used -- normalization is applied explicitly in
        # `normalize_image_latents` so the scaling cannot switch to batch
        # statistics in train mode. Keeping the module named `bn` lets
        # `generate.py` recover the stats from a checkpoint the same way it does
        # for the other models.
        self.bn = nn.BatchNorm2d(
            image_channels, eps=1e-4, momentum=bn_momentum, affine=False,
            track_running_stats=True,
        )
        self.bn.reset_running_stats()
        # Set when the image VAE trains: its latent distribution then drifts and
        # fixed buffers would go stale, so the BatchNorm runs for real. See
        # `normalize_image_latents`.
        self.track_latent_stats = False

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_basic_init)

        img_pos = get_2d_sincos_pos_embed(self.hidden_size, self.grid)
        self.img_pos_embed.data.copy_(torch.from_numpy(img_pos).float().unsqueeze(0))
        n_txt = self.txt_pos_embed.shape[1]
        txt_pos = _1d_sincos(self.hidden_size, np.arange(n_txt, dtype=np.float32))
        self.txt_pos_embed.data.copy_(torch.from_numpy(txt_pos).float().unsqueeze(0))

        nn.init.normal_(self.task_embed, std=0.02)
        nn.init.normal_(self.type_embed, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            for stream in (block.img_block, block.txt_block):
                nn.init.constant_(stream.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(stream.adaLN_modulation[-1].bias, 0)
        for head in (self.img_final, self.txt_final):
            nn.init.constant_(head.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(head.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(head.linear.weight, 0)
            nn.init.constant_(head.linear.bias, 0)

    # ---------------------------------------------------------------- latents
    def init_bn(self, latents_scale, latents_bias):
        self.bn.running_mean = latents_bias.to(self.bn.running_mean.device).flatten()
        self.bn.running_var = (1.0 / latents_scale.to(self.bn.running_var.device).flatten()).pow(2)

    def extract_latents_stats(self):
        return dict(latents_scale=self.bn.running_var.rsqrt(), latents_bias=self.bn.running_mean)

    def normalize_image_latents(self, z):
        if self.track_latent_stats:
            # Batch statistics normalize and the running estimate trails behind
            # for inference, as in SiT's end-to-end path. Needed whenever the
            # encoder moves, or when the buffers were only seeded from FLUX's
            # scalar shift/scale rather than measured per channel.
            return self.bn(z)
        bias = self.bn.running_mean.view(1, -1, 1, 1)
        scale = self.bn.running_var.rsqrt().view(1, -1, 1, 1)
        return (z - bias) * scale

    def denormalize_image_latents(self, z):
        """Back to raw VAE space, so sampled latents can go straight to decode."""
        bias = self.bn.running_mean.view(1, -1, 1, 1)
        scale = self.bn.running_var.rsqrt().view(1, -1, 1, 1)
        return z / scale + bias

    def _patchify(self, x):
        p = self.patch_size
        B, C, H, W = x.shape
        h, w = H // p, W // p
        x = x.reshape(B, C, h, p, w, p).permute(0, 2, 4, 1, 3, 5).contiguous()
        return x.reshape(B, h * w, C * p * p)

    def _unpatchify(self, tokens):
        p, C, g = self.patch_size, self.image_channels, self.grid
        B = tokens.shape[0]
        x = tokens.reshape(B, g, g, C, p, p).permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.reshape(B, C, g * p, g * p)

    def interpolant(self, t):
        if self.path_type == "linear":
            return 1 - t, t, torch.full_like(t, -1.0), torch.full_like(t, 1.0)
        if self.path_type == "cosine":
            return (
                torch.cos(t * np.pi / 2),
                torch.sin(t * np.pi / 2),
                -np.pi / 2 * torch.sin(t * np.pi / 2),
                np.pi / 2 * torch.cos(t * np.pi / 2),
            )
        raise NotImplementedError(self.path_type)

    def sample_text_timesteps(self, B, n_blocks, device, dtype):
        """One timestep per text block, expanded to per-token."""
        t_blk = torch.rand((B, n_blocks), device=device, dtype=dtype)
        return t_blk.repeat_interleave(self.text_block_size, dim=1), t_blk

    # ---------------------------------------------------------------- forward
    def denoise(
        self, img_t, txt_noisy, text_valid, is_t2i, t_img, t_txt_tok,
        txt_clean=None, cond_drop=None, align_only=False,
    ):
        """Velocity prediction for inputs that are already noised.

        This is the whole trunk; :meth:`forward` only adds noise to clean latents
        and compares against the target, so training and sampling cannot drift
        apart.

        Args:
            img_t: noisy image latent ``(B, C, H, W)``, **already normalized**.
            txt_noisy: noisy text latent ``(B, L, text_channels)``.
            text_valid: ``(B, L)`` bool marking positions to attend to.
            is_t2i: ``(B,)`` bool giving each sample's direction. A python bool
                or a plain ``str`` task name is accepted and broadcast, which is
                what the samplers pass since their batches are one direction.
            t_img: ``(B,)`` image timestep.
            t_txt_tok: ``(B, L)`` per-token text timestep.
            txt_clean: clean text history ``(B, L, text_channels)``. When given,
                the 2L layout ``[clean | noisy]`` is used.
            cond_drop: ``(B,)`` bool hiding the condition modality per sample.
            align_only: stop after the projections and return
                ``(None, None, zs)``, skipping the remaining blocks and both
                output heads. This is the cheap pass REPA-E uses to send the
                alignment gradient into the VAE alone, as in ``models/sit.py``.

        Returns:
            ``(v_img_pred, v_txt_pred, zs)``. ``v_txt_pred`` covers the noisy
            copy only; ``zs`` holds the REPA projections of the image stream, and
            is empty when the model was built without ``z_dims``.
        """
        assert not align_only or self.projectors, "align_only needs z_dims"
        B, L = txt_noisy.shape[0], txt_noisy.shape[1]
        is_t2i = _as_direction(is_t2i, B, txt_noisy.device)
        use_2l = txt_clean is not None

        img_tok = self.img_embedder(self._patchify(img_t)) + self.img_pos_embed
        img_tok = img_tok + self.type_embed[0].view(1, 1, -1)

        txt_pos = self.txt_pos_embed[:, :L]
        if use_2l:
            clean_tok = self.txt_embedder(txt_clean) + txt_pos
            clean_tok = clean_tok + self.type_embed[1].view(1, 1, -1)
            noisy_tok = self.txt_embedder(txt_noisy) + txt_pos
            noisy_tok = noisy_tok + self.type_embed[2].view(1, 1, -1)
            txt_tok = torch.cat([clean_tok, noisy_tok], dim=1)
            t_txt_all = torch.cat([torch.zeros_like(t_txt_tok), t_txt_tok], dim=1)
        else:
            txt_tok = self.txt_embedder(txt_noisy) + txt_pos
            txt_tok = txt_tok + self.type_embed[1].view(1, 1, -1)
            t_txt_all = t_txt_tok

        # Per-token conditioning: own timestep plus the direction being run.
        task_vec = self.task_embed[torch.where(is_t2i, 0, 1)].view(B, 1, -1)
        c_img = self.t_embedder(
            t_img.view(B, 1).expand(B, self.n_img).reshape(-1)
        ).view(B, self.n_img, -1) + task_vec
        c_txt = self.t_embedder(t_txt_all.reshape(-1)).view(B, t_txt_all.shape[1], -1) + task_vec

        attn_mask = build_joint_mask(
            self.n_img, L, text_valid, self.text_block_size, is_t2i, use_2l,
            cond_drop=cond_drop,
        )

        img_h, txt_h = img_tok, txt_tok
        zs = []
        for i, block in enumerate(self.blocks):
            img_h, txt_h = block(img_h, txt_h, c_img, c_txt, attn_mask)
            if self.projectors and (i + 1) == self.encoder_depth:
                zs = [proj(img_h) for proj in self.projectors]
                if align_only:
                    return None, None, zs

        # Both heads run in both directions even though only one is supervised.
        # Sampling the direction per step would otherwise leave the other head
        # without a gradient, which breaks DDP unless find_unused_parameters is
        # on; the idle head is folded in with a zero coefficient instead.
        v_img_pred = self._unpatchify(self.img_final(img_h, c_img))
        v_txt_pred = self.txt_final(txt_h[:, -L:], c_txt[:, -L:])
        return v_img_pred, v_txt_pred, zs

    def forward(
        self,
        z_img,
        z_txt,
        text_valid,
        is_t2i,
        t_img=None,
        t_txt=None,
        eps_img=None,
        eps_txt=None,
        cond_drop=None,
        align_only=False,
    ):
        """Run both directions in one batch and return the per-sample loss.

        Args:
            z_img: clean image latent ``(B, C, H, W)``, unnormalized.
            z_txt: clean text latent ``(B, L, text_channels)``, already
                LayerNorm-normalized by the text VAE. ``L`` must be a multiple
                of ``text_block_size``.
            text_valid: ``(B, L)`` bool marking real (non-padding) positions.
            is_t2i: ``(B,)`` bool giving each sample's direction, so one batch
                can carry both. Each sample is supervised on the modality it
                generates and contributes nothing to the other one's loss.
            t_img, t_txt, eps_img, eps_txt, cond_drop: draws to reuse instead of
                sampling. Everything sampled here is returned, so the alignment
                pass and the denoising pass can be run on identical inputs the
                way ``train_e2e.py`` reuses ``time_input`` and ``noises``.
            align_only: return only ``zs`` (and the draws), for the pass whose
                gradient is meant for the VAE alone.
        """
        B, L = z_txt.shape[0], z_txt.shape[1]
        assert L % self.text_block_size == 0, (L, self.text_block_size)
        device, dtype = z_txt.device, z_txt.dtype
        is_t2i = _as_direction(is_t2i, B, device)

        z_img_n = self.normalize_image_latents(z_img)

        if eps_img is None:
            eps_img = torch.randn_like(z_img_n)
        if eps_txt is None:
            eps_txt = torch.randn_like(z_txt)

        # The condition modality is clean, i.e. sits at t=0, and the generated
        # one is noised. Both draws happen for every sample and the direction
        # picks which one survives.
        if t_img is None:
            t_img = torch.rand((B,), device=device, dtype=dtype)
        if t_txt is None:
            t_txt, _ = self.sample_text_timesteps(
                B, L // self.text_block_size, device, dtype
            )
        t_img = torch.where(is_t2i, t_img, torch.zeros_like(t_img))
        t_txt_tok = torch.where(is_t2i.view(B, 1), torch.zeros_like(t_txt), t_txt)

        a_i, s_i, da_i, ds_i = self.interpolant(t_img.view(B, 1, 1, 1))
        img_t = a_i * z_img_n + s_i * eps_img
        v_img_target = da_i * z_img_n + ds_i * eps_img

        a_t, s_t, da_t, ds_t = self.interpolant(t_txt_tok.unsqueeze(-1))
        txt_noisy = a_t * z_txt + s_t * eps_txt
        v_txt_target = da_t * z_txt + ds_t * eps_txt

        # For i2t the clean copy is a conditioning signal only: detaching it
        # keeps the diffusion gradient off the history path, which would
        # otherwise bypass the coefficient that scales the noisy path. For t2i
        # the clean slot *is* the text input, so it has to stay attached or the
        # text encoder would only ever hear from i2t samples -- unless
        # `detach_condition` asks for exactly that, cutting the t2i caption out
        # of the text VAE's training entirely.
        if self.detach_condition:
            txt_clean = z_txt.detach()
        else:
            txt_clean = torch.where(is_t2i.view(B, 1, 1), z_txt, z_txt.detach())

        if cond_drop is None and self.training and self.cond_dropout > 0:
            cond_drop = torch.rand((B,), device=device) < self.cond_dropout

        draws = dict(
            t_img=t_img, t_txt=t_txt_tok, eps_img=eps_img, eps_txt=eps_txt,
            cond_drop=cond_drop,
        )

        v_img_pred, v_txt_pred, zs = self.denoise(
            img_t, txt_noisy, text_valid, is_t2i, t_img, t_txt_tok,
            txt_clean=txt_clean, cond_drop=cond_drop, align_only=align_only,
        )
        if align_only:
            return dict(zs=zs, **draws)

        out = dict(
            v_img_pred=v_img_pred, v_txt_pred=v_txt_pred, zs=zs, **draws,
        )
        loss_img = ((v_img_pred - v_img_target) ** 2).mean(dim=(1, 2, 3))
        err = ((v_txt_pred - v_txt_target) ** 2).mean(dim=-1)
        # Only i2t samples are supervised on text; a t2i sample's noisy slot is
        # masked out of attention and gets weight 0 here.
        w = (text_valid & ~is_t2i.view(B, 1)).to(err.dtype)
        loss_txt = (err * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)

        # Both heads stay in the graph for every sample: `where` backpropagates
        # zeros into the branch it did not take, which is what keeps DDP from
        # tripping over an unused head when a batch happens to be one-directional.
        out["prior_loss"] = torch.where(is_t2i, loss_img, loss_txt)
        return out


# ------------------------------ configs -------------------------------------
def JointMMDiT_XL_2(**kwargs):
    return BlockCausalJointMMDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def JointMMDiT_L_2(**kwargs):
    return BlockCausalJointMMDiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def JointMMDiT_B_2(**kwargs):
    return BlockCausalJointMMDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


JointMMDiT_models = {
    "JointMMDiT-XL/2": JointMMDiT_XL_2,
    "JointMMDiT-L/2": JointMMDiT_L_2,
    "JointMMDiT-B/2": JointMMDiT_B_2,
}
