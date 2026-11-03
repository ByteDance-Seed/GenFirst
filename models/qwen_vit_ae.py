# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

"""Autoencoder whose encoder is Qwen3-VL's vision tower.

The motivation is the one measurement that sank the previous approach: an
ImageNet ridge linear probe reads 67.1% off this ViT's features and 3.1% off a
FLUX latent. Asking a head to *create* CLIP-space structure inside a code trained
only for L1+LPIPS is a different and much harder problem than asking a
bottleneck to *preserve* structure that is already there. Unified
recon+understanding autoencoders are built the second way -- the encoder is a
pretrained semantic ViT and the latent is a linear projection of that trunk --
and this module is the same construction with Qwen3-VL's tower in that role.

Latent geometry is chosen so that nothing downstream changes. The tower is
patch-16, so a 256px image gives a 16x16 grid of tokens; each token's 1024-d
feature is bottlenecked to ``C * merge^2`` numbers and *reshaped* into a
``merge x merge`` spatial block of ``C`` channels. With C=16 and merge=2 the
latent is (B, 16, 32, 32) -- byte-identical in shape to FLUX's f8 latent, so
``latent_size = resolution // 8``, ``in_channels = 16``, the MMDiT's patch-2
embedder, its latent-normalising BatchNorm, the entropy regulariser and the
whole loss config carry over untouched. MMDiT tokens then correspond 1:1 to ViT
tokens, and each ViT token's 16x16 pixel patch maps onto exactly the 2x2 latent
block that covers those pixels.

Two things about the tower's token order matter and are easy to get wrong. The
official image processor packs patches in *merge-block-major* order (each group
of ``merge^2`` consecutive tokens is one 2x2 spatial block) because the merger
does ``view(-1, hidden * merge^2)`` on consecutive tokens. ``_pack`` reproduces
that packing exactly (verified against ``Qwen2VLImageProcessorFast``) so the
tower's own position and rope logic stays correct, and ``_to_raster`` then
undoes the permutation before the latent is built.

Requires ``transformers >= 4.57`` for Qwen3-VL. That is an optional dependency,
needed only by this module and models/understanding_prior.py.
"""

import contextlib
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from dictdot import dictdot

from .autoencoder import DiagonalGaussianDistribution


DEFAULT_QWEN3VL = "Qwen/Qwen3-VL-2B-Instruct"


class VlmTokens(NamedTuple):
    """The vision tower's output, in the form its language model consumes.

    ``embeds`` are the merged image tokens, (B, n_tok, llm_dim). ``deepstack`` are
    the same-shaped features tapped at ``deepstack_visual_indexes`` and added into
    the language model's first few layers; dropping them would feed the model a
    strictly poorer input than it was trained on. ``grid`` is the token grid after
    merging, which the prompt length and the mrope ids are derived from.
    """

    embeds: torch.Tensor
    deepstack: list
    grid: tuple

# Qwen3-VL inherits CLIP's normalisation statistics.
QWEN_MEAN = (0.48145466, 0.4578275, 0.40821073)
QWEN_STD = (0.26862954, 0.26130258, 0.27577711)


class _Block(nn.Module):
    def __init__(self, width, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        hidden = int(width * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class ViTDecoder(nn.Module):
    """Latent -> pixels, one ``patch x patch`` RGB block per token.

    ``conv_refine`` is a deliberate deviation from that plain design. Predicting a
    16x16 patch per token with no spatial mixing at pixel level leaves visible
    seams at patch borders; that is affordable when the decoder is warm-started
    from an already GAN-trained AE, but this one starts from scratch. Two 3x3
    convolutions on the assembled image cost almost nothing and give the seams
    somewhere to go.
    """

    def __init__(self, latent_channels, merge, patch, width=1024, depth=12,
                 num_heads=16, grid=16, conv_refine=True):
        super().__init__()
        self.merge = merge
        self.patch = patch
        self.stem = nn.Linear(latent_channels * merge * merge, width)
        self.pos = nn.Parameter(torch.zeros(1, width, grid, grid))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([_Block(width, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 3 * patch * patch)
        self.refine = None
        if conv_refine:
            self.refine = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 3, 3, padding=1),
            )
            nn.init.zeros_(self.refine[-1].weight)
            nn.init.zeros_(self.refine[-1].bias)

    def _pos_tokens(self, h, w):
        pos = self.pos
        if pos.shape[-2] != h or pos.shape[-1] != w:
            pos = F.interpolate(pos, size=(h, w), mode="bicubic", align_corners=False)
        return pos.flatten(2).transpose(1, 2)

    def forward(self, z):
        b, c, hh, ww = z.shape
        m = self.merge
        gh, gw = hh // m, ww // m
        # (B, C, gh*m, gw*m) -> (B, gh*gw, C*m*m), raster order over the token grid.
        x = z.reshape(b, c, gh, m, gw, m).permute(0, 2, 4, 1, 3, 5).reshape(b, gh * gw, c * m * m)
        x = self.stem(x) + self._pos_tokens(gh, gw)
        for blk in self.blocks:
            x = blk(x)
        x = self.head(self.norm(x))
        p = self.patch
        img = x.reshape(b, gh, gw, 3, p, p).permute(0, 3, 1, 4, 2, 5).reshape(b, 3, gh * p, gw * p)
        if self.refine is not None:
            img = img + self.refine(img)
        return img


class QwenViTAutoencoder(nn.Module):
    """Qwen3-VL vision tower + Gaussian bottleneck + ViT decoder.

    Exposes the interface the training loop and losses expect: ``forward`` returns
    ``(posterior, z, recon)``, ``encode`` returns a ``DiagonalGaussianDistribution``
    and ``decode`` returns ``dictdot(sample=...)``.
    """

    def __init__(
        self,
        model_path=DEFAULT_QWEN3VL,
        latent_channels=16,
        decoder_width=1024,
        decoder_depth=12,
        decoder_heads=16,
        decoder_grid=16,
        conv_refine=True,
        freeze_encoder=True,
        grad_checkpointing=True,
        logvar_init=-6.0,
        load_dtype=torch.bfloat16,
    ):
        super().__init__()
        try:
            from transformers import Qwen3VLForConditionalGeneration
        except ImportError as e:
            import transformers

            raise ImportError(
                "--vae qwenvit needs Qwen3-VL, which requires transformers>=4.57 "
                f"(found {transformers.__version__}). Install it with: "
                "pip install -U 'transformers>=4.57'"
            ) from e

        try:
            vlm = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, dtype=load_dtype, attn_implementation="sdpa",
            )
        except TypeError:  # older kwarg name
            vlm = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=load_dtype, attn_implementation="sdpa",
            )
        cfg = vlm.config.vision_config
        self.patch = int(cfg.patch_size)
        self.merge = int(cfg.spatial_merge_size)
        self.vit_dim = int(cfg.hidden_size)
        self.llm_dim = int(cfg.out_hidden_size)
        self.temporal_patch = int(getattr(cfg, "temporal_patch_size", 2))

        # Only the vision tower is needed here; the language stack is loaded
        # separately (and only when the NLL understanding loss is selected).
        self.visual = vlm.model.visual.float()
        del vlm

        self.embed_dim = latent_channels
        self.latent_channels = latent_channels
        self.freeze_encoder = freeze_encoder

        # The merger stays frozen in every stage, for two reasons. It is the
        # tower's interface to the language model, so letting it move would let the
        # caption-NLL objective be reduced by reshaping the LLM's input space
        # instead of by making the latent readable -- the loss would stop measuring
        # what it is supposed to measure. It is also unused by the reconstruction
        # path, which taps features before it, so once the tower is unfrozen DDP
        # aborts on it as a parameter that received no gradient.
        self.visual.merger.requires_grad_(False)
        if hasattr(self.visual, "deepstack_merger_list"):
            self.visual.deepstack_merger_list.requires_grad_(False)

        if freeze_encoder:
            self.visual.requires_grad_(False)
            self.visual.eval()
        elif grad_checkpointing:
            try:
                self.visual.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except AttributeError:
                self.visual.gradient_checkpointing = True

        self.register_buffer("norm_mean", torch.tensor(QWEN_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("norm_std", torch.tensor(QWEN_STD).view(1, 3, 1, 1), persistent=False)

        # The bottleneck: a single encoder projection, split into two heads so the
        # mean/logvar layout in `moments` cannot be mixed up.
        #
        # The LayerNorm is not optional. Pre-merge features are a raw residual
        # stream with entries in the hundreds; reading them directly gave a latent
        # with std 36 and a mean^2 term of 1.1e7, which made the auxiliary KL
        # (10.9) dominate the reconstruction loss (0.69). The tower's own consumer
        # does exactly this normalisation as the merger's first op, so seed from
        # its weights and inherit the calibration the tower was trained with.
        self.pre_norm = nn.LayerNorm(self.vit_dim)
        with torch.no_grad():
            src = self.visual.merger.norm
            if src.weight.shape == self.pre_norm.weight.shape:
                self.pre_norm.weight.copy_(src.weight.float())
                self.pre_norm.bias.copy_(src.bias.float())

        out_per_token = latent_channels * self.merge * self.merge
        self.to_mean = nn.Linear(self.vit_dim, out_per_token)
        self.to_logvar = nn.Linear(self.vit_dim, out_per_token)
        # Start near-deterministic at the variance the trained FLUX encoder
        # settles on (logged latent/var_mean ~= 0.044, i.e. logvar ~= -3.1);
        # a unit-variance start would drown the mean in sampling noise.
        nn.init.zeros_(self.to_logvar.weight)
        nn.init.constant_(self.to_logvar.bias, logvar_init)
        # Measured: the mean head's output has std ~0.22 at init, so the default
        # logvar of -6 puts the posterior std at 0.05, roughly 4x below the signal.
        # -3 (chosen for a unit-std mean) made noise and signal equal power, which
        # just hands the decoder noise to reconstruct. The entropy regulariser
        # pushes the variance up from here, so erring on the quiet side is right.

        self.decoder = ViTDecoder(
            latent_channels=latent_channels, merge=self.merge, patch=self.patch,
            width=decoder_width, depth=decoder_depth, num_heads=decoder_heads,
            grid=decoder_grid, conv_refine=conv_refine,
        )

    def train(self, mode=True):
        """Keep a frozen tower in eval mode.

        The training loop calls ``vae.train()`` every step, which would otherwise
        drag the frozen tower along with it. Qwen3-VL's vision stack has no
        dropout so the two modes are numerically identical today, but relying on
        that is how a future config change turns into a silent train/eval skew.
        """
        super().train(mode)
        if self.freeze_encoder:
            self.visual.eval()
        return self

    # -- tower plumbing -----------------------------------------------------
    def _pack(self, x):
        """[-1,1] images -> (packed patches, grid_thw) in the processor's order."""
        x = (x + 1.0) / 2.0
        x = (x - self.norm_mean.to(x.dtype)) / self.norm_std.to(x.dtype)
        b, c, h, w = x.shape
        p, m, t = self.patch, self.merge, self.temporal_patch
        assert h % (p * m) == 0 and w % (p * m) == 0, (
            f"{h}x{w} must be divisible by patch*merge = {p * m}"
        )
        gh, gw = h // p, w // p
        # Matches Qwen2VLImageProcessor exactly: merge-block-major, then channel,
        # temporal and the two intra-patch axes.
        v = x[:, None].expand(b, t, c, h, w)
        v = v.reshape(b, t, c, gh // m, m, p, gw // m, m, p)
        v = v.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        packed = v.reshape(b * gh * gw, c * t * p * p)
        grid = torch.tensor([[1, gh, gw]] * b, dtype=torch.long, device=x.device)
        return packed, grid, (gh, gw)

    def _to_raster(self, feats, gh, gw):
        """(B, gh*gw, D) merge-block-major -> (B, gh, gw, D) raster."""
        b, n, d = feats.shape
        m = self.merge
        return (
            feats.reshape(b, gh // m, gw // m, m, m, d)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(b, gh, gw, d)
        )

    def _from_raster(self, feats):
        """Inverse of ``_to_raster``: (B, gh, gw, D) -> (B, gh*gw, D)."""
        b, gh, gw, d = feats.shape
        m = self.merge
        return (
            feats.reshape(b, gh // m, m, gw // m, m, d)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(b, gh * gw, d)
        )

    def vit_forward(self, x):
        """One tower pass, both taps.

        Returns ``(feats, vlm)``. ``feats`` is the pre-merge residual stream in
        raster order, (B, gh, gw, vit_dim) -- what the bottleneck reads. ``vlm`` is
        the tower's *own* output, i.e. bit-for-bit what Qwen3-VL hands to its
        language model: merged image embeddings plus the deepstack features tapped
        at the intermediate layers. The merger runs either way (it is inside the
        tower's forward), so taking its output costs nothing beyond the reshape,
        and an understanding loss built on ``vlm`` shares this single pass with
        reconstruction instead of paying for a second one.
        """
        packed, grid, (gh, gw) = self._pack(x)
        captured = {}

        def hook(_module, args):
            captured["h"] = args[0]

        handle = self.visual.merger.register_forward_pre_hook(hook)
        try:
            out = self.visual(packed, grid_thw=grid)
        finally:
            handle.remove()

        b = x.shape[0]
        feats = self._to_raster(captured["h"].reshape(b, gh * gw, self.vit_dim), gh, gw)
        merged, deepstack = out if isinstance(out, (tuple, list)) else (out, ())
        m = self.merge
        vlm = VlmTokens(
            embeds=merged.reshape(b, -1, self.llm_dim),
            deepstack=[d.reshape(b, -1, self.llm_dim) for d in (deepstack or ())],
            grid=(gh // m, gw // m),
        )
        return feats, vlm

    def vit_features(self, x):
        """Pre-merge last-block features, in raster order: (B, gh, gw, vit_dim)."""
        return self.vit_forward(x)[0]

    # -- E2E-DM VAE interface ----------------------------------------------
    def encode(self, x, return_vlm=False):
        # nullcontext, not enable_grad: the latter would override an ambient
        # no_grad and start building a graph through the tower on the sampling and
        # evaluation paths.
        ctx = torch.no_grad() if self.freeze_encoder else contextlib.nullcontext()
        with ctx:
            feats, vlm = self.vit_forward(x)
        if self.freeze_encoder:
            feats = feats.detach()
        feats = self.pre_norm(feats)
        mean = self._unpatchify_latent(self.to_mean(feats))
        logvar = self._unpatchify_latent(self.to_logvar(feats))
        posterior = DiagonalGaussianDistribution(torch.cat([mean, logvar], dim=1))
        return (posterior, vlm) if return_vlm else posterior

    def _unpatchify_latent(self, tok):
        """(B, gh, gw, C*m*m) -> (B, C, gh*m, gw*m)."""
        b, gh, gw, _ = tok.shape
        c, m = self.latent_channels, self.merge
        return (
            tok.reshape(b, gh, gw, c, m, m)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(b, c, gh * m, gw * m)
        )

    def decode(self, z):
        return dictdot(dict(sample=self.decoder(z)))

    def forward(self, x, return_recon=True, return_vlm=False):
        """``(posterior, z, recon)``, plus ``VlmTokens`` when ``return_vlm``.

        The understanding branch has to be reached from this return value rather
        than by calling ``vit_forward`` separately: DDP discovers which parameters
        participate by walking the autograd graph back from what ``forward``
        returns, so a loss built on tensors it never saw would leave the tower's
        gradients unsynchronised across ranks.
        """
        out = self.encode(x, return_vlm=return_vlm)
        posterior, vlm = out if return_vlm else (out, None)
        z = posterior.sample()
        recon = self.decode(z).sample if return_recon else None
        return (posterior, z, recon, vlm) if return_vlm else (posterior, z, recon)

    # -- for the caption-NLL understanding loss -----------------------------
    def latent_to_llm_embeds(self, z, read_out):
        """Latent -> the token embeddings Qwen's language model expects.

        ``read_out`` maps a latent block back to the tower's ``vit_dim``; the
        tower's own (frozen) merger then produces ``llm_dim`` tokens. Routing the
        NLL through the real merger is what makes this objective a question about
        the *bottleneck* rather than about a projector's ability to invent an
        embedding space from scratch.
        """
        b, c, hh, ww = z.shape
        m = self.merge
        gh, gw = hh // m, ww // m
        tok = z.reshape(b, c, gh, m, gw, m).permute(0, 2, 4, 1, 3, 5).reshape(b, gh, gw, c * m * m)
        feats = read_out(tok)
        packed = self._from_raster(feats).reshape(b * gh * gw, self.vit_dim)
        merged = self.visual.merger(packed.to(self.visual.merger.linear_fc1.weight.dtype))
        return merged.reshape(b, -1, self.llm_dim)


def load_qwenvit_vae(ckpt_path, decoder_grid, fallback=None, freeze_encoder=True,
                     device=None):
    """Rebuild the autoencoder a checkpoint was trained as.

    Two checkpoint formats reach the evaluation scripts. ``train_e2e.py`` writes
    the whole argparse namespace under ``args``, so the geometry can be recovered
    exactly; ``save_vae_weights.py`` writes a bare state dict, which carries no
    geometry at all. For the second one the caller has to supply ``fallback``,
    and getting it wrong is not silent -- a mis-shaped tensor fails to load.

    Returns ``(vae, targs)``, ``targs`` being the training args when there were
    any and ``{}`` otherwise.
    """
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    targs = obj.get("args", {}) if isinstance(obj, dict) else {}
    if not isinstance(targs, dict):
        targs = vars(targs)
    sd = obj["vae"] if isinstance(obj, dict) and "vae" in obj else obj
    fallback = fallback or {}

    vit_path = targs.get("vae_vit_path") or fallback.get("vae_vit_path")
    if not vit_path:
        # Without this the tower path reaches from_pretrained as None and the
        # error surfaces as a 401 from huggingface.co/None.
        raise ValueError(
            f"{ckpt_path} carries no vae_vit_path in its saved args, so it is an "
            f"extracted VAE rather than a train_e2e.py checkpoint. Either point at "
            f"a training checkpoint or pass the tower path and decoder geometry "
            f"explicitly."
        )
    vae = QwenViTAutoencoder(
        model_path=vit_path,
        latent_channels=int(targs.get("latent_channels",
                                      fallback.get("latent_channels", 16))),
        decoder_width=int(targs.get("vae_decoder_width",
                                    fallback.get("decoder_width", 1024))),
        decoder_depth=int(targs.get("vae_decoder_depth",
                                    fallback.get("decoder_depth", 12))),
        decoder_heads=int(targs.get("vae_decoder_heads",
                                    fallback.get("decoder_heads", 16))),
        decoder_grid=decoder_grid,
        conv_refine=bool(targs.get("vae_decoder_conv_refine",
                                   fallback.get("conv_refine", True))),
        freeze_encoder=freeze_encoder,
    )
    if device is not None:
        vae = vae.to(device)
    missing, unexpected = vae.load_state_dict(sd, strict=False)

    # strict=False is needed because a training checkpoint carries tensors this
    # module does not own (the EMA copy, the discriminator), but a *missing*
    # weight is fatal: load_state_dict would leave that part of the network at
    # its init -- the pretrained tower, or a random decoder -- and the number
    # would be reported as if it came from the checkpoint.
    if missing:
        raise RuntimeError(
            f"{len(missing)} tensors missing from {ckpt_path} (e.g. {missing[:4]}); "
            f"this checkpoint does not hold the weights being evaluated.")
    step = obj.get("steps", "?") if isinstance(obj, dict) else "?"
    print(f">>> qwenvit from {ckpt_path} (step {step}, "
          f"missing={len(missing)} unexpected={len(unexpected)})", flush=True)
    return vae, targs
