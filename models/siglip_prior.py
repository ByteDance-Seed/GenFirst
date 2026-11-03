# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

"""SigLIP-style understanding prior on continuous VAE latents.

Ported from a unified reconstruction+understanding autoencoder that trains a ViT
autoencoder under four terms: SigLIP sigmoid contrastive + caption-token
classification + L1 + LPIPS. This module supplies the two *understanding* terms;
reconstruction stays with the existing VAE loss.

Why this is preferred over the caption-NLL prior in models/understanding_prior.py:
NLL through a frozen VLM is dominated by the language prior, so the
image-dependent part (the PMI) is a small difference of two large numbers and its
gradient is mostly signal about English. Both terms here are discriminative and
put every bit of their gradient on image-text correspondence. The caption term
additionally reweights its targets by inverse document frequency, which drives
stopwords to ~zero weight explicitly rather than hoping they cancel.

Two deliberate differences from the reference implementation, both forced by this
setting:

*Where the image feature is tapped.* The reference runs the contrastive loss on a
pooled branch of the *ViT trunk* and derives the AE latent as a separate linear
projection of the same last-block tokens, so its latent inherits semantics
indirectly. A conv VAE has no trunk to share, so the pool sits on the latent
itself -- the very tensor the MMDiT consumes. That is a strictly stronger
requirement, and it is the point of the experiment.

*How cross-rank negatives are collected.* The reference trains both towers, so it
needs a gradient-aware ring exchange of text features. The text tower here is
frozen, which means text features carry no gradient and a plain ``all_gather``
suffices. The result is arithmetically identical to the ring: that loss is one
local term plus ``W-1`` negative-only terms, each divided by the local batch,
which partitions exactly the ``B_local x B_global`` matrix computed here.
"""

import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

# One source of truth for where CLIP lives; see models/text_encoder.py.
from models.text_encoder import DEFAULT_CLIP


def _world():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    return 1, 0


def _gather(t):
    """Concatenate ``t`` across ranks, this rank's slice at index ``rank``.

    No autograd: every caller passes features from the frozen text tower.
    """
    world, _ = _world()
    if world == 1:
        return t
    buf = [torch.empty_like(t) for _ in range(world)]
    dist.all_gather(buf, t.contiguous())
    return torch.cat(buf, dim=0)


# ---------------------------------------------------------------------------
# latent -> pooled feature
# ---------------------------------------------------------------------------
class _MAPHead(nn.Module):
    """Attention pooling: a single learnable probe cross-attends over tokens.

    Mirrors the reference multihead attention-pooling head (probe token, q/k/v
    projections, output projection, then a residual MLP);
    ``nn.MultiheadAttention`` supplies the projections that head spells out by
    hand.
    """

    def __init__(self, width, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.probe = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.trunc_normal_(self.probe, std=0.02)
        self.attn = nn.MultiheadAttention(width, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(width)
        hidden = int(width * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width))

    def forward(self, x):
        q = self.probe.expand(x.shape[0], -1, -1)
        out = self.attn(q, x, x, need_weights=False)[0].squeeze(1)
        return out + self.mlp(self.norm(out))


class _SelfAttnBlock(nn.Module):
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


class LatentPooler(nn.Module):
    """Patchified latent -> one pooled vector.

    ``mode`` is the capacity knob, and the reason it exists is that a falling
    contrastive loss is only evidence about the *latent* if the pool is too weak
    to have learned the task by itself:
      * ``mean``     : mean over tokens. Weakest; no way to select regions.
      * ``attnpool`` : one cross-attention probe. The default, and the direct
                       analogue of how the reference pools its trunk.
      * ``transformer``: self-attention blocks before the probe. For measuring
                       how much a *frozen* latent carries at all, i.e. the
                       ceiling of the whole approach -- not for training runs.
    """

    def __init__(self, in_dim, width=1024, num_heads=8, mode="attnpool", depth=4, grid=16):
        super().__init__()
        self.mode = mode
        self.stem = nn.Linear(in_dim, width)
        # Kept as a 2D map so it can be resampled: the 512px phase doubles the
        # resolution, which doubles the latent grid and hence the token count.
        self.pos = nn.Parameter(torch.zeros(1, width, grid, grid))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            [_SelfAttnBlock(width, num_heads) for _ in range(depth)] if mode == "transformer" else []
        )
        self.pool = None if mode == "mean" else _MAPHead(width, num_heads)
        self.final_norm = nn.LayerNorm(width)

    def _pos_tokens(self, h, w):
        pos = self.pos
        if pos.shape[-2] != h or pos.shape[-1] != w:
            pos = F.interpolate(pos, size=(h, w), mode="bicubic", align_corners=False)
        return pos.flatten(2).transpose(1, 2)

    def forward(self, x, grid_hw):
        """x: (B, N, in_dim) patch tokens; grid_hw: (h, w) of that token grid."""
        h = self.stem(x) + self._pos_tokens(*grid_hw)
        for blk in self.blocks:
            h = blk(h)
        pooled = h.mean(dim=1) if self.pool is None else self.pool(h)
        return self.final_norm(pooled)


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------
class SigLipLoss(nn.Module):
    """Sigmoid pairwise image-text loss.

        loss = -mean_i sum_j log sigmoid( L_ij * (t * f_i . g_j + b) )

    with ``L_ij = +1`` for the matched pair and ``-1`` otherwise, ``j`` ranging
    over the *global* batch of texts. Sum over ``j`` rather than mean is the
    published normalisation: it makes the loss magnitude grow with the negative
    count, so 256 global samples here sit ~54x below the reference
    implementation's 13,824. ``--und-coeff`` is what absorbs that.
    """

    def forward(self, img, txt, logit_scale, logit_bias):
        world, rank = _world()
        txt_all = _gather(txt)
        b = img.shape[0]

        sim = img @ txt_all.t()
        logits = logit_scale * sim + logit_bias
        labels = -torch.ones_like(logits)
        pos = torch.arange(b, device=img.device)
        labels[pos, rank * b + pos] = 1.0
        loss = -F.logsigmoid(labels * logits).sum() / b

        with torch.no_grad():
            pos_sim = sim[pos, rank * b + pos]
            neg_sum = sim.sum() - pos_sim.sum()
            neg_sim = neg_sum / max(sim.numel() - b, 1)
            # In-batch retrieval against every text on every rank. This is the
            # headline metric: it is bounded, interpretable against the
            # 1/(B*W) chance level, and it moves from the very first steps.
            r1 = (sim.argmax(dim=1) == rank * b + pos).float().mean()
        return loss, {
            "pos_sim": pos_sim.mean(),
            "neg_sim": neg_sim,
            "i2t_r1": r1,
            "num_negatives": b * world - 1,
        }


class CaptionClsLoss(nn.Module):
    """Caption-token classification with IDF reweighting.

    The pooled image feature predicts a bag of the caption's tokens. Targets are
    multi-hot, scaled by ``log(num_samples / cap_fq)`` from running token
    statistics, then L1-normalised into a distribution for a soft cross-entropy.

    Two properties earn it its place next to the contrastive term. It needs no
    negatives, so unlike SigLIP it does not weaken when the batch is small --
    which matters at 256 against the reference implementation's 13,824. And the
    IDF factor is an explicit answer to what sinks the caption-NLL prior: a token
    appearing in most captions gets a weight near zero, so the gradient is spent
    on content words instead of on syntax.
    """

    def __init__(self, vocab_size, ignore_ids=()):
        super().__init__()
        self.register_buffer("cap_fq", torch.zeros(1, vocab_size))
        self.register_buffer("num_samples", torch.zeros(1))
        self.register_buffer(
            "ignore_ids", torch.tensor(sorted(set(ignore_ids)), dtype=torch.long)
        )

    def forward(self, logits, token_ids):
        b, v = logits.shape
        targets = torch.zeros(b, v, dtype=torch.float32, device=logits.device)
        targets.scatter_(1, token_ids, 1.0)
        if self.ignore_ids.numel():
            # Padding/BOS/EOS are present in every caption, so "predict them" is
            # vacuous. IDF would shrink them anyway; dropping them keeps the
            # normalised target from spending mass on them at step 0.
            targets[:, self.ignore_ids] = 0.0

        world, _ = _world()
        self.cap_fq += targets.sum(dim=0, keepdim=True) / b
        self.num_samples += 1
        if world > 1:
            dist.all_reduce(self.cap_fq, op=dist.ReduceOp.AVG)
            dist.all_reduce(self.num_samples, op=dist.ReduceOp.AVG)
        floor = 1.0 / (world * b)
        idf = torch.log(
            (self.num_samples.clamp(min=1e-7) + floor) / (self.cap_fq.clamp(min=1e-7) + floor)
        )
        targets = targets * idf.to(targets.dtype)

        weights = F.normalize(targets, p=1, dim=1).clamp(1e-7, 1.0)
        return -(F.log_softmax(logits, dim=1) * weights).sum(dim=1).mean()


# ---------------------------------------------------------------------------
# the prior
# ---------------------------------------------------------------------------
class SiglipPrior(nn.Module):
    def __init__(
        self,
        tokenizer,
        latent_channels=16,
        latent_patch=2,
        head_mode="attnpool",
        width=1024,
        num_heads=8,
        depth=4,
        grid=16,
        embed_dim=768,
        cls_coeff=1.0,
        text_projection=True,
        clip_path=DEFAULT_CLIP,
        max_caption_tokens=77,
        logit_scale_max=100.0,
        device="cuda",
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_patch = latent_patch
        self.cls_coeff = cls_coeff
        self.max_caption_tokens = max_caption_tokens
        self.logit_scale_max = logit_scale_max
        self.tokenizer = tokenizer

        in_dim = latent_channels * latent_patch * latent_patch
        self.pooler = LatentPooler(
            in_dim=in_dim, width=width, num_heads=num_heads, mode=head_mode,
            depth=depth, grid=grid,
        )
        self.embed = nn.Linear(width, embed_dim)

        # Reference init values; logit_bias starting at -10 is what keeps the
        # sigmoid near zero for the overwhelmingly negative pair matrix, so the
        # loss at step 0 is ~10 (the positives it is missing) rather than being
        # swamped by the B*W negatives it already gets right.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.logit_bias = nn.Parameter(torch.tensor(-10.0))
        self.siglip_loss = SigLipLoss()

        vocab = len(tokenizer)
        if cls_coeff > 0:
            # Fed the *unnormalised* contrastive embedding, matching how the
            # reference feeds its cls head the pre-normalisation image features.
            self.cls_head = nn.Linear(embed_dim, vocab)
            ignore = {
                i for i in (
                    tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id,
                ) if i is not None
            }
            self.cls_loss = CaptionClsLoss(vocab, ignore_ids=ignore)
        else:
            self.cls_head = None
            self.cls_loss = None

        # CLIP's text_projection maps the text encoder's pooled state into the
        # shared image-text space. Taking just this matrix leaves the frozen
        # encoder in train_e2e.py untouched -- its `pooled` still conditions
        # the MMDiT exactly as before -- while the prior gets to train against
        # the real CLIP space, which is what makes ImageNet zero-shot with the
        # standard prompt templates a valid readout later.
        if text_projection:
            from transformers import CLIPTextModelWithProjection

            proj = CLIPTextModelWithProjection.from_pretrained(clip_path)
            w = proj.text_projection.weight.data.clone().float()
            del proj
            assert w.shape[0] == embed_dim, (
                f"CLIP text projection is {w.shape[0]}-d but embed_dim={embed_dim}"
            )
            self.register_buffer("text_proj_w", w)
        else:
            self.text_proj_w = None

        self.float()
        self.to(device)

    # -- parameters ---------------------------------------------------------
    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def param_groups(self, weight_decay):
        """Keep the temperature and bias out of weight decay.

        Decay pulls ``logit_scale`` toward 0, i.e. temperature toward 1, and at
        this batch size the term is small enough that the pull would dominate its
        own gradient.
        """
        no_decay = {"logit_scale", "logit_bias"}
        decay_p, nodecay_p = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (nodecay_p if n in no_decay else decay_p).append(p)
        return [
            {"params": decay_p, "weight_decay": weight_decay},
            {"params": nodecay_p, "weight_decay": 0.0},
        ]

    # -- inputs -------------------------------------------------------------
    def patchify(self, z):
        """(B, C, H, W) -> (B, N, C*p*p) plus the (h, w) token grid."""
        b, c, hh, ww = z.shape
        p = self.latent_patch
        assert c == self.latent_channels, f"latent has {c} channels, expected {self.latent_channels}"
        assert hh % p == 0 and ww % p == 0, f"latent {hh}x{ww} not divisible by patch {p}"
        h, w = hh // p, ww // p
        x = z.reshape(b, c, h, p, w, p).permute(0, 2, 4, 1, 3, 5).reshape(b, h * w, c * p * p)
        return x, (h, w)

    def tokenize(self, captions, device):
        """Caption token ids for the classification target.

        Re-tokenised here rather than threaded out of the text encoder: it is a
        few milliseconds of CPU per step against a ~1s step, and it keeps the
        encoder's public signature (and its three call sites) untouched.
        """
        tok = self.tokenizer(
            [c or "" for c in captions],
            padding="max_length",
            max_length=self.max_caption_tokens,
            truncation=True,
            return_tensors="pt",
        )
        return tok.input_ids.to(device)

    def text_embed(self, pooled):
        t = pooled.float()
        if self.text_proj_w is not None:
            t = t @ self.text_proj_w.t()
        return F.normalize(t, dim=-1)

    # -- forward ------------------------------------------------------------
    def forward(self, z, text_pooled, captions=None):
        """``z``: normalised latent (B, C, H, W). ``text_pooled``: (B, D_text)."""
        x, grid = self.patchify(z.float())
        feat = self.pooler(x, grid)
        emb = self.embed(feat)
        img = F.normalize(emb, dim=-1)
        txt = self.text_embed(text_pooled)

        scale = self.logit_scale.exp().clamp(max=self.logit_scale_max)
        siglip, diag = self.siglip_loss(img, txt, scale, self.logit_bias)

        loss = siglip
        out = {
            "loss": loss,
            "siglip": siglip.detach(),
            "logit_scale": scale.detach(),
            "logit_bias": self.logit_bias.detach(),
            **{k: (v.detach() if torch.is_tensor(v) else v) for k, v in diag.items()},
        }
        if self.cls_loss is not None and captions is not None:
            ids = self.tokenize(captions, z.device)
            cls = self.cls_loss(self.cls_head(emb), ids)
            out["loss"] = loss + self.cls_coeff * cls
            out["cls"] = cls.detach()
        return out
