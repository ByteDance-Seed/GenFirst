# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

"""Cola text VAE adapted to the E2E-DM autoencoder interface.

Maps token ids to one continuous latent vector per token and back to vocabulary
logits. Two properties differ from the image VAE and shape how this is used:

* The latent needs no scale/shift. The encoder applies a non-affine LayerNorm to
  the posterior mean, so every latent vector is already zero-mean / unit-variance
  across its 16 channels. ``latents_scale``/``latents_bias`` exist only so the
  two modalities present the same interface to the joint DiT.
* Sequences are kept in the "NA" flatten-concat layout (all samples concatenated
  along one axis, boundaries carried in ``txt_shape``) rather than padded to a
  common length, because caption lengths are strongly bimodal.
"""

import math
import os

import torch
import torch.nn.functional as F
from torch import nn

from models.cola_text_vae import ColaTextVAEConfig, ColaTextVAEModel


class TextPosterior:
    """Diagonal Gaussian over an NA-flattened token sequence.

    Mirrors ``models.autoencoder.DiagonalGaussianDistribution`` but splits
    mean/logvar on the channel axis (-1) instead of dim 1, since text latents are
    ``(L_total, latent_dim)`` rather than ``NCHW``.
    """

    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self):
        return self.mean + self.std * torch.randn_like(self.mean)

    def mode(self):
        return self.mean

    def kl(self):
        """Per-token KL to N(0, I), *summed* over channels. Shape ``(L_total,)``.

        Channel reduction matches the image posterior in ``models.autoencoder``,
        which sums KL over all latent elements.
        """
        if self.deterministic:
            return torch.zeros(self.mean.shape[0], device=self.mean.device)
        return 0.5 * torch.sum(
            self.mean.pow(2) + self.var - 1.0 - self.logvar, dim=-1
        )

    def entropy(self, weight=None):
        """Per-token differential entropy, *averaged* over channels.

        The mean (rather than sum) over channels mirrors the image posterior, so
        an entropy weight tuned on the image branch transfers unchanged. Only
        ``logvar`` carries gradient; maximizing this widens the posterior and
        resists collapse toward low-variance, easy-to-denoise codes.

        ``weight`` exists for interface parity with the image branch, where it
        carries the per-channel ``latents_scale`` from the DiT BN. Text latents
        are normalized architecturally with a uniform scale of 1.0, so passing it
        is a no-op here.
        """
        if self.deterministic:
            return torch.zeros(self.mean.shape[0], device=self.mean.device)
        term = 1.0 + math.log(2.0 * math.pi) + self.logvar
        if weight is not None:
            term = weight.detach().to(term.dtype).reshape(1, -1) * term
        return 0.5 * torch.mean(term, dim=-1)


class ColaTextVAE(nn.Module):
    """Frozen-or-trainable text VAE over token ids.

    Args:
        pretrained_path: directory holding ``config.json`` +
            ``model.safetensors`` for ``ColaTextVAEModel``. When ``None`` the
            architecture is built from config alone (resume path).
        config_path: config directory when it differs from ``pretrained_path``.
        train_trunk: train the 4+4 transformer blocks and the patch/latent
            projections (~194M params).
        train_embeddings: train ``encoder.wte`` and ``decoder.final_layer``
            (~308M params, 61% of the model). Off by default: the token
            inventory is unchanged, and training them costs a large amount of
            optimizer state while adding a collapse surface.
    """

    # Latent normalization is architectural (non-affine LayerNorm on the mean),
    # so these are exact rather than measured statistics.
    latents_scale = 1.0
    latents_bias = 0.0

    def __init__(
        self,
        pretrained_path=None,
        config_path=None,
        train_trunk=True,
        train_embeddings=False,
    ):
        super().__init__()
        cfg_src = config_path or pretrained_path
        if cfg_src is None:
            raise ValueError("ColaTextVAE needs pretrained_path or config_path.")
        self.config = ColaTextVAEConfig.from_pretrained(cfg_src)

        if pretrained_path is not None and os.path.isfile(
            os.path.join(pretrained_path, "model.safetensors")
        ):
            self.vae = ColaTextVAEModel.from_pretrained(
                pretrained_path, config=self.config
            )
        else:
            self.vae = ColaTextVAEModel(self.config)

        self.latent_dim = self.config.latent_dim
        self.vocab_size = self.config.vocab_size
        self.patch_size = self.config.patch_size
        self.mask_token_id = self.config.vocab_size  # wte has vocab_size + 1 rows

        self.set_trainable(trunk=train_trunk, embeddings=train_embeddings)

    # ------------------------------------------------------------------
    # Freezing
    # ------------------------------------------------------------------

    def set_trainable(self, trunk=True, embeddings=False):
        for p in self.vae.parameters():
            p.requires_grad_(False)
        if trunk:
            for name, p in self.vae.named_parameters():
                if not self._is_embedding_param(name):
                    p.requires_grad_(True)
        if embeddings:
            self.vae.encoder.wte.weight.requires_grad_(True)
            for p in self.vae.decoder.final_layer.parameters():
                p.requires_grad_(True)
        self.train_trunk, self.train_embeddings = trunk, embeddings

    @staticmethod
    def _is_embedding_param(name):
        return name.startswith("encoder.wte") or name.startswith("decoder.final_layer")

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def param_summary(self):
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return dict(total=total, trainable=train, frozen=total - train)

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def txt_shape(self, input_ids_list):
        """Per-sample latent lengths as the ``(B, 1)`` LongTensor Cola expects."""
        device = input_ids_list[0].device
        lens = [ids.shape[0] // self.patch_size for ids in input_ids_list]
        return torch.tensor([[n] for n in lens], dtype=torch.long, device=device)

    def encode(self, input_ids_list):
        """Encode a list of 1-D id tensors.

        Returns ``(posterior, txt_shape)`` where the posterior covers all samples
        concatenated along the token axis: ``(sum(n_i), latent_dim)``.
        """
        out = self.vae.encode(input_ids_list)
        params = torch.cat(
            [d.parameters for d in out.latent_dists], dim=0
        )  # (L_total, 2 * latent_dim)
        return TextPosterior(params), self.txt_shape(input_ids_list)

    def decode(self, z, txt_shape, txt_q_shape=None):
        """Decode NA-flattened latents to logits ``(L_total * patch_size, vocab)``."""
        if txt_q_shape is None:
            txt_q_shape = txt_shape
        logits = self.vae.decode(z, txt_shape=txt_shape, txt_q_shape=txt_q_shape)
        return logits.squeeze(0)

    def forward(self, input_ids_list, sample_posterior=True):
        """Encode and decode in one call, returning plain tensors.

        Only ``forward`` is intercepted once this module is wrapped for
        distributed training, so the full encode-decode path lives here and the
        posterior is rebuilt from ``params`` by the caller.

        Returns ``(params, z, logits, txt_shape)``, where ``params`` is the
        concatenated mean/logvar that :class:`TextPosterior` wraps.
        """
        posterior, txt_shape = self.encode(input_ids_list)
        z = posterior.sample() if sample_posterior else posterior.mode()
        logits = self.decode(z, txt_shape)
        return posterior.parameters, z, logits, txt_shape

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------

    @staticmethod
    def recon_loss(logits, input_ids_list, mask=None):
        """Mean per-token cross-entropy plus token accuracy.

        ``logits`` is the NA-flattened ``(L_total, vocab)`` decoder output, whose
        row order matches ``torch.cat(input_ids_list)``.

        ``mask`` is a ``(B,)`` bool keeping whole samples. Its tokens are zeroed
        while the denominator stays the batch's full token count, the same
        convention the entropy term uses, so an all-selected mask reproduces the
        unmasked loss exactly.
        """
        targets = torch.cat(input_ids_list)
        if mask is None:
            ce = F.cross_entropy(logits.float(), targets, reduction="mean")
            with torch.no_grad():
                acc = (logits.argmax(-1) == targets).float().mean()
            return ce, acc

        sizes = [t.numel() for t in input_ids_list]
        tok_keep = torch.repeat_interleave(
            mask.to(logits.device),
            torch.tensor(sizes, device=logits.device),
        )
        per_token = F.cross_entropy(logits.float(), targets, reduction="none")
        ce = (per_token * tok_keep).sum() / per_token.numel()
        with torch.no_grad():
            hit = (logits.argmax(-1) == targets).float() * tok_keep
            acc = hit.sum() / tok_keep.sum().clamp_min(1.0)
        return ce, acc


def na_to_padded(z_na, txt_shape, length):
    """Scatter NA-flattened latents ``(L_total, C)`` into ``(B, length, C)``.

    The text VAE works in NA form (samples concatenated, no padding) while the
    joint DiT wants a dense grid, so one side has to convert. Positions past a
    sample's own length are left at zero and are excluded by ``text_valid``.
    """
    sizes = txt_shape.flatten().tolist()
    out = z_na.new_zeros(len(sizes), length, z_na.shape[-1])
    for i, part in enumerate(z_na.split(sizes)):
        out[i, : part.shape[0]] = part
    return out


def padded_to_na(z_padded, txt_shape):
    """Inverse of :func:`na_to_padded`."""
    sizes = txt_shape.flatten().tolist()
    return torch.cat([z_padded[i, :n] for i, n in enumerate(sizes)], dim=0)


def _per_sample(per_token, txt_shape, reduce, mask=None):
    """Reduce NA per-token values to one value per sample, then average.

    ``reduce="sum"`` / ``"mean"`` selects the within-sample reduction so callers
    can reproduce the image branch, which sums KL but averages entropy over
    latent elements before averaging across the batch.

    ``mask`` zeroes samples before the batch average without shrinking the
    denominator, so masking half the batch halves the term. See
    ``loss.losses._latent_reg_term`` for why that is the reduction the entropy
    regularizer needs. An all-selected batch matches the unmasked result.
    """
    parts = per_token.split(txt_shape.flatten().tolist())
    if reduce == "sum":
        vals = torch.stack([p.sum() for p in parts])
    else:
        vals = torch.stack([p.mean() for p in parts])
    if mask is not None:
        vals = vals * mask.to(vals.dtype)
    return vals.mean()


class TextVAELoss(nn.Module):
    """Text-branch loss, mirroring the image branch term for term.

        total = recon_w * CE  +  kl_w * KL  -  entropy_w * H(q)  +  prior_w * L_prior

    Reductions follow ``loss/losses.py``: KL is summed over a sample's latent
    elements, entropy is averaged over them, and both are then averaged over the
    batch -- entropy over its ``entropy_mask`` subset, if one is given. Entropy is
    *maximized*, so it enters with a negative sign; the reported ``entropy_loss``
    is ``-H(q)`` so its weight stays positive.

    ``L_prior`` is the text-latent flow-matching loss of the joint DiT; it is
    computed by the training script and passed in rather than computed here.
    """

    def __init__(self, recon_weight=1.0, kl_weight=1e-6, entropy_weight=0.1, prior_weight=0.1):
        super().__init__()
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.entropy_weight = entropy_weight
        self.prior_weight = prior_weight

    def forward(self, logits, input_ids_list, posterior, txt_shape, prior_loss=None,
                entropy_mask=None, recon_mask=None):
        """``entropy_mask`` is a ``(B,)`` bool selecting the samples whose text
        latent the diffusion model *generates* (i2t). Entropy is the counterweight
        to that model's pull toward low-variance, easy-to-denoise codes, so a
        sample whose text is only a clean condition (t2i) must not receive it:
        there the pressure is unopposed and the posterior only inflates -- the
        sigma drift of 0.053 -> 0.51 seen in the first joint run.

        ``recon_mask`` additionally restricts CE and KL, which by default stay
        over the whole batch because the VAE reconstructs every caption whatever
        direction it was used in. Passing ``~is_t2i`` here is the stricter recipe
        where a caption serving as a condition trains nothing at all: no CE, no
        KL, no entropy, and no diffusion gradient either.
        """
        ce, acc = ColaTextVAE.recon_loss(logits, input_ids_list, mask=recon_mask)
        kl = _per_sample(posterior.kl(), txt_shape, "sum", mask=recon_mask)
        ent = _per_sample(posterior.entropy(), txt_shape, "mean", mask=entropy_mask)

        total = self.recon_weight * ce + self.kl_weight * kl - self.entropy_weight * ent
        if prior_loss is not None:
            total = total + self.prior_weight * prior_loss

        # CE is reported per *kept* token so its curve stays comparable to a run
        # with no mask; the loss itself keeps the full token count as denominator.
        ce_rep = ce
        if recon_mask is not None:
            lens = txt_shape.flatten().to(ce.dtype)
            share = (lens * recon_mask.to(ce.dtype)).sum() / lens.sum().clamp_min(1)
            ce_rep = ce / share.clamp_min(1e-8)

        log = dict(
            total_loss=total.detach(),
            recon_ce=ce_rep.detach(),
            token_acc=acc.detach(),
            kl_value=kl.detach(),
            entropy_value=ent.detach(),
            weighted_kl=(self.kl_weight * kl).detach(),
            weighted_entropy=(-self.entropy_weight * ent).detach(),
            # Diagnostics for latent drift under the end-to-end diffusion
            # gradient: sigma and per-dim logSNR both move well before token
            # accuracy does.
            posterior_sigma=posterior.std.mean().detach(),
            logsnr=(
                posterior.mean.var(0) / posterior.var.mean(0)
            ).clamp_min(1e-12).log10().mean().detach(),
        )
        if prior_loss is not None:
            log["prior_loss"] = prior_loss.detach()
            log["weighted_prior"] = (self.prior_weight * prior_loss).detach()
        return total, log
