# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

"""Joint image+text diffusion with an end-to-end-trained text VAE.

Setup:

* The image branch is selected by ``--image-vae-mode``:

  ``frozen``
      Weights and latent statistics come from a previous end-to-end T2I run and
      stay fixed. Only the encoder runs, under ``no_grad``, so the image branch
      contributes no reconstruction, LPIPS, or GAN loss.
  ``e2e-recon`` / ``e2e-gan``
      The image VAE trains too. The decoder runs and is constrained by
      L1 + LPIPS + entropy (plus a discriminator in ``e2e-gan``), reusing
      ``configs/l1_lpips_entropy_gan.yaml`` from the image-only runs. Roughly
      doubles step time and peak memory.

* The text VAE is **trainable** unless ``--freeze-text-vae``. Its trunk is
  optimized; the two 100k-row embedding tables stay frozen, since the token
  inventory is unchanged and they account for 61% of the parameters.
* One block-causal joint MMDiT models both modalities. Each *sample* is assigned
  a direction with probability ``--t2i-prob``, so both appear in every step:
  ``t2i`` denoises the image latent conditioned on clean text, ``i2t`` denoises
  text under the 2L layout conditioned on the clean image latent.

Text-branch loss, matching the image branch term for term::

    total = recon_w * CE + kl_w * KL - entropy_w * H(q) + prior_w * L_prior

As in ``train_e2e.py``, ``L_prior`` reaches the DiT at weight 1 (``prior_w=1``)
and reaches each encoder only through the latent it feeds the DiT, scaled by
``--vae-diffusion-coeff`` for text and ``--image-vae-diffusion-coeff`` for
images. Each coefficient is therefore the weight this loss carries on that VAE,
set equal to the entropy weight opposing it -- the pairing the image-only recipe
holds at every stage. The clean half of the 2L sequence is detached inside the
model so the history path cannot bypass the coefficient.

REPA alignment runs on the image stream, again as in that recipe: DINOv2 features
of the t2i samples are matched against the DiT's projections at
``--encoder-depth``, weighted ``--proj-coeff`` for the DiT itself, and a second
DiT-frozen forward carries ``--vae-align-proj-coeff`` into the image encoder
alone. That second term is the largest in the VAE's loss and is what keeps the
latent space semantically anchored while the diffusion gradient reshapes it. Only
t2i samples have an image being generated, so both terms are summed over that
subset and divided by the full batch, the normalization the diffusion loss uses
-- a coefficient then means per sample what it means in the image-only recipe.

``--detach-condition`` is the conservative fallback: a modality that only
conditions is treated as fixed data, contributing no reconstruction, no
regularizer and no diffusion gradient, so each VAE trains only on the samples
whose modality the DiT generates. That leaves the image encoder seeing exactly
what it sees in the image-only recipe, at the cost of half the effective batch
per VAE and of a latent space no longer shaped by its conditioning role.
"""

import argparse
import copy
import json
import logging
import os

import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from tqdm.auto import tqdm

from dataset import CaptionCollator, ParquetT2IDataset
from loss.losses import ReconstructionLoss_Single_Stage
from models.autoencoder import FluxVAE
from models.joint_mmdit import JointMMDiT_models
from models.text_vae import ColaTextVAE, TextPosterior, TextVAELoss, na_to_padded
from utils import count_trainable_params, load_encoders, preprocess_imgs_vae

logger = logging.getLogger(__name__)


class _GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


def grad_scale(x, scale):
    if scale == 1.0:
        return x
    return _GradScale.apply(x, scale)


def requires_grad(model, flag):
    for p in model.parameters():
        p.requires_grad_(flag)


class GradGuard:
    """Accepts or refuses one module's update, judging by its gradient norm.

    Clipping rescales a gradient without turning it, so a norm far outside the
    recent range still takes a full-size step along whatever pathological
    direction produced it. Every joint run so far died exactly that way: from a
    healthy state, one isolated image-VAE norm ~1000x the median (4672 against a
    median of 2.35 at step 2547; 3278 and 1957 in the two runs before it), which
    clipping applied at norm 1.0, image recon L1 stepping up in that same
    optimizer step, and an unrecoverable runaway within a dozen steps. The
    non-finite check fired far too late, and the same story repeated with and
    without REPA, so the trigger is not what matters -- being able to refuse the
    step is. fp16 gets this for free from GradScaler, which skips the step when
    the scaled gradient overflows; bf16 has no scaler, hence this.

    The reference is an EMA over accepted norms only, so a refused spike cannot
    raise the bar for the next one. `spike_factor` sits far above anything seen
    in a healthy stretch (those reach ~30x the median) and far below the norms
    that killed these runs.
    """

    def __init__(self, spike_factor=100.0, warmup=50, decay=0.99):
        self.spike_factor = spike_factor
        self.warmup = warmup
        self.decay = decay
        self.ref = None
        self.n_ok = 0

    def check(self, norm):
        """``(ok, reason)`` for an already-reduced norm, updating the reference."""
        if norm is None:
            return True, None
        if not torch.isfinite(norm).all():
            return False, "non-finite"
        v = float(norm)
        if self.n_ok >= self.warmup and v > self.spike_factor * self.ref:
            return False, f"{v:.4g}, {v / self.ref:.0f}x the running norm {self.ref:.3g}"
        self.n_ok += 1
        self.ref = v if self.ref is None else self.decay * self.ref + (1 - self.decay) * v
        return True, None


def top_grad_params(named_params, k=3):
    """The largest per-parameter gradient norms, to place a refused step.

    Clipping has already scaled every gradient by the same factor, so the
    ranking still says which part of the module produced the norm.
    """
    scored = []
    for name, p in named_params:
        if p.grad is None:
            continue
        v = float(p.grad.detach().float().norm())
        scored.append((v if v == v else float("inf"), v, name))
    scored.sort(reverse=True)
    return ", ".join(f"{name} {v:.3g}" for _, v, name in scored[:k])


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = dict(ema_model.named_parameters())
    for name, param in model.named_parameters():
        ema_params[name].mul_(decay).add_(param.detach(), alpha=1 - decay)

    # The latent-normalization statistics live in buffers, so a drifting image
    # encoder would leave the EMA copy normalizing with stale numbers.
    ema_buffers = dict(ema_model.named_buffers())
    for name, buf in model.named_buffers():
        if buf.is_floating_point():
            ema_buffers[name].mul_(decay).add_(buf.detach(), alpha=1 - decay)
        else:
            ema_buffers[name].copy_(buf)


def preprocess_raw_image(x, enc_type, resolution=256):
    """Encoder-side preprocessing, kept identical to REPA/train_joint_frozen.py."""
    if "dinov2" in enc_type:
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x / 255.0)
        return F.interpolate(x, 224 * (resolution // 256), mode="bicubic")
    if "mocov3" in enc_type or "mae" in enc_type or "dinov1" in enc_type:
        return Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x / 255.0)
    raise NotImplementedError(enc_type)


def alignment_loss(zs, zs_tilde):
    """Per-sample negative cosine similarity, as in REPA/train_joint_frozen.py.

    Cosine over channels, mean over tokens, mean over encoders. The batch
    dimension survives so callers can weight samples.
    """
    total = 0.0
    for z, z_tilde in zip(zs, zs_tilde):
        z = F.normalize(z.float(), dim=-1)
        z_tilde = F.normalize(z_tilde.float(), dim=-1)
        total = total - (z * z_tilde).sum(dim=-1).mean(dim=-1)
    return total / max(len(zs), 1)


def projection_loss(zs, zs_tilde, is_t2i):
    """Alignment over the t2i subset, or ``None`` when the model has no heads.

    Only a t2i sample has an image the DiT generates, so the targets exist for
    that subset alone. When a batch draws none, the term is folded in with a zero
    coefficient rather than skipped: the heads have to stay in the graph or DDP
    reports them unused.

    The average is over that subset, so this is the cosine itself, comparable to
    the image-only run's curve. Callers scale it by the subset's share of the
    batch to get the normalization the coefficients assume -- see `align_w`.
    """
    if not zs_tilde:
        return None
    if zs is None:
        return 0.0 * torch.stack([z.float().mean() for z in zs_tilde]).sum()
    assert zs[0].shape[1] == zs_tilde[0].shape[1], (
        f"encoder gives {zs[0].shape[1]} tokens but the image stream has "
        f"{zs_tilde[0].shape[1]}"
    )
    return alignment_loss(zs, [z[is_t2i] for z in zs_tilde]).mean()


def build_image_vae(args, device, trainable):
    """Image VAE plus the per-channel latent statistics the DiT normalizes with.

    Two sources: an extracted end-to-end checkpoint (``--image-vae-ckpt``, which
    ships measured statistics alongside), or the pretrained FLUX folder
    (``--image-vae-pretrained``), whose only available statistics are the scalar
    ``shift``/``scale`` from its config. The latter is a seed, not a measurement,
    so it has to be refined online -- see ``--track-latent-stats``.
    """
    vae = FluxVAE(pretrained_path=args.image_vae_pretrained)
    if args.image_vae_ckpt:
        sd = torch.load(args.image_vae_ckpt, map_location="cpu")
        if isinstance(sd, dict) and "vae" in sd and "vae.encoder.conv_in.weight" not in sd:
            sd = sd["vae"]
        missing, unexpected = vae.load_state_dict(sd, strict=False)
        if missing or unexpected:
            logger.warning(f"image VAE load: missing={len(missing)} unexpected={len(unexpected)}")
    vae = vae.to(device)
    requires_grad(vae, trainable)
    vae.train() if trainable else vae.eval()

    if args.image_latents_stats:
        stats = torch.load(args.image_latents_stats, map_location="cpu")
        scale = stats["latents_scale"].squeeze().to(device)
        bias = stats["latents_bias"].squeeze().to(device)
    else:
        n = vae.embed_dim
        scale = torch.full((n,), float(FluxVAE.scaling_factor), device=device)
        bias = torch.full((n,), float(FluxVAE.shift_factor), device=device)
    return vae, scale, bias


def _norm_str(n):
    return "n/a" if n is None else f"{n.item():.4g}"


def build_image_vae_loss(args, device):
    """Image reconstruction loss, reusing the config of the image-only e2e runs."""
    cfg = OmegaConf.load(args.loss_cfg_path)
    cfg.losses.kl_weight = args.latent_reg_weight
    # 0, the launcher's default, is a no-op: no EMA buffers are registered and the
    # term is never computed. Raise it only against a discriminator that is
    # running away, since LeCam penalizes the real/fake logit gap -- that gap is
    # what saturated both hinges around step 127k of the first run and left the
    # generator with an unbounded -mean(logits_fake).
    cfg.losses.lecam_regularization_weight = args.lecam_weight
    if args.image_vae_mode == "e2e-recon":
        # Zero factor skips the discriminator forward in both branches.
        cfg.losses.discriminator_factor = 0.0
        cfg.losses.discriminator_weight = 0.0
    loss_fn = ReconstructionLoss_Single_Stage(cfg).to(device)
    if args.disc_pretrained_ckpt:
        # The image VAE warm-starts, so a randomly initialized discriminator
        # would be the weaker player and feed it noise for the first thousands of
        # steps -- `discriminator_start` is 0, i.e. it is active immediately.
        sd = torch.load(args.disc_pretrained_ckpt, map_location="cpu")
        if isinstance(sd, dict) and "discriminator" in sd:
            sd = sd["discriminator"]
        loss_fn.discriminator.load_state_dict(sd)
        logger.info(f"discriminator warm-started from {args.disc_pretrained_ckpt}")
    if args.image_vae_mode == "e2e-recon":
        # Without a discriminator backward the module cannot be wrapped for DDP
        # (its discriminator would never receive a gradient), and an unsynced
        # `logvar` would drift apart across ranks. Freeze it instead.
        requires_grad(loss_fn, False)
    return loss_fn


def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    device = accelerator.device
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    save_dir = os.path.join(args.output_dir, args.exp_name)
    if accelerator.is_main_process:
        os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)
        with open(os.path.join(save_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{save_dir}/log.txt")],
        )

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    assert args.resolution % 8 == 0
    latent_size = args.resolution // 8

    train_image_vae = args.image_vae_mode != "frozen"
    image_vae, latents_scale, latents_bias = build_image_vae(args, device, train_image_vae)
    vae_loss_fn = build_image_vae_loss(args, device) if train_image_vae else None

    text_vae = ColaTextVAE(
        pretrained_path=args.text_vae_path,
        train_trunk=not args.freeze_text_vae,
        train_embeddings=args.train_text_embeddings,
    ).to(device)
    text_crit = TextVAELoss(
        recon_weight=args.recon_weight,
        kl_weight=args.kl_weight,
        entropy_weight=args.entropy_weight,
        prior_weight=args.prior_weight,
    )

    # REPA alignment on the image stream, as in the image-only e2e recipe: the
    # DiT gets `--proj-coeff` and the image VAE gets `--vae-align-proj-coeff`
    # through a second, cheap forward. Without it the VAE's latent space has no
    # semantic anchor and only L1/LPIPS/GAN/entropy shape it.
    if args.use_repa:
        encoders, encoder_types, architectures = load_encoders(
            args.enc_type, device, args.resolution
        )
        z_dims = [enc.embed_dim for enc in encoders]
    else:
        encoders, encoder_types, architectures, z_dims = [], [], [], []
    for enc in encoders:
        requires_grad(enc, False)
        enc.eval()
    # The second pass exists to shape the VAE, so it is pointless once the VAE is
    # frozen; the DiT keeps its own alignment either way.
    align_vae = train_image_vae and args.use_repa and args.vae_align_proj_coeff > 0

    model = JointMMDiT_models[args.model](
        input_size=latent_size,
        image_channels=16,
        text_channels=text_vae.latent_dim,
        text_block_size=args.text_block_size,
        max_text_tokens=args.max_text_tokens,
        path_type=args.path_type,
        cond_dropout=args.cond_dropout,
        bn_momentum=args.bn_momentum,
        z_dims=z_dims,
        encoder_depth=args.encoder_depth,
        detach_condition=args.detach_condition,
    ).to(device)
    model.init_bn(latents_scale=latents_scale, latents_bias=latents_bias)

    # A drifting encoder, or statistics that were only seeded from FLUX's scalar
    # shift/scale, need the BatchNorm to keep measuring.
    model.track_latent_stats = args.track_latent_stats or train_image_vae
    if model.track_latent_stats and accelerator.use_distributed:
        # Otherwise each rank normalizes by its own 16-sample batch statistics.
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    ema = copy.deepcopy(model).to(device)
    requires_grad(ema, False)
    ema.eval()

    if accelerator.is_main_process:
        s = text_vae.param_summary()
        logger.info(f"joint MMDiT params: {count_trainable_params(model):,}")
        logger.info(
            f"text VAE params: total {s['total']:,} trainable {s['trainable']:,} "
            f"frozen {s['frozen']:,}"
        )
        logger.info(
            f"loss = {args.recon_weight}*CE + {args.kl_weight}*KL "
            f"- {args.entropy_weight}*H(q) + {args.prior_weight}*prior; "
            f"t2i_prob={args.t2i_prob}, cond_dropout={args.cond_dropout}"
        )
        # As in train_e2e.py, the diffusion loss reaches the DiT at weight 1 and
        # each VAE only through its gradient scale, so the coefficient below is
        # directly the weight that opposes that VAE's entropy term. Anything other
        # than prior_weight=1 multiplies into both coefficients and breaks the
        # pairing -- 0.1 once left the image encoder at 1e-3 against 0.01.
        logger.info(
            f"diffusion loss on the VAEs: text {args.prior_weight}*"
            f"{args.vae_diffusion_coeff} vs entropy {args.entropy_weight}, image "
            f"{args.prior_weight}*{args.image_vae_diffusion_coeff} vs entropy "
            f"{args.latent_reg_weight}"
        )
        logger.info(
            "H(q) is masked to the samples whose modality is generated: text "
            "entropy on i2t samples, image entropy on t2i ones. It counteracts "
            "the diffusion pull toward collapse, which a clean condition is not "
            "under."
        )
        if args.detach_condition:
            logger.info(
                "detach-condition: a conditioning modality is fully detached -- "
                "no CE/L1/LPIPS/GAN, no KL, no entropy and no diffusion gradient. "
                "Each VAE therefore trains on ~t2i_prob of the batch."
            )
        logger.info(
            f"grad guard: refuse a module's step at {args.spike_factor}x its "
            f"running norm (armed after {args.guard_warmup} accepted steps), "
            f"per module, clip at {args.max_grad_norm}"
        )
        logger.info(
            f"REPA: on={args.use_repa} enc={args.enc_type} z_dims={z_dims} "
            f"layer={args.encoder_depth} dit_coeff={args.proj_coeff} "
            f"vae_coeff={args.vae_align_proj_coeff} (vae pass on={align_vae}, "
            "t2i samples only)"
        )
        logger.info(
            f"image VAE: {args.image_vae_mode}, trainable params "
            f"{count_trainable_params(image_vae):,}, "
            f"latent_reg_weight={args.latent_reg_weight}, "
            f"image_vae_diffusion_coeff={args.image_vae_diffusion_coeff}, "
            f"track_latent_stats={model.track_latent_stats} (bn_momentum={args.bn_momentum})"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,
    )
    text_params = text_vae.trainable_parameters()
    optimizer_txt = (
        torch.optim.AdamW(
            text_params, lr=args.text_vae_lr,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=0.0, eps=args.adam_epsilon,
        )
        if text_params
        else None
    )
    optimizer_vae = optimizer_disc = None
    if train_image_vae:
        optimizer_vae = torch.optim.AdamW(
            image_vae.parameters(), lr=args.vae_learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,
        )
        if args.image_vae_mode == "e2e-gan":
            # The whole loss module, as in train_e2e.py: besides the
            # discriminator it holds the learnable `logvar` that scales the
            # reconstruction term.
            optimizer_disc = torch.optim.AdamW(
                vae_loss_fn.parameters(), lr=args.disc_learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,
            )

    dataset = ParquetT2IDataset(
        [r for r in args.data_dir.split(",") if r],
        resolution=args.resolution,
        caption_columns=[c for c in args.caption_columns.split(",") if c],
        rows_per_file=args.parquet_rows_per_file,
        random_caption=args.random_caption,
    )
    collator = CaptionCollator(
        args.tokenizer_path, block_size=args.text_block_size, max_tokens=args.max_text_tokens
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size // accelerator.num_processes),
        shuffle=True, num_workers=args.num_workers, pin_memory=True,
        drop_last=True, collate_fn=collator,
    )

    update_ema(ema, model, decay=0)
    model.train()

    global_step = 0
    n_skipped = 0
    # Skipping a step protects the weights, but a branch that has broken for good
    # keeps skipping forever, and the loss curves of the branches that still work
    # look fine. One such run burned 27k of 33k steps before anyone noticed, so
    # treat a sustained skip rate as the failure it is.
    n_recent_skipped = 0
    skip_window_start = 0
    guards = {
        label: GradGuard(spike_factor=args.spike_factor, warmup=args.guard_warmup)
        for label in ("model", "text", "vae", "disc")
    }
    # The scaler and the guard cover different failures, so fp16 needs both.
    # GradScaler refuses a step whose gradient is *non-finite* and then halves the
    # scale; about a finite gradient it has nothing to say. GradGuard refuses a
    # finite norm far outside the recent range. The first fp16 run died in the gap
    # between them: 22.7k steps healthy, then the text gradient norm ran
    # 0.23 -> 4.4 -> 48 -> 314 -> 885 over ~15 steps, every value finite, so the
    # scaler passed them all; clipping rescaled each to norm 1 without turning it,
    # and ~20 full-size steps along that direction collapsed the text posterior
    # (sigma 0.44 -> 0.013, CE to chance) before anything reached NaN.
    #
    # So route on this module's own norm: non-finite to the scaler, which has to
    # see it to lower the scale, finite to GradGuard. Deciding from the norm also
    # keeps the attribution exact, which `AcceleratedOptimizer.step_was_skipped`
    # cannot: it reports whether the *shared* scale dropped, so one module's
    # overflow marked every module skipped, and the logs blamed the wrong one.
    uses_scaler = accelerator.mixed_precision == "fp16"
    if accelerator.is_main_process:
        logger.info(
            f"mixed precision {accelerator.mixed_precision}: GradGuard refuses "
            "finite spikes"
            + (", GradScaler refuses overflows and lowers the scale"
               if uses_scaler else " (bf16 has no scaler)")
        )
    if args.resume_step > 0:
        ckpt = torch.load(
            f"{args.output_dir}/{args.exp_name}/checkpoints/{args.resume_step:07d}.pt",
            map_location="cpu",
        )
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        text_vae.load_state_dict(ckpt["text_vae"])
        optimizer.load_state_dict(ckpt["opt"])
        if optimizer_txt is not None and ckpt.get("opt_txt") is not None:
            optimizer_txt.load_state_dict(ckpt["opt_txt"])
        if train_image_vae and ckpt.get("image_vae") is not None:
            image_vae.load_state_dict(ckpt["image_vae"])
            if ckpt.get("opt_vae") is not None:
                optimizer_vae.load_state_dict(ckpt["opt_vae"])
        if optimizer_disc is not None and ckpt.get("vae_loss_fn") is not None:
            # Turning LeCam on registers two EMA buffers a checkpoint from a run
            # without it cannot supply; they start from zero, which is where a
            # fresh run starts them anyway.
            missing, unexpected = vae_loss_fn.load_state_dict(
                ckpt["vae_loss_fn"], strict=False
            )
            if missing or unexpected:
                logger.warning(
                    f"vae_loss_fn resume: missing={missing} unexpected={unexpected}"
                )
            if ckpt.get("opt_disc") is not None:
                optimizer_disc.load_state_dict(ckpt["opt_disc"])
        global_step = ckpt["steps"]

        # Optimizer state carries its own learning rates, so loading it would
        # reinstate the ones the checkpoint was written with and silently ignore
        # the command line. There is no scheduler here, so the CLI value is the
        # intended rate -- which matters when a resume is what changes it.
        for opt, lr in (
            (optimizer, args.learning_rate),
            (optimizer_txt, args.text_vae_lr),
            (optimizer_vae, args.vae_learning_rate),
            (optimizer_disc, args.disc_learning_rate),
        ):
            if opt is None:
                continue
            for group in opt.param_groups:
                group["lr"] = lr
        logger.info(
            f"resumed from step {global_step}, learning rates taken from the "
            f"command line: model {args.learning_rate}, text {args.text_vae_lr}, "
            f"image vae {args.vae_learning_rate}, disc {args.disc_learning_rate}"
        )

    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    if optimizer_txt is not None:
        # Wrapping a module with no trainable parameters would fail, so the text
        # VAE is only prepared when it is actually being trained.
        text_vae_p, optimizer_txt = accelerator.prepare(text_vae, optimizer_txt)
    else:
        text_vae_p = text_vae.eval()
    if train_image_vae:
        image_vae, optimizer_vae = accelerator.prepare(image_vae, optimizer_vae)
    if optimizer_disc is not None:
        vae_loss_fn, optimizer_disc = accelerator.prepare(vae_loss_fn, optimizer_disc)

    use_wandb = accelerator.is_main_process and args.report_to == "wandb"
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.exp_name, dir=save_dir,
                   config=vars(copy.deepcopy(args)), settings={"console": "off"})

    def save_checkpoint(step):
        state = {
            "model": accelerator.unwrap_model(model).state_dict(),
            "ema": ema.state_dict(),
            "text_vae": accelerator.unwrap_model(text_vae_p).state_dict(),
            "opt": optimizer.state_dict(),
            "opt_txt": optimizer_txt.state_dict() if optimizer_txt else None,
            "args": args,
            "steps": step,
        }
        if train_image_vae:
            state["image_vae"] = accelerator.unwrap_model(image_vae).state_dict()
            state["opt_vae"] = optimizer_vae.state_dict()
        if optimizer_disc is not None:
            state["vae_loss_fn"] = accelerator.unwrap_model(vae_loss_fn).state_dict()
            state["opt_disc"] = optimizer_disc.state_dict()
        torch.save(state, f"{save_dir}/checkpoints/{step:07d}.pt")

    progress = tqdm(range(args.max_train_steps), initial=global_step, desc="Steps",
                    disable=not accelerator.is_local_main_process)

    task_gen = torch.Generator(device="cpu")
    task_gen.manual_seed(
        (args.seed if args.seed is not None else 0) + accelerator.process_index
    )

    done = False
    for epoch in range(args.epochs):
        if done:
            break
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            text_ids = [t.to(device, non_blocking=True) for t in batch["text_ids"]]
            text_valid = batch["text_valid"].to(device, non_blocking=True)
            L_pad = text_valid.shape[1]
            # Direction per sample, so both share a step. The image VAE then
            # gets its diffusion gradient every step instead of every other one,
            # which is what a whole-batch direction used to cost it.
            is_t2i = (
                torch.rand(text_valid.shape[0], generator=task_gen) < args.t2i_prob
            ).to(device)

            processed = preprocess_imgs_vae(images)
            if not train_image_vae:
                with torch.no_grad(), accelerator.autocast():
                    z_img = image_vae.encode(processed).sample()

            # Alignment targets exist only where an image is being generated, so
            # the encoder runs on the t2i subset, as in the frozen stage. `None`
            # means this batch drew none and contributes no alignment.
            zs = None
            if args.use_repa and bool(is_t2i.any()):
                with torch.no_grad(), accelerator.autocast():
                    zs = []
                    for encoder, enc_type in zip(encoders, encoder_types):
                        raw = preprocess_raw_image(
                            images[is_t2i].float(), enc_type,
                            resolution=args.resolution,
                        )
                        z = encoder.forward_features(raw)
                        if "mocov3" in enc_type:
                            z = z[:, 1:]
                        if "dinov2" in enc_type:
                            z = z["x_norm_patchtokens"]
                        zs.append(z)

            accumulated = [model] + ([image_vae] if train_image_vae else [])
            with accelerator.accumulate(accumulated), accelerator.autocast():
                if train_image_vae:
                    # The module-level forward is the DDP-safe entry point:
                    # encode/decode sit behind the wrapper.
                    posterior_img, z_img_raw, recon = image_vae(processed)
                    z_img_dit = grad_scale(z_img_raw, args.image_vae_diffusion_coeff)
                    if args.detach_condition:
                        # An i2t sample's image is a condition, so its latent
                        # reaches the DiT with the encoder cut off behind it.
                        z_img_dit = torch.where(
                            is_t2i.view(-1, 1, 1, 1), z_img_dit, z_img_raw.detach()
                        )
                else:
                    posterior_img = recon = None
                    z_img_dit = z_img.detach()

                params, z_txt_na, logits, txt_shape = text_vae_p(text_ids)
                posterior = TextPosterior(params)

                # Only this path carries the diffusion gradient into the text
                # encoder; the 2L clean copy is detached inside the model.
                z_txt = na_to_padded(
                    grad_scale(z_txt_na, args.vae_diffusion_coeff), txt_shape, L_pad
                )

                # REPA-E's VAE-side alignment. The DiT is frozen for this pass
                # and the latent is *not* grad-scaled, so `vae_align_proj_coeff`
                # lands on the image encoder at full strength and on nothing
                # else -- the DiT's own alignment comes from the pass below at
                # `proj_coeff`. The text latent is detached: this term is an
                # image-latent anchor, and the reference has no text VAE in it.
                # The pass stops at `encoder_depth`, and every draw it made is
                # reused below so both passes see the same noise, as
                # train_e2e.py does with `time_input` / `noises`. Both run in
                # train mode, so the latent BatchNorm now updates its running
                # statistics twice per step -- the same thing happens in
                # train_e2e.py, and it only makes those statistics track a
                # drifting encoder at roughly twice the momentum.
                # The diffusion loss is averaged over the whole batch, with each
                # sample contributing through the modality it generates. The
                # alignment has to be normalized the same way or a t2i sample
                # would carry ~2x the coefficient it does in the image-only
                # recipe, where every sample is t2i and both terms average over
                # the same batch. This share turns the subset average into a
                # sum over the batch.
                align_w = is_t2i.float().mean()

                draws, vae_align_loss = {}, None
                if align_vae:
                    requires_grad(model, False)
                    align_out = model(
                        z_img_raw,
                        na_to_padded(z_txt_na.detach(), txt_shape, L_pad),
                        text_valid, is_t2i, align_only=True,
                    )
                    requires_grad(model, True)
                    vae_align_loss = (
                        args.vae_align_proj_coeff * align_w
                        * projection_loss(zs, align_out["zs"], is_t2i)
                    )
                    draws = {k: v for k, v in align_out.items() if k != "zs"}

                out = model(z_img_dit, z_txt, text_valid, is_t2i, **draws)
                proj_loss = projection_loss(zs, out["zs"], is_t2i)
                # Entropy goes only to the samples whose latent is being
                # generated, i.e. the ones actually under the diffusion model's
                # pull toward collapse. On a sample where the modality is the
                # clean condition that pull is absent, so the term would be
                # unopposed and inflate the posterior -- which is the sigma
                # ratchet the earlier runs died of.
                # `--detach-condition` goes further and drops CE and KL on the
                # conditioning side too, so the caption of a t2i sample is not
                # part of the text VAE's training in any way.
                cond_mask = ~is_t2i if args.detach_condition else None
                loss, log = text_crit(
                    logits, text_ids, posterior, txt_shape,
                    prior_loss=out["prior_loss"].mean(),
                    entropy_mask=~is_t2i, recon_mask=cond_mask,
                )

                vae_log = None
                if train_image_vae:
                    # Per-channel BN scale reweights the entropy regularizer so
                    # its gradient to `logvar` matches what the DiT actually
                    # sees, as in train_e2e.py. Detached: a constant here.
                    scale_perchan = (
                        accelerator.unwrap_model(model)
                        .extract_latents_stats()["latents_scale"]
                        .detach()
                    )
                    vae_loss, vae_log = vae_loss_fn(
                        processed, recon, posterior_img, global_step, "generator",
                        latents_scale=scale_perchan, entropy_mask=is_t2i,
                        sample_mask=is_t2i if args.detach_condition else None,
                    )
                    loss = loss + vae_loss.mean()

                # Both alignment terms are in the same backward as everything
                # else; which module each one reaches was fixed at forward time.
                if proj_loss is not None:
                    loss = loss + args.proj_coeff * align_w * proj_loss
                if vae_align_loss is not None:
                    loss = loss + vae_align_loss

                accelerator.backward(loss)
                grad_norm = grad_norm_txt = grad_norm_vae = None
                skip_update = False
                model_stepped = True
                vae_refused = False
                if accelerator.sync_gradients:
                    # One unscale covering every optimizer in this backward, then
                    # plain clipping per module. `accelerator.clip_grad_norm_`
                    # unscales *all* tracked optimizers, so calling it once per
                    # module would unscale the same gradients repeatedly and
                    # assert in fp16; this is the pattern train_e2e.py uses.
                    # Outside fp16 `unscale_gradients` is a no-op and the plain
                    # clip is what `clip_grad_norm_` was calling anyway, so bf16
                    # numerics are unchanged.
                    accelerator.unscale_gradients(
                        optimizer=[
                            o for o in (optimizer, optimizer_txt, optimizer_vae)
                            if o is not None
                        ]
                    )
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )
                    if text_params:
                        grad_norm_txt = torch.nn.utils.clip_grad_norm_(
                            text_params, args.max_grad_norm
                        )
                    if train_image_vae:
                        grad_norm_vae = torch.nn.utils.clip_grad_norm_(
                            image_vae.parameters(), args.max_grad_norm
                        )
                    # Judged per module: one bad image-VAE gradient used to freeze
                    # the DiT and the text encoder along with it, which cost 1400
                    # consecutive steps of learning in the last run. Every rank
                    # sees the same reduced norms and holds the same guard state,
                    # so they all reach the same verdict without communicating.
                    refused = []
                    for label, norm, opt, named in (
                        ("model", grad_norm, optimizer,
                         accelerator.unwrap_model(model).named_parameters()),
                        ("text", grad_norm_txt, optimizer_txt, None),
                        ("vae", grad_norm_vae, optimizer_vae,
                         image_vae.named_parameters() if train_image_vae else None),
                    ):
                        if opt is None:
                            continue
                        if norm is not None and not torch.isfinite(norm).all():
                            # Step anyway: the scaler refuses it internally, and
                            # only a call it sees lets it halve the scale. Its
                            # reference stays clean because GradGuard never
                            # records a non-finite norm.
                            if uses_scaler:
                                opt.step()
                            ok, why = False, "non-finite gradient"
                        else:
                            ok, why = guards[label].check(norm)
                            if ok:
                                opt.step()
                        if not ok:
                            # Read the per-parameter norms before zero_grad drops
                            # them: which part of the module produced the spike is
                            # the only clue to what triggers it.
                            detail = f" [{top_grad_params(named)}]" if named else ""
                            refused.append(f"{label} {why}{detail}")
                            if label == "model":
                                model_stepped = False
                            if label == "vae":
                                vae_refused = True
                        opt.zero_grad(set_to_none=True)

                    skip_update = bool(refused)
                    if skip_update:
                        n_skipped += 1
                        n_recent_skipped += 1
                        what = "; ".join(refused)
                        if vae_refused and optimizer_disc is not None:
                            what += "; discriminator held"
                        logger.warning(
                            f"step {global_step}: refused {what} (model="
                            f"{_norm_str(grad_norm)} text={_norm_str(grad_norm_txt)} "
                            f"vae={_norm_str(grad_norm_vae)}), "
                            f"{n_skipped} refused so far"
                        )

                disc_log = None
                if optimizer_disc is not None:
                    # Second backward, on the detached reconstruction: the
                    # discriminator sees the generator's output as data.
                    with accelerator.autocast():
                        d_loss, disc_log = vae_loss_fn(
                            processed, recon.detach(), posterior_img,
                            global_step, "discriminator",
                        )
                    accelerator.backward(d_loss.mean())
                    # The adversarial pair has to move together. In the fp16
                    # reference one GradScaler governs both optimizers, so an
                    # overflow skips generator and discriminator alike. Stepping
                    # the discriminator through a refused VAE update instead lets
                    # it win by default, and that is what turned a handful of
                    # spikes at step 1144 into a saturated hinge (disc_loss
                    # 1.0 -> 0.002, logit gap 5.5) by 1400: with the hinge
                    # satisfied on both sides, -mean(logits_fake) is unbounded, so
                    # the VAE gradient sat at 1e4-1e5 and every one of its updates
                    # was refused from then on.
                    disc_ok = not vae_refused
                    if disc_ok and accelerator.sync_gradients:
                        accelerator.unscale_gradients(optimizer=[optimizer_disc])
                        grad_norm_disc = torch.nn.utils.clip_grad_norm_(
                            vae_loss_fn.parameters(), args.max_grad_norm
                        )
                        disc_ok, why = guards["disc"].check(grad_norm_disc)
                        if not disc_ok:
                            logger.warning(
                                f"step {global_step}: refused discriminator {why} "
                                f"({_norm_str(grad_norm_disc)})"
                            )
                            if uses_scaler and not torch.isfinite(grad_norm_disc).all():
                                # As above: the scaler needs the call to lower the
                                # scale, and it refuses the step itself.
                                optimizer_disc.step()
                                disc_ok = False
                    if disc_ok:
                        # In fp16 the scaler still has the last word here, exactly
                        # as it does for the generator side.
                        optimizer_disc.step()
                    optimizer_disc.zero_grad(set_to_none=True)

                # Only when the DiT itself moved: an EMA that keeps averaging in
                # the same weights would otherwise drift toward them.
                if accelerator.sync_gradients and model_stepped:
                    update_ema(ema, accelerator.unwrap_model(model), decay=args.ema_decay)

            if accelerator.sync_gradients:
                progress.update(1)
                global_step += 1
                # Both directions are in every batch now, so each prior loss is
                # averaged over its own samples across all ranks. Their scales
                # differ by ~4x, and a single mean would hide either one moving.
                prior_all = accelerator.gather(out["prior_loss"].detach())
                t2i_all = accelerator.gather(is_t2i)
                logs = {
                    "loss": accelerator.gather(loss.detach()).mean().item(),
                    "t2i_frac": t2i_all.float().mean().item(),
                    # Baselines on the pretrained VAE: CE 0.002, acc 100%,
                    # sigma 0.05, logSNR +2.4. CE and sigma move well before
                    # token accuracy does, so they are the drift warning.
                    "text/recon_ce": accelerator.gather(log["recon_ce"]).mean().item(),
                    "text/token_acc": accelerator.gather(log["token_acc"]).mean().item(),
                    "text/posterior_sigma": accelerator.gather(log["posterior_sigma"]).mean().item(),
                    "text/logsnr": accelerator.gather(log["logsnr"]).mean().item(),
                    "text/kl_value": accelerator.gather(log["kl_value"]).mean().item(),
                    "text/entropy_value": accelerator.gather(log["entropy_value"]).mean().item(),
                    "text_len": float(L_pad),
                    "epoch": epoch,
                }
                for name, sel in (("t2i", t2i_all), ("i2t", ~t2i_all)):
                    if sel.any():
                        logs[f"prior_loss/{name}"] = prior_all[sel].mean().item()
                # One number for both terms: the two passes see identical values
                # and differ only in where the gradient goes. Every rank has to
                # reach this gather, including one whose batch drew no t2i sample
                # at all -- gathering under a rank-local condition mismatches the
                # collective and hangs the run. Undoing `align_w` turns each
                # rank's value back into a sum over its own t2i samples, so the
                # ratio is the cosine averaged over every t2i sample there is.
                if proj_loss is not None:
                    proj_sum = accelerator.gather(
                        proj_loss.detach() * align_w * is_t2i.shape[0]
                    ).sum()
                    if t2i_all.any():
                        logs["proj_loss"] = (proj_sum / t2i_all.sum()).item()
                if grad_norm is not None:
                    logs["grad_norm"] = accelerator.gather(grad_norm).mean().item()
                if grad_norm_txt is not None:
                    logs["grad_norm_txt"] = accelerator.gather(grad_norm_txt).mean().item()
                if grad_norm_vae is not None:
                    logs["grad_norm_vae"] = accelerator.gather(grad_norm_vae).mean().item()
                # A rising count is the early warning that the GAN is running
                # away, long before the reconstruction metrics show it.
                logs["skipped_updates"] = n_skipped
                window = global_step - skip_window_start
                if window >= args.skip_window:
                    rate = n_recent_skipped / window
                    logs["skip_rate"] = rate
                    if rate > args.max_skip_rate:
                        raise RuntimeError(
                            f"{n_recent_skipped}/{window} updates skipped for "
                            f"non-finite gradients (rate {rate:.0%} > "
                            f"{args.max_skip_rate:.0%}). A branch has diverged and "
                            "training it further is pointless: check "
                            "image/recon_l1 and image/posterior_sigma around the "
                            "first skipped step, lower --latent-reg-weight, and "
                            "resume from the last good checkpoint."
                        )
                    n_recent_skipped = 0
                    skip_window_start = global_step
                if vae_log is not None:
                    logs.update({
                        "image/recon_l1": accelerator.gather(vae_log["reconstruction_loss"]).mean().item(),
                        "image/lpips": accelerator.gather(vae_log["perceptual_loss"]).mean().item(),
                        "image/latent_reg": accelerator.gather(vae_log["latent_reg_value"]).mean().item(),
                        "image/aux_kl": accelerator.gather(vae_log["aux_kl_value"]).mean().item(),
                        "image/gan_loss": accelerator.gather(vae_log["gan_loss"]).mean().item(),
                        "image/posterior_sigma": posterior_img.std.mean().item(),
                        # Synced by SyncBN, so no gather needed.
                        "image/latents_scale": scale_perchan.float().mean().item(),
                    })
                if disc_log is not None:
                    logs.update({
                        "image/disc_loss": accelerator.gather(disc_log["discriminator_loss"]).mean().item(),
                        "image/logits_real": accelerator.gather(disc_log["logits_real"]).mean().item(),
                        "image/logits_fake": accelerator.gather(disc_log["logits_fake"]).mean().item(),
                    })
                progress.set_postfix(loss=logs["loss"], acc=logs["text/token_acc"])
                if accelerator.is_main_process:
                    if use_wandb:
                        wandb.log(logs, step=global_step)
                    if global_step % args.log_every == 0 or global_step == 1:
                        line = (
                            f"step {global_step:>7} loss {logs['loss']:.4f} "
                            f"prior t2i {logs.get('prior_loss/t2i', float('nan')):.4f} "
                            f"i2t {logs.get('prior_loss/i2t', float('nan')):.4f} | "
                            f"CE {logs['text/recon_ce']:.5f} acc {100*logs['text/token_acc']:.2f}% "
                            f"sigma {logs['text/posterior_sigma']:.4f} "
                            f"logSNR {logs['text/logsnr']:+.3f}"
                        )
                        if "proj_loss" in logs:
                            line += f" | align {logs['proj_loss']:+.4f}"
                        if vae_log is not None:
                            line += (
                                f" | img L1 {logs['image/recon_l1']:.4f} "
                                f"lpips {logs['image/lpips']:.4f} "
                                f"gan {logs['image/gan_loss']:+.3f} "
                                f"scale {logs['image/latents_scale']:.3f}"
                            )
                        logger.info(line)

            if (
                global_step % args.checkpointing_steps == 0
                and global_step > 0
                and accelerator.sync_gradients
                and accelerator.is_main_process
            ):
                save_checkpoint(global_step)
                logger.info(f"Saved checkpoint at step {global_step}")

            if global_step >= args.max_train_steps:
                done = True
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_checkpoint(global_step)
        logger.info(f"Saved final checkpoint at step {global_step}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-name", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="exps")
    p.add_argument("--report-to", type=str, default="wandb", choices=["wandb", "none"])
    p.add_argument("--wandb-project", type=str, default="genfirst")
    p.add_argument("--model", type=str, default="JointMMDiT-L/2", choices=list(JointMMDiT_models))

    p.add_argument("--data-dir", type=str, required=True, help="comma-separated parquet roots")
    p.add_argument("--caption-columns", type=str, default="caption,prompt,text",
                   help="Comma-separated candidate caption columns in the parquet shards.")
    p.add_argument("--random-caption", action="store_true")
    p.add_argument("--parquet-rows-per-file", type=int, default=128)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)

    p.add_argument("--image-vae-ckpt", type=str, default=None,
                   help="extracted e2e VAE checkpoint; omit to use --image-vae-pretrained alone")
    p.add_argument("--image-vae-pretrained", type=str, default=None,
                   help="pretrained FLUX `vae` folder (diffusers layout)")
    p.add_argument("--image-latents-stats", type=str, default=None,
                   help="measured per-channel latent stats; omit to seed from FLUX's "
                        "scalar shift/scale, which then needs --track-latent-stats")
    p.add_argument("--image-vae-mode", type=str, default="frozen",
                   choices=["frozen", "e2e-recon", "e2e-gan"],
                   help="frozen: encoder only, no image loss. e2e-*: train the VAE with "
                        "L1+LPIPS+entropy (+GAN), which roughly doubles step time and memory")
    p.add_argument("--loss-cfg-path", type=str, default="configs/l1_lpips_entropy_gan.yaml")
    p.add_argument("--disc-pretrained-ckpt", type=str, default=None,
                   help="warm-start the discriminator, so it is not the weaker player "
                        "against an already-trained VAE")
    p.add_argument("--lecam-weight", type=float, default=1e-3,
                   help="LeCam regularization on the discriminator logits; 0 disables "
                        "it, which lets the hinge saturate and the GAN run away")
    p.add_argument("--latent-reg-weight", type=float, default=0.01,
                   help="image posterior entropy weight (stage-2 value of the image-only runs)")
    p.add_argument("--image-vae-diffusion-coeff", type=float, default=0.01,
                   help="weight of the diffusion loss on the image encoder, "
                        "paired with --latent-reg-weight")
    # REPA alignment, at the image-only e2e recipe's values. `proj-coeff` is the
    # DiT's own term; `vae-align-proj-coeff` is the second, DiT-frozen pass whose
    # gradient goes to the image VAE alone.
    p.add_argument("--use-repa", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--enc-type", type=str, default="dinov2-vit-b")
    p.add_argument("--encoder-depth", type=int, default=8)
    p.add_argument("--proj-coeff", type=float, default=0.5)
    p.add_argument("--vae-align-proj-coeff", type=float, default=1.5)
    p.add_argument("--spike-factor", type=float, default=100.0,
                   help="refuse a module's update when its gradient norm exceeds "
                        "this multiple of its running norm. Healthy stretches "
                        "reach ~30x; the spikes that killed the earlier runs were "
                        "900-2000x")
    p.add_argument("--guard-warmup", type=int, default=50,
                   help="accepted updates to observe before the spike test arms")
    p.add_argument("--skip-window", type=int, default=2000,
                   help="steps per window for the refused-update rate check")
    p.add_argument("--max-skip-rate", type=float, default=0.25,
                   help="abort when more than this fraction of a window was skipped; "
                        "isolated spikes are normal, a sustained rate is divergence")
    p.add_argument("--bn-momentum", type=float, default=0.0)
    p.add_argument("--track-latent-stats", action="store_true",
                   help="keep measuring the latent statistics instead of using the "
                        "loaded ones; implied when the image VAE trains")
    p.add_argument("--text-vae-path", type=str, required=True)
    p.add_argument("--tokenizer-path", type=str, required=True)
    p.add_argument("--freeze-text-vae", action="store_true")
    p.add_argument("--train-text-embeddings", action="store_true",
                   help="also train wte and the decoder output projection (+308M params)")

    p.add_argument("--text-block-size", type=int, default=16)
    p.add_argument("--max-text-tokens", type=int, default=256)
    p.add_argument("--t2i-prob", type=float, default=0.5)
    p.add_argument("--cond-dropout", type=float, default=0.1,
                   help="probability of hiding the condition modality, which trains the "
                        "unconditional branch that classifier-free guidance needs")
    p.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])

    p.add_argument("--recon-weight", type=float, default=1.0)
    p.add_argument("--kl-weight", type=float, default=1e-6)
    p.add_argument("--entropy-weight", type=float, default=0.01,
                   help="text posterior entropy weight, paired with "
                        "--vae-diffusion-coeff")
    p.add_argument("--prior-weight", type=float, default=1.0,
                   help="weight of the diffusion loss, as in train_e2e.py where "
                        "the denoising loss enters the total at 1. Leave it at 1: "
                        "it multiplies into both diffusion coefficients below and "
                        "would break their pairing with the entropy weights")
    p.add_argument("--vae-diffusion-coeff", type=float, default=0.01,
                   help="weight of the diffusion loss on the text encoder, "
                        "paired with --entropy-weight")
    p.add_argument("--detach-condition", action="store_true",
                   help="train each VAE only on the samples whose modality the "
                        "DiT generates: an image in an i2t sample, or a caption in "
                        "a t2i sample, then contributes no reconstruction, "
                        "regularizer or diffusion gradient at all. Halves each "
                        "VAE's effective batch, and drops the conditioning "
                        "gradient that is the one path the image-only recipe "
                        "never had to carry")

    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--text-vae-lr", type=float, default=1e-5)
    p.add_argument("--vae-learning-rate", type=float, default=1e-4)
    p.add_argument("--disc-learning-rate", type=float, default=1e-4)
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.999)
    p.add_argument("--adam-weight-decay", type=float, default=0.0)
    p.add_argument("--adam-epsilon", type=float, default=1e-8)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.9999)

    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--max-train-steps", type=int, default=400000)
    p.add_argument("--checkpointing-steps", type=int, default=10000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--resume-step", type=int, default=0)
    p.add_argument("--mixed-precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--allow-tf32", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not args.image_vae_ckpt and not args.image_vae_pretrained:
        p.error("need --image-vae-ckpt or --image-vae-pretrained")
    if not args.image_latents_stats and not (
        args.track_latent_stats or args.image_vae_mode != "frozen"
    ):
        p.error(
            "without --image-latents-stats the statistics are only seeded from FLUX's "
            "scalar shift/scale, so --track-latent-stats is required to refine them"
        )
    if args.track_latent_stats or args.image_vae_mode != "frozen":
        if args.bn_momentum == 0.0:
            p.error("--bn-momentum must be > 0 when the latent statistics are tracked")
    return args


if __name__ == "__main__":
    main(parse_args())
