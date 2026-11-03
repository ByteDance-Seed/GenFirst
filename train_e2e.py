# Copyright (c) 2025 Xingjian Leng and Jaskirat Singh
#
# From REPA-E (https://github.com/End2End-Diffusion/REPA-E), which this tree
# derives from. Licensed under the MIT license in LICENSE.

import argparse
import contextlib
import copy
import logging
import os
import json
import math
from pathlib import Path
from collections import OrderedDict

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from torchvision.utils import make_grid
from tqdm.auto import tqdm
from omegaconf import OmegaConf
import wandb

from dataset import CustomINH5Dataset, ParquetT2IDataset, ParquetClassDataset
from loss.losses import ReconstructionLoss_Single_Stage
from models.autoencoder import vae_models
from models.sit import SiT_models
from models.mmdit_e2e import MMDiT_models
from samplers import euler_sampler
from utils import load_encoders, normalize_latents, denormalize_latents, preprocess_imgs_vae, count_trainable_params, compute_latent_stats, reduce_latent_stats
from eval_utils import OnlineEvaluator, run_online_eval

logger = get_logger(__name__)

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)


def vae_encoder_params(vae):
    """The VAE's *pretrained* encoder parameters, if it has any.

    Only the Qwen3-VL tower qualifies: the conv VAEs are trained from scratch (or
    from a checkpoint of this same recipe), so there is no pretrained geometry to
    preserve and nothing to separate out. Returning an empty list for them keeps
    their optimizer a single group, exactly as before.
    """
    visual = getattr(vae, "visual", None)
    return list(visual.parameters()) if visual is not None else []


def preprocess_raw_image(x, enc_type):
    resolution = x.shape[-1]
    if 'clip' in enc_type:
        x = x / 255.
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    elif 'dinov1' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')

    return x


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z - latents_bias) * latents_scale # normalize
    return z 


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

    # Also perform EMA on BN buffers
    ema_buffers = OrderedDict(ema_model.named_buffers())
    model_buffers = OrderedDict(model.named_buffers())

    for name, buffer in model_buffers.items():
        name = name.replace("module.", "")
        if buffer.dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
            # Apply EMA only to float buffers
            ema_buffers[name].mul_(decay).add_(buffer.data, alpha=1 - decay)
        else:
            # Direct copy for non-float buffers
            ema_buffers[name].copy_(buffer)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


class _GradScale(torch.autograd.Function):
    """Identity in the forward pass; scales the back-propagated gradient.

    Used in the end-to-end path: we feed ``x = _GradScale.apply(z, coeff)`` to
    the SiT diffusion pass so the denoising gradient that reaches the VAE
    encoder (through ``z``) is multiplied by ``coeff``, while the SiT
    parameters still receive the unscaled denoising gradient (their gradient
    path does not go through ``z``).
    """

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = float(scale)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


def grad_scale(x, scale):
    if scale == 1.0:
        return x
    return _GradScale.apply(x, scale)


@torch.no_grad()
def mmdit_text_euler_sampler(model, xT, context, pooled, null_context, null_pooled,
                             num_steps=50, cfg_scale=4.0, path_type="linear"):
    """Minimal Euler sampler (v-prediction) for the text-conditional MMDiT."""
    x = xT
    n = x.shape[0]
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = torch.full((n,), 1.0 - i * dt, device=x.device, dtype=x.dtype)
        v_cond = model.inference(x, t, context, pooled)
        if cfg_scale != 1.0:
            v_uncond = model.inference(x, t, null_context, null_pooled)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = v_cond
        x = x - v * dt
    return x


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):    
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None if args.report_to == "none" else args.report_to,
        project_config=accelerator_project_config,
    )

    # Every wandb call is gated on this, so `--report-to none` runs without a
    # wandb account at all.
    use_wandb = accelerator.is_main_process and args.report_to == "wandb"

    # set up the logger and checkpoint dirs
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # Create model:
    if args.vae == "f8d4":
        assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
        latent_size = args.resolution // 8
        in_channels = 4
    elif args.vae == "f16d32":
        assert args.resolution % 16 == 0, "Image size must be divisible by 16 (for the VAE encoder)."
        latent_size = args.resolution // 16
        in_channels = 32
    elif args.vae == "flux":
        assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
        latent_size = args.resolution // 8
        in_channels = 16
    elif args.vae == "flux2":
        # FLUX.2 autoencoder: f8 with 32 latent channels. We keep the latent
        # unfolded (no 2x2 patch fold), so the token count matches the FLUX.1
        # runs and only the VAE differs.
        assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
        latent_size = args.resolution // 8
        in_channels = 32
    elif args.vae == "qwenvit":
        # Qwen3-VL's tower is patch-16, and each of its tokens is bottlenecked into
        # a 2x2 spatial block, so the latent grid is resolution/16*2 = resolution/8
        # -- the same geometry as FLUX. With --latent-channels 16 the latent is
        # shape-identical to FLUX's, which is what keeps the MMDiT, its BN and the
        # loss config unchanged.
        assert args.resolution % 32 == 0, (
            "Image size must be divisible by patch*merge = 32 for the Qwen3-VL tower."
        )
        latent_size = args.resolution // 8
        in_channels = args.latent_channels
    else:
        raise NotImplementedError()

    if args.enc_type != None:
        encoders, encoder_types, architectures = load_encoders(
            args.enc_type, device, args.resolution
        )
    else:
        raise NotImplementedError()
    z_dims = [encoder.embed_dim for encoder in encoders] if args.enc_type != 'None' else [0]

    text_encoder = None
    if args.conditioning == "text":
        from models.text_encoder import FrozenTextEncoder
        text_encoder = FrozenTextEncoder(
            encoder_type=args.text_encoder,
            model_name=args.text_encoder_name,
            max_length=args.text_max_length,
            device=device,
        )
        if accelerator.is_main_process:
            logger.info(
                f"Text encoder: {args.text_encoder} "
                f"(context_dim={text_encoder.context_dim}, pooled_dim={text_encoder.pooled_dim})"
            )

    if args.conditioning == "text":
        assert args.model in MMDiT_models, (
            f"--conditioning text requires an MMDiT model, got --model {args.model}"
        )
        model = MMDiT_models[args.model](
            input_size=latent_size,
            in_channels=in_channels,
            context_dim=text_encoder.context_dim,
            pooled_dim=text_encoder.pooled_dim,
            caption_dropout_prob=args.cfg_prob,
            z_dims=z_dims,
            encoder_depth=args.encoder_depth,
            bn_momentum=args.bn_momentum,
            path_type=args.path_type,
        )
    else:
        block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
        model = SiT_models[args.model](
            input_size=latent_size,
            in_channels=in_channels,
            num_classes=args.num_classes,
            class_dropout_prob=args.cfg_prob,
            z_dims=z_dims,
            encoder_depth=args.encoder_depth,
            bn_momentum=args.bn_momentum,
            **block_kwargs
        )

    # make a copy of the model for EMA
    model = model.to(device)
    ema = copy.deepcopy(model).to(device)  # Create an EMA of the model for use after training

    # Load VAE and create a EMA of the VAE
    if args.vae in ("flux", "flux2"):
        vae = vae_models[args.vae](pretrained_path=args.vae_pretrained_path).to(device)
        stats_path = (args.vae_ckpt or "").replace(".pt", "-latents-stats.pt")
        if args.vae_ckpt and os.path.isfile(args.vae_ckpt):
            if args.vae == "flux2":
                vae.load_pretrained(args.vae_ckpt)
            else:
                vae_ckpt = torch.load(args.vae_ckpt, map_location=device)
                if isinstance(vae_ckpt, dict) and "vae" in vae_ckpt and "vae.encoder.conv_in.weight" not in vae_ckpt:
                    vae_ckpt = vae_ckpt["vae"]
                vae.load_state_dict(vae_ckpt, strict=False)
                del vae_ckpt
        if args.vae_ckpt and os.path.isfile(stats_path):
            latents_stats = torch.load(stats_path)
            latents_scale = latents_stats["latents_scale"].squeeze().to(device)
            latents_bias = latents_stats["latents_bias"].squeeze().to(device)
        elif hasattr(vae, "initial_latents_stats"):
            latents_scale, latents_bias = vae.initial_latents_stats(device=device)
        else:
            from models.autoencoder import FluxVAE
            latents_scale = torch.full((in_channels,), float(FluxVAE.scaling_factor), device=device)
            latents_bias = torch.full((in_channels,), float(FluxVAE.shift_factor), device=device)
    elif args.vae == "qwenvit":
        # Imported here rather than through `vae_models` so that runs which do not
        # use it never need the Qwen3-VL dependency (transformers >= 4.57).
        from models.qwen_vit_ae import QwenViTAutoencoder

        vae = QwenViTAutoencoder(
            model_path=args.vae_vit_path,
            latent_channels=in_channels,
            decoder_width=args.vae_decoder_width,
            decoder_depth=args.vae_decoder_depth,
            decoder_heads=args.vae_decoder_heads,
            decoder_grid=latent_size // 2,
            conv_refine=args.vae_decoder_conv_refine,
            freeze_encoder=args.vae_freeze_encoder,
            grad_checkpointing=args.vae_grad_checkpointing,
            logvar_init=args.vae_logvar_init,
        ).to(device)
        if args.vae_ckpt and os.path.isfile(args.vae_ckpt):
            vae_ckpt = torch.load(args.vae_ckpt, map_location="cpu")
            if isinstance(vae_ckpt, dict) and "vae" in vae_ckpt:
                vae_ckpt = vae_ckpt["vae"]
            missing, unexpected = vae.load_state_dict(vae_ckpt, strict=False)
            if accelerator.is_main_process:
                logger.info(f"Loaded VAE from {args.vae_ckpt} "
                            f"(missing={len(missing)} unexpected={len(unexpected)})")
            del vae_ckpt
        # Nothing pretrained sets the scale of this latent, so BN starts at the
        # identity and adapts online via --bn-momentum.
        latents_scale = torch.ones((in_channels,), device=device)
        latents_bias = torch.zeros((in_channels,), device=device)
        if accelerator.is_main_process:
            logger.info(
                f"Qwen3-VL AE: latent={in_channels}x{latent_size}x{latent_size} "
                f"(rate {in_channels * latent_size ** 2}) "
                f"encoder_frozen={args.vae_freeze_encoder}"
            )
    else:
        vae = vae_models[args.vae]().to(device)
        vae_ckpt = torch.load(args.vae_ckpt, map_location=device)
        vae.load_state_dict(vae_ckpt, strict=False)   # We may not have the projection layer in the VAE
        del vae_ckpt

        # Load the VAE latents-stats for BN layer initialization
        latents_stats = torch.load(args.vae_ckpt.replace(".pt", "-latents-stats.pt"))
        latents_scale = latents_stats["latents_scale"].squeeze().to(device)
        latents_bias = latents_stats["latents_bias"].squeeze().to(device)

    requires_grad(ema, False)
    print("Total trainable params in VAE:", count_trainable_params(vae))

    model.init_bn(latents_bias=latents_bias, latents_scale=latents_scale)

    # Apply SyncBN if more than 1 GPU is used
    if accelerator.use_distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    loss_cfg = OmegaConf.load(args.loss_cfg_path)
    vae_loss_fn = ReconstructionLoss_Single_Stage(loss_cfg).to(device)
    # Optional CLI overrides for the latent regularizer. `entropy` is what makes
    # end-to-end training stable; see --latent-reg.
    if args.latent_reg is not None:
        vae_loss_fn.latent_reg = args.latent_reg
    if args.latent_reg_weight is not None:
        vae_loss_fn.kl_weight = args.latent_reg_weight
    if args.entropy_aux_kl_weight is not None:
        vae_loss_fn.aux_kl_weight = args.entropy_aux_kl_weight
    # Checked here rather than in parse_args because the regularizer can also come
    # from the loss config, in which case parse_args cannot see it.
    if not args.gen_loss and getattr(vae_loss_fn, "latent_reg", "kl") == "entropy":
        raise ValueError(
            f"{args.loss_cfg_path} uses the entropy regularizer, which defends "
            "against a denoising gradient that --no-gen-loss removes; without it "
            "the regularizer inflates the posterior instead. Use "
            "configs/l1_lpips_kl_gan_disc10k.yaml for the reconstruction-only stage."
        )
    if accelerator.is_main_process:
        logger.info(
            f"VAE latent regularizer: {getattr(vae_loss_fn, 'latent_reg', 'n/a')} "
            f"(weight={getattr(vae_loss_fn, 'kl_weight', 'n/a')}, "
            f"aux_kl_weight={getattr(vae_loss_fn, 'aux_kl_weight', 'n/a')}); "
            f"vae_diffusion_grad={args.vae_diffusion_grad} (coeff={args.vae_diffusion_coeff})"
        )

    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- understanding prior on the latent -----------------------------------
    # Adds an understanding objective to the VAE loss so the same latent has to
    # serve generation and understanding at once. --und-coeff 0 (the default)
    # disables the whole path and nothing extra is loaded. See docs/UND.md.
    und_prior = None
    und_params = []
    if args.und_coeff > 0:
        # parse_args already rejects this, but the siglip arm reads
        # text_encoder.tokenizer, which is None without text conditioning.
        assert args.conditioning == "text", (
            "--und-coeff aligns latents to captions, so it needs --conditioning text"
        )
    if args.und_coeff > 0 and args.und_type == "nll":
        # Caption NLL through Qwen's frozen language model. Complementary to
        # siglip rather than a competitor: siglip asks whether the latent can be
        # told apart from other captions in the batch, NLL asks whether a language
        # model can actually read the caption off it, token by token.
        from models.understanding_prior import UnderstandingPrior

        und_prior = UnderstandingPrior(
            model_path=args.und_vlm_path,
            latent_channels=in_channels,
            latent_patch=args.und_latent_patch,
            projector_mode=args.und_projector_mode,
            projector_width=args.und_width,
            projector_depth=args.und_depth,
            projector_heads=args.und_heads,
            load_vision=False,
            via_merger=args.und_via_merger,
            tap=args.und_tap,
            use_deepstack=args.und_deepstack,
            max_caption_tokens=args.text_max_length,
            pmi_mode=args.und_pmi_mode,
            grad_checkpointing=True,
            device=device,
        )
        und_params = und_prior.trainable_parameters()
        if accelerator.is_main_process:
            # The two taps have genuinely different knobs, and printing the
            # irrelevant ones invites reading a run as configured differently than
            # it is: on the trunk tap there is no projector to size and no warm-up
            # to serve, since nothing on that path is trainable.
            where = (f"projector={args.und_projector_mode} via_merger={args.und_via_merger} "
                     f"warmup={args.und_warmup_steps}" if args.und_tap == "latent"
                     else f"deepstack={args.und_deepstack} ceiling={args.und_pmi_ceiling}")
            logger.info(
                f"Understanding prior: nll coeff={args.und_coeff} tap={args.und_tap} "
                f"{where} ({sum(p.numel() for p in und_params)/1e6:.2f}M trainable) "
                f"sub_batch={args.und_batch_size or 'full'} pmi={args.und_pmi_mode}"
            )
    elif args.und_coeff > 0:
        from models.siglip_prior import SiglipPrior
        from models.text_encoder import DEFAULT_CLIP

        # The projection into CLIP's shared space is only defined for CLIP; other
        # towers align against their own pooled space instead.
        use_text_proj = args.und_text_projection and args.text_encoder == "clip"
        embed_dim = 768 if use_text_proj else text_encoder.pooled_dim
        und_prior = SiglipPrior(
            tokenizer=text_encoder.tokenizer,
            latent_channels=in_channels,
            latent_patch=args.und_latent_patch,
            head_mode=args.und_head_mode,
            width=args.und_width,
            num_heads=args.und_heads,
            depth=args.und_depth,
            # Fixed table size, resampled in the forward to whatever grid the
            # latent actually has. Tying it to latent_size instead would change
            # its shape on the 256->512 phase and the warm-start would drop it.
            grid=args.und_pos_grid,
            embed_dim=embed_dim,
            cls_coeff=args.und_cls_coeff,
            text_projection=use_text_proj,
            # Same CLIP the text encoder above resolved, so the prior's copy of
            # text_projection matches the pooled features it is fed.
            clip_path=args.text_encoder_name or DEFAULT_CLIP,
            max_caption_tokens=args.text_max_length,
            device=device,
        )
        und_params = und_prior.trainable_parameters()
        if accelerator.is_main_process:
            logger.info(
                f"Understanding prior: coeff={args.und_coeff} "
                f"head={args.und_head_mode} width={args.und_width} "
                f"({sum(p.numel() for p in und_params)/1e6:.2f}M trainable) "
                f"latent_patch={args.und_latent_patch} embed_dim={embed_dim} "
                f"text_proj={use_text_proj} cls_coeff={args.und_cls_coeff} "
                f"warmup={args.und_warmup_steps}"
            )

    # The trunk tap forks understanding off before the bottleneck, so it needs the
    # tower's VLM-side output threaded out of the VAE's forward.
    und_on_trunk = und_prior is not None and args.und_type == "nll" and args.und_tap == "trunk"
    vae_fwd_kwargs = {"return_vlm": True} if und_on_trunk else {}

    def backward_ctx():
        """Autocast off while backpropagating a graph containing the trunk tap.

        Running backward under autocast is discouraged generally, and with this arm
        it is broken outright: the frozen language model is bf16 while the run's
        autocast dtype is fp16, and deepstack injects its features with an
        index_put_, which carries autocast's "promote to widest" policy and refuses
        to reconcile bf16 against fp16 ("Unexpected floating ScalarType in
        at::autocast::prioritize"). The forward already runs with autocast off for
        the neighbouring reason of precision. Scoped to this arm rather than applied
        to every backward, so no other arm's numerics move; the ops in the graph
        already carry the dtypes the forward chose for them, so this only changes
        what happens to be broken.
        """
        if not und_on_trunk:
            return contextlib.nullcontext()
        return torch.autocast("cuda", enabled=False)

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Define the optimizers for SiT, VAE, and VAE loss function separately
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    # A pretrained encoder cannot share a learning rate with a randomly initialised
    # decoder. Adam's step is lr * g / sqrt(v), so early on every coordinate moves by
    # about lr no matter how small its gradient is: at 1e-4 the tower's features drift
    # ~1e-4 per coordinate per step, coherently across all its blocks, and Qwen's
    # frozen merger and language model -- which need that geometry exactly as
    # pretrained -- stop being able to read the image within about five steps.
    # Measured: caption PMI falls from 1.13 to 0.005 nats/token by step 6 at 1e-4, and
    # holds at ~1.0 for at least 30 steps at 1e-6. A linear probe hides this, because
    # it refits to whatever geometry it is given; the frozen language model cannot.
    tower = [p for p in vae_encoder_params(vae) if p.requires_grad]
    tower_ids = {id(p) for p in tower}
    rest = [p for p in vae.parameters() if p.requires_grad and id(p) not in tower_ids]
    enc_lr = args.vae_encoder_lr if args.vae_encoder_lr > 0 else args.vae_learning_rate
    # Split only when the two rates actually differ: a checkpoint written without
    # the split stores a single Adam group in `vae.parameters()` order, so building
    # two groups of the same lr would make resume die with "different number of
    # parameter groups".
    split_encoder = bool(tower) and enc_lr != args.vae_learning_rate
    if split_encoder:
        vae_groups = [
            {"params": rest, "lr": args.vae_learning_rate},
            {"params": tower, "lr": enc_lr},
        ]
    else:
        # Deliberately the unfiltered `vae.parameters()`: that is what checkpoints
        # written before the split contain, and filtering here would change the
        # single group's length and break resuming them. Frozen entries cost
        # nothing, since Adam only allocates state once a gradient arrives.
        vae_groups = list(vae.parameters())
    optimizer_vae = torch.optim.AdamW(
        vae_groups,
        lr=args.vae_learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    if accelerator.is_main_process and tower:
        if split_encoder:
            logger.info(
                f"VAE optimizer: {sum(p.numel() for p in rest)/1e6:.1f}M params at "
                f"lr={args.vae_learning_rate}, pretrained encoder "
                f"{sum(p.numel() for p in tower)/1e6:.1f}M at lr={enc_lr}"
            )
        else:
            logger.info(
                f"VAE optimizer: single group, "
                f"{sum(p.numel() for p in vae_groups)/1e6:.1f}M params at "
                f"lr={args.vae_learning_rate}"
            )
    optimizer_loss_fn = torch.optim.AdamW(
        vae_loss_fn.parameters(),
        lr=args.disc_learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    # Checkpoints carry only the trainable part of the prior. The nll prior is an
    # nn.Module that owns a frozen 2B language model, so its full state_dict would
    # add several GB of unchanged weights to every save. Stays None on the trunk
    # tap, whose prior is frozen end to end and so has nothing worth saving.
    und_ckpt_module = None
    if und_prior is not None:
        und_ckpt_module = und_prior.projector if args.und_type == "nll" else und_prior

    # The pool and the two heads are the only trainable parts of the understanding
    # prior (the text tower stays frozen). They get their own optimizer so the
    # learning rate is independent of the VAE's, and their own param groups so the
    # temperature and bias escape weight decay.
    optimizer_und = None
    # The trunk tap owns nothing trainable, and AdamW rejects an empty param list.
    if und_prior is not None and len(und_params) > 0:
        # The nll prior has only a projector, so plain decay on everything is
        # right; siglip also owns a temperature and a bias, which must escape it.
        groups = (und_prior.param_groups(args.adam_weight_decay)
                  if hasattr(und_prior, "param_groups")
                  else [{"params": und_params, "weight_decay": args.adam_weight_decay}])
        optimizer_und = torch.optim.AdamW(
            groups,
            lr=args.und_learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
        )

    # Setup data
    if args.conditioning == "text":
        data_roots = [r for r in args.data_dir.split(",") if r]
        caption_columns = [c for c in args.caption_columns.split(",") if c]
        train_dataset = ParquetT2IDataset(
            data_roots,
            resolution=args.resolution,
            caption_columns=caption_columns,
            rows_per_file=args.parquet_rows_per_file,
        )
    elif args.class_data_format == "parquet":
        data_roots = [r for r in args.data_dir.split(",") if r]
        train_dataset = ParquetClassDataset(
            data_roots,
            resolution=args.resolution,
            label_column=args.class_label_column,
            label_map_file=args.class_label_map,
            rows_per_file=args.parquet_rows_per_file,
        )
    else:
        train_dataset = CustomINH5Dataset(args.data_dir)
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")

    # Setup evaluation data + metrics (rFID / PSNR / SSIM / gFID)
    evaluator = None
    eval_dataloader = None
    if args.eval_steps > 0 and args.conditioning != "text" and args.class_data_format == "h5":
        eval_dataset = CustomINH5Dataset(args.eval_data_dir or args.data_dir)
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.eval_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        evaluator = OnlineEvaluator(
            device, eval_recon=args.eval_recon, eval_gen=args.eval_gen, fid_feature=args.fid_feature,
        )

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights

    # Start with eval mode for all models
    model.eval()
    ema.eval()
    vae.eval()

    if args.disc_pretrained_ckpt is not None:
        # Load the discriminator from a pretrained checkpoint if provided
        disc_ckpt = torch.load(args.disc_pretrained_ckpt, map_location=device)
        vae_loss_fn.discriminator.load_state_dict(disc_ckpt)
        if accelerator.is_main_process:
            logger.info(f"Loaded discriminator from {args.disc_pretrained_ckpt}")

    # resume
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt_path = f'{args.cont_dir}/checkpoints/{ckpt_name}'

        # If the checkpoint exists, we load the checkpoint and resume the training
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        vae.load_state_dict(ckpt['vae'])
        vae_loss_fn.discriminator.load_state_dict(ckpt['discriminator'])
        optimizer.load_state_dict(ckpt['opt']),
        optimizer_vae.load_state_dict(ckpt['opt_vae'])
        optimizer_loss_fn.load_state_dict(ckpt['opt_disc'])
        if und_ckpt_module is not None and 'und' in ckpt:
            und_ckpt_module.load_state_dict(ckpt['und'])
        global_step = ckpt['steps']
    elif args.init_ckpt:
        assert os.path.isfile(args.init_ckpt), f"--init-ckpt not found: {args.init_ckpt}"
        ckpt = torch.load(args.init_ckpt, map_location='cpu')

        def _warmstart_load(module, sd, name):
            cur = module.state_dict()
            filtered = {k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}
            skipped = [k for k in sd if k not in filtered]
            missing, unexpected = module.load_state_dict(filtered, strict=False)
            if accelerator.is_main_process:
                logger.info(f"warm-start [{name}]: loaded={len(filtered)} "
                            f"skipped(shape)={skipped} missing={len(missing)} "
                            f"unexpected={len(unexpected)}")

        _warmstart_load(model, ckpt['model'], "model")
        _warmstart_load(ema, ckpt['ema'], "ema")
        _warmstart_load(vae, ckpt['vae'], "vae")
        if 'discriminator' in ckpt:
            _warmstart_load(vae_loss_fn.discriminator, ckpt['discriminator'], "discriminator")
        if und_ckpt_module is not None and 'und' in ckpt:
            # Shape-filtered like the rest: the pooler's position table is stored at
            # --und-pos-grid and resampled in the forward, so a 256->512 warm-start
            # keeps it, but a run that changed the head's width or depth drops it.
            _warmstart_load(und_ckpt_module, ckpt['und'], "und")
        if accelerator.is_main_process:
            logger.info(f"Warm-started from {args.init_ckpt} (optimizers/step reset)")

    # Allow larger cache size for DYNAMo compilation
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 512
    # Model compilation for better performance
    if args.compile:
        model = torch.compile(model, backend="inductor", mode="default")
        vae = torch.compile(vae, backend="inductor", mode="default")
        vae_loss_fn = torch.compile(vae_loss_fn, backend="inductor", mode="default")

    model, vae, vae_loss_fn, optimizer, optimizer_vae, optimizer_loss_fn, train_dataloader = accelerator.prepare(
        model, vae, vae_loss_fn, optimizer, optimizer_vae, optimizer_loss_fn, train_dataloader
    )
    if eval_dataloader is not None:
        eval_dataloader = accelerator.prepare(eval_dataloader)
    if optimizer_und is not None:
        # Prepare the optimizer but NOT the prior: DDP-wrapping a module whose
        # trunk is frozen buys nothing and trips unused-parameter checks, so the
        # projector's gradients are all-reduced by hand below. The optimizer still
        # has to be registered so fp16 gradient unscaling covers it.
        optimizer_und = accelerator.prepare(optimizer_und)

    def sync_und_grads():
        """Average projector gradients across ranks (no DDP wrapper on the prior)."""
        if not accelerator.use_distributed:
            return
        for p in und_params:
            if p.grad is not None:
                torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
                p.grad.div_(accelerator.num_processes)

    def understanding_loss(z_raw, captions, text_pooled, latents_scale, latents_bias, step,
                           vlm=None):
        """Understanding loss: ``--und-type`` picks which, ``--und-tap`` picks where.

        With ``--und-tap latent`` (the default) the prior reads the bottleneck's
        output: the same normalised latent the diffusion model consumes, using the
        SiT/MMDiT BN statistics (detached) so the two views stay aligned as BN
        adapts. Understanding then sits *downstream* of the bottleneck and its
        gradient reaches the encoder only through it.

        With ``--und-tap trunk`` the two objectives fork at the tower instead:
        understanding reads the tower's own VLM-side output while reconstruction and
        denoising read the bottleneck, as siblings. Nothing on the understanding
        path is trainable or reparameterised, so it is the stock Qwen3-VL forward
        and the only way to lower the NLL is to make the trunk more informative.
        Neither the BN normalisation nor the warm-up applies: there is no head to
        warm up, and at step 0 the trunk is still the pretrained one, which makes
        this the one gradient in the system that is trustworthy immediately.

        siglip scores the whole batch -- the text side is already computed for the
        MMDiT conditioning and the image side is one pooling head, and the
        contrastive term wants every sample it can get as a negative. nll runs a
        2B language model forward *and* backward, so it takes --und-batch-size
        samples; the sub-batch is the leading slice, which is already a random
        subset because the loader shuffles.
        """
        # The contrastive matrix, the logsigmoid, the vocabulary-wide softmax and
        # the cross-entropy over Qwen's 152k vocabulary all want full precision; the
        # surrounding fp16 autocast would cost accuracy for no memory saving, and
        # would also fight the language model's own bf16 weights.
        if und_on_trunk:
            n = min(args.und_batch_size or vlm.embeds.shape[0], vlm.embeds.shape[0])
            sub = vlm._replace(
                embeds=vlm.embeds[:n], deepstack=[d[:n] for d in vlm.deepstack]
            )
            with torch.autocast("cuda", enabled=False):
                return und_prior.forward_vlm(sub, list(captions[:n]))

        z_und = z_raw
        if step < args.und_warmup_steps:
            # Head-only warm-up: a freshly initialised head emits a direction
            # uncorrelated with anything, and letting that gradient into the
            # encoder early just injects noise.
            z_und = z_und.detach()
        z_und = (z_und - latents_bias.view(1, -1, 1, 1)) * latents_scale.view(1, -1, 1, 1)
        with torch.autocast("cuda", enabled=False):
            if args.und_type == "nll":
                n = args.und_batch_size or z_und.shape[0]
                n = min(n, z_und.shape[0])
                return und_prior(z_und[:n].float(), list(captions[:n]))
            return und_prior(z_und.float(), text_pooled, list(captions))

    if use_wandb:
        # Initialize wandb directly rather than through accelerate's tracker
        # wrapper, so the run config and step numbering stay under our control.
        wandb.init(
            project=args.wandb_project,
            name=f"{args.exp_name}",
            dir=save_dir,
            config=vars(copy.deepcopy(args)),
            settings={"console": "off"},
        )

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Labels / captions for periodic qualitative samples.
    sample_batch_size = max(1, args.num_vis_samples // accelerator.num_processes)
    if args.conditioning == "text":
        _sample_prompts = [
            "a photograph of an astronaut riding a horse",
            "a cute corgi puppy sitting in a field of flowers, bokeh",
            "an oil painting of a lighthouse during a storm",
            "a bowl of ramen with an egg, food photography",
        ]
        sample_prompts = [_sample_prompts[i % len(_sample_prompts)] for i in range(sample_batch_size)]
        with torch.no_grad():
            sample_context, sample_pooled, _ = text_encoder.encode(sample_prompts)
            null_context, null_pooled, _ = text_encoder.encode([""] * sample_batch_size)
        sample_context = sample_context.to(device); sample_pooled = sample_pooled.to(device)
        null_context = null_context.to(device); null_pooled = null_pooled.to(device)
        ys = None
    else:
        ys = torch.randint(1000, size=(sample_batch_size,), device=device).to(device)
    # Create sampling noise:
    xT = torch.randn((sample_batch_size, in_channels, latent_size, latent_size), device=device)

    # main training loop
    for epoch in range(args.epochs):
        model.train()

        for raw_image, y in train_dataloader:
            raw_image = raw_image.to(device)
            if args.conditioning == "text":
                with torch.no_grad():
                    _ctx, _pooled, _ = text_encoder.encode(list(y))
                cond_kwargs = {"context": _ctx.to(device), "pooled": _pooled.to(device)}
            else:
                cond_kwargs = {"y": y.to(device)}
            z = None

            # extract the dinov2 features
            with torch.no_grad():
                zs = []
                with accelerator.autocast():
                    for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
                        raw_image_ = preprocess_raw_image(raw_image, encoder_type)
                        z = encoder.forward_features(raw_image_)
                        if 'mocov3' in encoder_type: z = z = z[:, 1:] 
                        if 'dinov2' in encoder_type: z = z['x_norm_patchtokens']
                        zs.append(z)

            vae.train()
            model.train()
            with accelerator.accumulate([model, vae, vae_loss_fn]), accelerator.autocast():
                # 1). Forward pass: VAE
                processed_image = preprocess_imgs_vae(raw_image)
                # The trunk tap needs the tower's VLM-side output, which only the
                # Qwen3-VL AE can return; every other configuration gets the plain
                # three-tuple and leaves `vlm_tokens` None.
                vlm_tokens = None
                if vae_fwd_kwargs:
                    posterior, z, recon_image, vlm_tokens = vae(processed_image, **vae_fwd_kwargs)
                else:
                    posterior, z, recon_image = vae(processed_image)

                # Latent diagnostics (monitor collapse): computed every step, cheap.
                latent_stat_raw, latent_per_dim_std = compute_latent_stats(z, posterior)

                # Per-channel latents_scale (= 1/sqrt(running_var)) from the SiT BN.
                # The diffusion gradient that reaches `logvar` is scaled per
                # channel by this BN factor; we feed it to the entropy regularizer
                # so its gradient to `logvar` carries the same per-channel factor
                # (global strength still set by --latent-reg-weight). Detached:
                # used purely as a constant reweighting, no grad through BN stats.
                # The bias is only needed by the understanding prior, which reads the
                # same normalised latent the diffusion model consumes.
                _bn_stats = accelerator.unwrap_model(model).extract_latents_stats()
                latents_scale_perchan = _bn_stats["latents_scale"].detach()
                latents_bias_perchan = _bn_stats["latents_bias"].detach()

                # 2). Backward pass: VAE, compute the VAE loss, backpropagate, and update the VAE; Then, compute the discriminator loss and update the discriminator
                #    loss_kwargs used for SiT forward function, create here and can be reused for both VAE and SiT
                loss_kwargs = dict(
                    path_type=args.path_type,
                    prediction=args.prediction,
                    weighting=args.weighting,
                )
                # Record the time_input and noises for the VAE alignment, so that we avoid sampling again
                time_input = None
                noises = None

                if not args.vae_diffusion_grad:
                    # ===================== Baseline: REPA-E-style, no e2e gradient =====================
                    # Only the REPA alignment gradient reaches the VAE; the SiT
                    # diffusion (denoising) loss is isolated from the VAE via z.detach().
                    # Turn off grads for the SiT model (avoid REPA gradient on the SiT model)
                    requires_grad(model, False)
                    # Avoid BN stats to be updated by the VAE, unless the denoising
                    # pass is disabled -- then this align-only forward is the only one
                    # left, and BN has to keep tracking the latent statistics through
                    # it so the next stage inherits sane normalization.
                    if args.gen_loss:
                        model.eval()
                    else:
                        model.train()

                    vae_loss, vae_loss_dict = vae_loss_fn(processed_image, recon_image, posterior, global_step, "generator", latents_scale=latents_scale_perchan)
                    vae_loss = vae_loss.mean()

                    # Compute the REPA alignment loss for VAE updates
                    loss_kwargs["align_only"] = True
                    vae_align_outputs = model(
                        x=z,
                        **cond_kwargs,
                        zs=zs,
                        loss_kwargs=loss_kwargs,
                        time_input=time_input,
                        noises=noises,
                    )
                    vae_loss = vae_loss + args.vae_align_proj_coeff * vae_align_outputs["proj_loss"].mean()
                    # Save the `time_input` and `noises` and reuse them for the SiT model forward pass
                    time_input = vae_align_outputs["time_input"]
                    noises = vae_align_outputs["noises"]

                    if und_prior is not None:
                        und_outputs = understanding_loss(
                            z, y, cond_kwargs["pooled"],
                            latents_scale_perchan, latents_bias_perchan, global_step,
                            vlm=vlm_tokens,
                        )
                        vae_loss = vae_loss + args.und_coeff * und_outputs["loss"]

                    with backward_ctx():
                        accelerator.backward(vae_loss)
                    if accelerator.sync_gradients:
                        # Unscales the gradients of every registered optimizer, so
                        # the prior's are plain floats from here on.
                        grad_norm_vae = accelerator.clip_grad_norm_(vae.parameters(), args.max_grad_norm)
                        if und_params:
                            sync_und_grads()
                            grad_norm_und = torch.nn.utils.clip_grad_norm_(und_params, args.max_grad_norm)
                    optimizer_vae.step()
                    optimizer_vae.zero_grad(set_to_none=True)
                    if optimizer_und is not None:
                        optimizer_und.step()
                        optimizer_und.zero_grad(set_to_none=True)

                    # discriminator loss and update
                    d_loss, d_loss_dict = vae_loss_fn(processed_image, recon_image, posterior, global_step, "discriminator")
                    d_loss = d_loss.mean()
                    accelerator.backward(d_loss)
                    if accelerator.sync_gradients:
                        grad_norm_disc = accelerator.clip_grad_norm_(vae_loss_fn.parameters(), args.max_grad_norm)
                    optimizer_loss_fn.step()
                    optimizer_loss_fn.zero_grad(set_to_none=True)

                    # Turn the grads back on for the SiT model, and put the model into training mode
                    requires_grad(model, True)
                    model.train()

                    if args.gen_loss:
                        # 3). Forward pass: SiT
                        # **Avoid diffusion loss to backpropagate to the VAE, so we detach the `z`**
                        loss_kwargs["weighting"] = args.weighting
                        loss_kwargs["align_only"] = False
                        sit_outputs = model(
                            x=z.detach(),
                            **cond_kwargs,
                            zs=zs,
                            loss_kwargs=loss_kwargs,
                            time_input=time_input,
                            noises=noises,
                        )

                        # 4). Compute diffusion loss and REPA alignment loss, backpropagate the SiT loss, and update the model
                        sit_loss = sit_outputs["denoising_loss"].mean() + args.proj_coeff * sit_outputs["proj_loss"].mean()
                        accelerator.backward(sit_loss)
                        if accelerator.sync_gradients:
                            grad_norm_sit = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                else:
                    # ================= End-to-end: denoising gradient into the VAE =================
                    # The SiT denoising loss back-propagates into the VAE encoder
                    # (latent NOT detached). Combine the VAE loss and the SiT loss
                    # into a single backward so the shared latent `z` keeps one graph.

                    # VAE recon / perceptual / latent-reg / GAN-generator loss.
                    #
                    # NOTE:
                    # In the combined-backward path we run two SiT forwards
                    # (align_only then denoising) before a single backward call. If we
                    # do align_only in eval mode and denoising in train mode, SiT's BN
                    # running stats (shape [in_channels], e.g. [4]) can be updated
                    # in-place between the two forwards and trigger autograd version
                    # mismatch under torch.compile. Keep both forwards in train mode and
                    # freeze SiT params via requires_grad(False) for align_only.
                    requires_grad(model, False)
                    model.train()
                    vae_loss, vae_loss_dict = vae_loss_fn(processed_image, recon_image, posterior, global_step, "generator", latents_scale=latents_scale_perchan)
                    vae_loss = vae_loss.mean()

                    loss_kwargs["align_only"] = True
                    vae_align_outputs = model(
                        x=z,
                        **cond_kwargs,
                        zs=zs,
                        loss_kwargs=loss_kwargs,
                        time_input=time_input,
                        noises=noises,
                    )
                    vae_loss = vae_loss + args.vae_align_proj_coeff * vae_align_outputs["proj_loss"].mean()
                    time_input = vae_align_outputs["time_input"]
                    noises = vae_align_outputs["noises"]

                    if und_prior is not None:
                        und_outputs = understanding_loss(
                            z, y, cond_kwargs["pooled"],
                            latents_scale_perchan, latents_bias_perchan, global_step,
                            vlm=vlm_tokens,
                        )
                        vae_loss = vae_loss + args.und_coeff * und_outputs["loss"]

                    # SiT diffusion pass with the latent ATTACHED (no detach). The
                    # denoising gradient flows into the VAE through `z`, scaled by
                    # `vae_diffusion_coeff`; SiT parameters get the unscaled gradient.
                    requires_grad(model, True)
                    model.train()
                    loss_kwargs["weighting"] = args.weighting
                    loss_kwargs["align_only"] = False
                    sit_outputs = model(
                        x=grad_scale(z, args.vae_diffusion_coeff),
                        **cond_kwargs,
                        zs=zs,
                        loss_kwargs=loss_kwargs,
                        time_input=time_input,
                        noises=noises,
                    )
                    sit_loss = sit_outputs["denoising_loss"].mean() + args.proj_coeff * sit_outputs["proj_loss"].mean()

                    # Single combined backward: VAE <- recon+reg+align+understanding
                    # +denoising(via z); SiT <- denoising+proj.
                    total_loss = vae_loss + sit_loss
                    with backward_ctx():
                        accelerator.backward(total_loss)
                    if accelerator.sync_gradients:
                        # In fp16 mode, Accelerator.clip_grad_norm_() internally calls
                        # scaler.unscale_() on *all* tracked optimizers. Calling it
                        # twice before optimizer.step() would re-unscale and crash.
                        # The understanding optimizer has to be unscaled here too, or
                        # the prior would step on still-scaled gradients.
                        unscale = [optimizer_vae, optimizer]
                        if optimizer_und is not None:
                            unscale.append(optimizer_und)
                        accelerator.unscale_gradients(optimizer=unscale)
                        grad_norm_vae = torch.nn.utils.clip_grad_norm_(vae.parameters(), args.max_grad_norm)
                        grad_norm_sit = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        if und_params:
                            sync_und_grads()
                            grad_norm_und = torch.nn.utils.clip_grad_norm_(und_params, args.max_grad_norm)
                    optimizer_vae.step()
                    optimizer.step()
                    optimizer_vae.zero_grad(set_to_none=True)
                    optimizer.zero_grad(set_to_none=True)
                    if optimizer_und is not None:
                        optimizer_und.step()
                        optimizer_und.zero_grad(set_to_none=True)

                    # discriminator loss and update (recon is detached inside the loss fn)
                    d_loss, d_loss_dict = vae_loss_fn(processed_image, recon_image, posterior, global_step, "discriminator")
                    d_loss = d_loss.mean()
                    accelerator.backward(d_loss)
                    if accelerator.sync_gradients:
                        grad_norm_disc = accelerator.clip_grad_norm_(vae_loss_fn.parameters(), args.max_grad_norm)
                    optimizer_loss_fn.step()
                    optimizer_loss_fn.zero_grad(set_to_none=True)

                # 5). Update SiT EMA
                if accelerator.sync_gradients:
                    unwrapped_model = accelerator.unwrap_model(model)
                    update_ema(ema, unwrapped_model._orig_mod if args.compile else unwrapped_model)

            # enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Prepare the logs based on the current step
                logs = {
                    "epoch": epoch,
                    "vae_loss": accelerator.gather(vae_loss).mean().detach().item(),
                    "reconstruction_loss": accelerator.gather(vae_loss_dict["reconstruction_loss"].mean()).mean().detach().item(),
                    "perceptual_loss": accelerator.gather(vae_loss_dict["perceptual_loss"].mean()).mean().detach().item(),
                    "kl_loss": accelerator.gather(vae_loss_dict["kl_loss"].mean()).mean().detach().item(),
                    "aux_kl_value": accelerator.gather(vae_loss_dict["aux_kl_value"].mean()).mean().detach().item(),
                    "weighted_gan_loss": accelerator.gather(vae_loss_dict["weighted_gan_loss"].mean()).mean().detach().item(),
                    "discriminator_factor": accelerator.gather(vae_loss_dict["discriminator_factor"].mean()).mean().detach().item(),
                    "gan_loss": accelerator.gather(vae_loss_dict["gan_loss"].mean()).mean().detach().item(),
                    "d_weight": accelerator.gather(vae_loss_dict["d_weight"].mean()).mean().detach().item(),
                    "grad_norm_vae": accelerator.gather(grad_norm_vae).mean().detach().item(),
                    "vae_align_loss": accelerator.gather(vae_align_outputs["proj_loss"].mean()).mean().detach().item(),
                    "d_loss": accelerator.gather(d_loss).mean().detach().item(),
                    "grad_norm_disc": accelerator.gather(grad_norm_disc).mean().detach().item(),
                    "logits_real": accelerator.gather(d_loss_dict["logits_real"].mean()).mean().detach().item(),
                    "logits_fake": accelerator.gather(d_loss_dict["logits_fake"].mean()).mean().detach().item(),
                    "lecam_loss": accelerator.gather(d_loss_dict["lecam_loss"].mean()).mean().detach().item(),
                    # Per-channel BN scale (1/sqrt(running_var)) used to reweight the
                    # entropy regularizer; logged to track the effective magnitude.
                    # Synced across ranks via SyncBN, so no gather needed.
                    "latent/latents_scale_mean": latents_scale_perchan.float().mean().detach().item(),
                }
                if args.gen_loss:
                    logs.update({
                        "sit_loss": accelerator.gather(sit_loss).mean().detach().item(),
                        "denoising_loss": accelerator.gather(sit_outputs["denoising_loss"]).mean().detach().item(),
                        "proj_loss": accelerator.gather(sit_outputs["proj_loss"]).mean().detach().item(),
                        "grad_norm_sit": accelerator.gather(grad_norm_sit).mean().detach().item(),
                    })
                if und_prior is not None and args.und_type == "nll":
                    # `pmi` is the number to watch, not `nll`: the caption NLL is
                    # dominated by what a language model can guess from the caption
                    # prefix alone, so it falls even when the latent contributes
                    # nothing. pmi is that baseline subtracted -- the NLL against a
                    # mismatched latent minus the NLL against the real one -- so it
                    # is the part of the likelihood that the image actually earned.
                    logs["und/nll"] = accelerator.gather(und_outputs["nll"]).mean().detach().item()
                    logs["und/num_visual_tokens"] = und_outputs["num_visual_tokens"]
                    if und_params:
                        logs["und/grad_norm"] = accelerator.gather(grad_norm_und).mean().detach().item()
                    if "pmi" in und_outputs:
                        logs["und/pmi"] = accelerator.gather(und_outputs["pmi"]).mean().detach().item()
                        logs["und/null_nll"] = accelerator.gather(und_outputs["nll_null"]).mean().detach().item()
                        if args.und_pmi_ceiling > 0:
                            # The fraction of the base VLM's image-dependent
                            # likelihood still there. On the trunk tap this starts at
                            # 1.0 by construction, so the curve reads directly as
                            # "how much VLM readability has survived learning to
                            # reconstruct" rather than as an uncalibrated nat count.
                            logs["und/pmi_frac"] = logs["und/pmi"] / args.und_pmi_ceiling
                elif und_prior is not None:
                    # `i2t_r1` is the number to watch: the fraction of images whose
                    # own caption is the nearest text among all `num_negatives + 1`
                    # on every rank. Chance level is its reciprocal, so it is
                    # interpretable from step 1 -- unlike PMI, which sits inside its
                    # own noise for thousands of steps. `pos_sim - neg_sim` is the
                    # same signal without the argmax quantisation, and separates
                    # "not learning" from "collapsed to one direction" (both sims
                    # high, gap ~0).
                    logs["und/i2t_r1"] = accelerator.gather(und_outputs["i2t_r1"]).mean().detach().item()
                    logs["und/pos_sim"] = accelerator.gather(und_outputs["pos_sim"]).mean().detach().item()
                    logs["und/neg_sim"] = accelerator.gather(und_outputs["neg_sim"]).mean().detach().item()
                    logs["und/siglip"] = accelerator.gather(und_outputs["siglip"]).mean().detach().item()
                    logs["und/logit_scale"] = und_outputs["logit_scale"].item()
                    logs["und/logit_bias"] = und_outputs["logit_bias"].item()
                    logs["und/grad_norm"] = accelerator.gather(grad_norm_und).mean().detach().item()
                    logs["und/chance_r1"] = 1.0 / (und_outputs["num_negatives"] + 1)
                    if "cls" in und_outputs:
                        logs["und/cls"] = accelerator.gather(und_outputs["cls"]).mean().detach().item()
                # Latent diagnostics: reduce across ranks (collective -> all ranks).
                latent_logs = reduce_latent_stats(accelerator, latent_stat_raw)
                per_dim_std_global = accelerator.gather(
                    latent_per_dim_std.reshape(1, -1)
                ).mean(0)

                progress_bar.set_postfix(**logs)
                if use_wandb:
                    wandb.log({**logs, **latent_logs}, step=global_step)
                    wandb.log(
                        {"latent/z_perdim_std_hist": wandb.Histogram(
                            per_dim_std_global.detach().cpu().numpy())},
                        step=global_step,
                    )

            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    # `model` and `vae` are wrapped by the `accelerator` object, so we need to unwrap them
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_vae = accelerator.unwrap_model(vae)
                    unwrapped_vae_loss_fn = accelerator.unwrap_model(vae_loss_fn)

                    # model might be compiled, we extract the original model
                    original_model = unwrapped_model._orig_mod if args.compile else unwrapped_model
                    original_vae = unwrapped_vae._orig_mod if args.compile else unwrapped_vae
                    original_discriminator = unwrapped_vae_loss_fn._orig_mod.discriminator if args.compile else unwrapped_vae_loss_fn.discriminator

                    checkpoint = {
                        "model": original_model.state_dict(),
                        "ema": ema.state_dict(),
                        "vae": original_vae.state_dict(),
                        "discriminator": original_discriminator.state_dict(),
                        "opt": optimizer.state_dict(),
                        "opt_vae": optimizer_vae.state_dict(),
                        "opt_disc": optimizer_loss_fn.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    if und_ckpt_module is not None:
                        checkpoint["und"] = und_ckpt_module.state_dict()
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            if args.gen_loss and (global_step == 1 or (global_step % args.sampling_steps == 0 and global_step > 0)):
                # NOTE: Inference should use eval mode
                model.eval()
                vae.eval()
                with torch.no_grad():
                    unwrapped_model = accelerator.unwrap_model(model)
                    if args.conditioning == "text":
                        samples = mmdit_text_euler_sampler(
                            unwrapped_model, xT, sample_context, sample_pooled,
                            null_context, null_pooled, num_steps=50, cfg_scale=4.0,
                            path_type=args.path_type,
                        ).to(torch.float32)
                    else:
                        samples = euler_sampler(
                            unwrapped_model,
                            xT, 
                            ys,
                            num_steps=50, 
                            cfg_scale=4.0,
                            guidance_low=0.,
                            guidance_high=1.,
                            path_type=args.path_type,
                            heun=False,
                        ).to(torch.float32)
                    latents_stats = unwrapped_model.extract_latents_stats()
                    # reshape latents_stats to [1, C, 1, 1]
                    latents_scale = latents_stats['latents_scale'].view(1, in_channels, 1, 1)
                    latents_bias = latents_stats['latents_bias'].view(1, in_channels, 1, 1)
                    samples = accelerator.unwrap_model(vae).decode(
                        denormalize_latents(samples, latents_scale, latents_bias)).sample
                    samples = (samples + 1) / 2.
                out_samples = accelerator.gather(samples.to(torch.float32))
                if use_wandb:
                    wandb.log({"samples": wandb.Image(array2grid(out_samples))}, step=global_step)
                logging.info("Generating EMA samples done.")

            # rFID / PSNR / SSIM (reconstruction) + gFID (generation) evaluation
            # Smoke eval at step 1 (fail fast); regular eval every --eval-steps.
            is_smoke_eval = args.eval_at_start and global_step == 1
            is_regular_eval = (
                args.eval_steps > 0 and global_step % args.eval_steps == 0 and global_step > 0
            )
            if evaluator is not None and (is_smoke_eval or is_regular_eval):
                model.eval()
                vae.eval()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_vae = accelerator.unwrap_model(vae)

                def recon_fn(raw_image, y):
                    proc = preprocess_imgs_vae(raw_image)
                    with accelerator.autocast():
                        _, _, recon = unwrapped_vae(proc)
                    return (recon.float() + 1) / 2.

                def gen_fn(y):
                    nb = y.shape[0]
                    xT_ = torch.randn(nb, in_channels, latent_size, latent_size, device=device)
                    with accelerator.autocast():
                        samples = euler_sampler(
                            unwrapped_model, xT_, y, num_steps=args.eval_gen_steps,
                            cfg_scale=args.eval_cfg_scale, guidance_low=0., guidance_high=1.,
                            path_type=args.path_type, heun=False,
                        ).to(torch.float32)
                        ls = unwrapped_model.extract_latents_stats()
                        lscale = ls['latents_scale'].view(1, in_channels, 1, 1)
                        lbias = ls['latents_bias'].view(1, in_channels, 1, 1)
                        img = unwrapped_vae.decode(denormalize_latents(samples, lscale, lbias)).sample
                    return (img.float() + 1) / 2.

                # Smoke run uses very few samples just to exercise the code path.
                num_eval_samples = (
                    min(args.eval_smoke_samples, args.eval_num_samples)
                    if is_smoke_eval else args.eval_num_samples
                )
                eval_tag = "eval_smoke" if is_smoke_eval else "eval"
                if is_smoke_eval and accelerator.is_main_process:
                    logger.info(f"[smoke eval @ step {global_step}] running {num_eval_samples} samples/proc to check the eval path...")
                metrics = run_online_eval(
                    evaluator, eval_dataloader, device, num_eval_samples,
                    recon_fn=recon_fn if args.eval_recon else None,
                    gen_fn=gen_fn if args.eval_gen else None,
                )
                if accelerator.is_main_process:
                    if use_wandb:
                        wandb.log({f"{eval_tag}/{k}": v for k, v in metrics.items()}, step=global_step)
                    logger.info(f"[{eval_tag} @ step {global_step}] " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    model.eval()
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    if use_wandb:
        wandb.finish()
    # accelerator.trackers only exists once init_trackers has run, which it
    # never does here: logging goes through wandb directly.
    if getattr(accelerator, "trackers", None):
        accelerator.end_training()


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging params
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb", choices=["wandb", "none"],
                        help="Set to 'none' to train without a wandb account.")
    parser.add_argument("--wandb-project", type=str, default="genfirst",
                        help="wandb project name (only used with --report-to wandb).")
    parser.add_argument("--sampling-steps", type=int, default=10000)
    parser.add_argument("--num-vis-samples", type=int, default=16,
                        help="Total number of images in the wandb sample grid (kept small to stay under the media upload limit).")
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--continue-train-exp-dir", type=str, default=None)
    parser.add_argument("--init-ckpt", type=str, default="",
                        help="Warm-start weights (model/ema/vae/discriminator) from a "
                             "previous phase's .pt. Optimizers/step are reset and any "
                             "shape-mismatched tensor (e.g. pos_embed on a 256->512 "
                             "change) is skipped. Use to chain training phases.")
    parser.add_argument("--wandb-history-path", type=str, default=None)

    # SiT / MMDiT model params
    parser.add_argument("--model", type=str, default="SiT-XL/2",
                        choices=list(SiT_models.keys()) + list(MMDiT_models.keys()),
                        help="The model to train (SiT-* for class-cond, MMDiT-* for text-cond).")
    parser.add_argument("--conditioning", type=str, default="class", choices=["class", "text"],
                        help="'class' = ImageNet class labels (SiT); 'text' = captions (MMDiT).")
    parser.add_argument("--class-data-format", type=str, default="h5", choices=["h5", "parquet"],
                        help="Class-cond data source: 'h5' or parquet shards.")
    parser.add_argument("--class-label-column", type=str, default="class_label",
                        help="Class-label column name for --class-data-format parquet.")
    parser.add_argument("--class-label-map", type=str, default=None,
                        help="Optional txt file mapping string class names to integer ids.")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bn-momentum", type=float, default=0.1)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                        help="Whether to compile the model for faster training")

    # dataset params
    parser.add_argument("--data-dir", type=str, default="data",
                        help="ImageNet h5 dir (class), or comma-separated parquet roots (text).")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=256)

    # text-to-image params (used when --conditioning text)
    parser.add_argument("--text-encoder", type=str, default="clip", choices=["clip", "t5"])
    parser.add_argument("--text-encoder-name", type=str, default=None,
                        help="HF model id/path for the text encoder (defaults per --text-encoder).")
    parser.add_argument("--text-max-length", type=int, default=77)
    parser.add_argument("--caption-columns", type=str,
                        default="caption,prompt,text",
                        help="Comma-separated candidate caption columns in the parquet shards.")
    parser.add_argument("--parquet-rows-per-file", type=int, default=None,
                        help="Assume this many rows per parquet shard (skips metadata scan).")

    # precision params
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization params
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=50000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed params
    parser.add_argument("--seed", type=int, default=0)

    # cpu params
    parser.add_argument("--num-workers", type=int, default=4)

    # loss params
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"],
                        help="currently we only support v-prediction")
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b')
    parser.add_argument("--proj-coeff", type=float, default=0.5)
    parser.add_argument("--weighting", default="uniform", type=str, choices=["uniform", "lognormal"],
                        help="Loss weihgting, uniform or lognormal")

    # vae params
    parser.add_argument("--vae", type=str, default="f8d4",
                        choices=["f8d4", "f16d32", "flux", "flux2", "qwenvit"])
    parser.add_argument("--vae-ckpt", type=str, default="pretrained/sdvae-f8d4/sdvae-f8d4.pt",
                        help="Extracted VAE state_dict (.pt/.safetensors). Optional for --vae flux/flux2 when "
                             "initializing from --vae-pretrained-path.")
    parser.add_argument("--vae-pretrained-path", type=str, default=None,
                        help="For --vae flux/flux2: path to the pretrained FLUX.1/FLUX.2 `vae` diffusers folder.")

    # === --vae qwenvit: Qwen3-VL's vision tower as the encoder ===
    # The tower already carries semantics that a from-scratch conv VAE does not (an
    # ImageNet ridge linear probe reads 67.1% off it against 3.1% off a FLUX latent), so
    # the understanding objective becomes "does the bottleneck preserve it" rather
    # than "can a head invent it". Needs transformers >= 4.57. See docs/UND.md.
    parser.add_argument("--vae-vit-path", type=str, default="Qwen/Qwen3-VL-2B-Instruct",
                        help="Qwen3-VL model whose vision tower becomes the encoder.")
    parser.add_argument("--latent-channels", type=int, default=16,
                        help="Latent channels for --vae qwenvit. Each ViT token becomes a 2x2 "
                             "block, so the grid is resolution/8 and 16 makes the latent both "
                             "shape- and rate-identical to FLUX's: 16*32*32 = 16384 numbers at "
                             "256px, i.e. 64 numbers per 16x16x3 pixel patch, the same 12x "
                             "compression as FLUX's 192 -> 16. 64 would be 4x FLUX's rate.")
    parser.add_argument("--vae-freeze-encoder", action=argparse.BooleanOptionalAction, default=True,
                        help="Freeze the tower and train only the bottleneck and decoder. This is "
                             "the first phase of the recipe: reconstruction has to be working "
                             "before the encoder is allowed to move, or the decoder's early "
                             "gradient destroys the representation the tower came with.")
    parser.add_argument("--vae-grad-checkpointing", action=argparse.BooleanOptionalAction,
                        default=True, help="Checkpoint the tower's activations once it is unfrozen.")
    parser.add_argument("--vae-logvar-init", type=float, default=-6.0,
                        help="Initial posterior log-variance for --vae qwenvit. The mean head "
                             "starts with std ~0.22, so -6 (std 0.05) keeps the sampling noise "
                             "well below the signal; the entropy regularizer raises it from there.")
    parser.add_argument("--vae-decoder-width", type=int, default=1024)
    parser.add_argument("--vae-decoder-depth", type=int, default=12)
    parser.add_argument("--vae-decoder-heads", type=int, default=16)
    parser.add_argument("--vae-decoder-conv-refine", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Two 3x3 convolutions on the assembled image. The decoder emits one "
                             "16x16 RGB patch per token with no pixel-level mixing across borders, "
                             "which leaves visible seams when it is trained from scratch.")

    # vae loss params
    parser.add_argument("--disc-pretrained-ckpt", type=str, default=None)
    parser.add_argument("--loss-cfg-path", type=str, default="configs/l1_lpips_kl_gan.yaml")

    # vae training params
    parser.add_argument("--vae-learning-rate", type=float, default=1e-4)
    parser.add_argument("--vae-encoder-lr", type=float, default=0.0,
                        help="Separate learning rate for a pretrained VAE encoder (the "
                             "Qwen3-VL tower); 0 means use --vae-learning-rate, which is "
                             "the right choice for a conv VAE trained from scratch. It "
                             "matters because Adam moves every coordinate by about lr in "
                             "the early steps whatever its gradient is, so at 1e-4 the "
                             "tower drifts out of the geometry Qwen's frozen merger and "
                             "language model were trained to read and caption PMI "
                             "collapses within ~5 steps. Set it as large as keeps "
                             "und/pmi_frac up: that is the most encoder adaptation the "
                             "understanding path can survive.")
    parser.add_argument("--disc-learning-rate", type=float, default=1e-4)
    parser.add_argument("--vae-align-proj-coeff", type=float, default=1.5)
    # --no-gen-loss drops the denoising pass and trains reconstruction only. It
    # exists for the frozen-encoder warm-up of an encoder that arrives pretrained
    # but with no decoder (--vae qwenvit): there is nothing to denoise into yet, so
    # the decoder and bottleneck are fitted first. The align-only forward still
    # runs, which is what keeps the SiT's BatchNorm tracking the new latent's
    # statistics so the joint stage inherits sane normalization. See docs/UND.md.
    parser.add_argument("--gen-loss", action=argparse.BooleanOptionalAction, default=True,
                        help="Train the SiT denoising loss. --no-gen-loss leaves "
                             "reconstruction only, for a decoder warm-up stage.")

    # === In-training evaluation: rFID / PSNR / SSIM (recon) + gFID (gen) ===
    # NOTE: these use a subset of images and are a relative monitoring proxy,
    # not the official 50k-sample FID. Set --eval-steps 0 to disable.
    parser.add_argument("--eval-steps", type=int, default=50000,
                        help="Run rFID/PSNR/SSIM/gFID every N steps (0 disables).")
    parser.add_argument("--eval-num-samples", type=int, default=5000,
                        help="Number of images (per process) used per evaluation.")
    parser.add_argument("--eval-batch-size", type=int, default=32,
                        help="Per-process batch size for the evaluation loader.")
    parser.add_argument("--eval-data-dir", type=str, default=None,
                        help="Optional separate eval data dir (defaults to --data-dir).")
    parser.add_argument("--eval-recon", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-gen", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-gen-steps", type=int, default=50,
                        help="Sampler steps for gFID generation.")
    parser.add_argument("--eval-cfg-scale", type=float, default=1.0,
                        help="CFG scale for gFID generation (1.0 = no guidance).")
    parser.add_argument("--fid-feature", type=int, default=2048, choices=[64, 192, 768, 2048])
    # Run a tiny eval pass early (at step 1) so any error in the recon/gen/eval
    # code path surfaces immediately instead of crashing at the first
    # --eval-steps milestone (e.g. step 50000). Uses --eval-smoke-samples images.
    parser.add_argument("--eval-at-start", action=argparse.BooleanOptionalAction, default=True,
                        help="Run a small smoke eval at step 1 to fail fast on eval errors.")
    parser.add_argument("--eval-smoke-samples", type=int, default=64,
                        help="Number of images (per process) for the step-1 smoke eval.")

    # === End-to-end: denoising gradient into the VAE + entropy latent reg ===
    # When enabled, the SiT diffusion (denoising) loss is back-propagated into
    # the VAE encoder (the latent `z` is NOT detached for the SiT pass). To
    # prevent latent collapse under this gradient, use `--latent-reg entropy`.
    parser.add_argument("--vae-diffusion-grad", action=argparse.BooleanOptionalAction, default=False,
                        help="Let the SiT denoising loss back-propagate into the VAE encoder (removes z.detach()).")
    parser.add_argument("--vae-diffusion-coeff", type=float, default=1.0,
                        help="Scale on the denoising gradient that reaches the VAE (only used with --vae-diffusion-grad).")
    parser.add_argument("--latent-reg", type=str, default=None, choices=["kl", "entropy"],
                        help="Override the VAE latent regularizer. Default: use the value from the loss config yaml.")
    parser.add_argument("--latent-reg-weight", type=float, default=None,
                        help="Override the latent-reg weight (kl_weight) from the loss config yaml.")
    parser.add_argument("--entropy-aux-kl-weight", type=float, default=None,
                        help="Override the small auxiliary KL weight added on top of the entropy reg "
                             "(its mean^2 term keeps the encoder mean bounded). 0 disables it.")

    # === Understanding prior on the latent (docs/UND.md) ===
    # --und-coeff 0 disables the whole path and nothing extra is loaded.
    #
    # Two objectives, --und-type selects one. They ask different questions and are
    # not redundant:
    #   nll (default): caption negative log-likelihood under Qwen's frozen language
    #     model, reached through the tower's own frozen merger. Expensive -- a 2B
    #     forward and backward -- hence --und-batch-size, and it needs
    #     transformers >= 4.57. Read und/pmi, not und/nll.
    #   siglip: sigmoid pairwise contrastive + caption-token cls, on a weak
    #     attention pool of the latent. Cheap (one head, full batch, text side
    #     already computed for conditioning) and adds no dependency.
    parser.add_argument("--und-type", type=str, default="nll", choices=["siglip", "nll"],
                        help="Which understanding objective to add to the VAE loss.")
    parser.add_argument("--und-coeff", type=float, default=0.0,
                        help="Weight of the understanding loss added to the VAE loss (0 disables). "
                             "The SigLIP term sums over negatives rather than averaging, so its "
                             "magnitude grows with the global batch: calibrate by comparing "
                             "und/grad_norm against grad_norm_vae.")
    parser.add_argument("--und-latent-patch", type=int, default=2,
                        help="k x k latent patch per token; 2 gives 16x16=256 tokens at 256px f8.")
    parser.add_argument("--und-head-mode", type=str, default="attnpool",
                        choices=["mean", "attnpool", "transformer"],
                        help="Capacity of the pool over latent tokens. Keep it weak: with a "
                             "strong pool a falling loss only shows the pool learned to read, "
                             "not that the latent carries more. 'transformer' is for measuring "
                             "the ceiling of a frozen latent, not for training runs.")
    parser.add_argument("--und-width", type=int, default=1024,
                        help="Hidden width of the pool (siglip) or projector (nll).")
    parser.add_argument("--und-heads", type=int, default=8,
                        help="Attention heads in the pool / projector.")
    parser.add_argument("--und-depth", type=int, default=4,
                        help="Self-attention blocks before the pool (--und-head-mode transformer).")
    parser.add_argument("--und-pos-grid", type=int, default=16,
                        help="Size of the learnable 2D position table, resampled to the latent's "
                             "actual grid, so it survives the 256->512 phase unchanged.")
    parser.add_argument("--und-cls-coeff", type=float, default=1.0,
                        help="Weight of the caption-token classification term. Unlike the "
                             "contrastive term it needs no negatives, so it is the part that does "
                             "not weaken at small batch sizes; 0 disables it.")
    parser.add_argument("--und-text-projection", action=argparse.BooleanOptionalAction, default=True,
                        help="Align to CLIP's shared image-text space by applying CLIP's own "
                             "text_projection to the pooled caption vector. Only the matrix is "
                             "loaded, so the text encoder that conditions the MMDiT is untouched. "
                             "Off aligns to the raw pooled space instead.")
    # --- --und-type nll only ---
    parser.add_argument("--und-vlm-path", type=str, default="Qwen/Qwen3-VL-2B-Instruct",
                        help="Frozen VLM whose language model scores the caption.")
    parser.add_argument("--und-via-merger", action=argparse.BooleanOptionalAction, default=True,
                        help="Route the latent through the vision tower's own frozen patch merger "
                             "on its way into the language model, instead of letting the projector "
                             "map straight into embedding space. With the merger fixed, the only "
                             "way to lower the NLL is to make the latent readable; a free projector "
                             "can lower it by inventing an embedding space of its own. Also matches "
                             "how Qwen really sees an image: 64 merged tokens at 256px, not 256.")
    parser.add_argument("--und-tap", type=str, default="latent", choices=["latent", "trunk"],
                        help="Where the understanding loss reads from. 'latent' puts it downstream "
                             "of the bottleneck, so its gradient reaches the tower only through a "
                             "16x compression. 'trunk' forks the two objectives at the tower "
                             "instead -- understanding reads the tower's own VLM-side output "
                             "(merged tokens plus deepstack, the stock Qwen3-VL forward) while "
                             "reconstruction and denoising read the bottleneck, as siblings. The "
                             "trunk path has no trainable parameter, no projector and no "
                             "temperature, so it cannot be lowered by anything except making the "
                             "trunk more informative; the cost is that the latent is then "
                             "supervised only indirectly, by being a linear read of that trunk. "
                             "Requires --vae qwenvit with the encoder unfrozen, and nll.")
    parser.add_argument("--und-deepstack", action=argparse.BooleanOptionalAction, default=True,
                        help="On the trunk tap, also feed the deepstack features Qwen3-VL taps at "
                             "its intermediate vision layers into the language model's first "
                             "layers, as the real model does. Besides being the faithful forward, "
                             "it puts the understanding gradient directly onto the tower's early "
                             "blocks instead of only at its output.")
    parser.add_argument("--und-pmi-ceiling", type=float, default=0.0,
                        help="Reference PMI for the und/pmi_frac log, in nats/token. The natural "
                             "value is the base VLM's own PMI, which the trunk tap reproduces at "
                             "step 0, so a short run with a frozen encoder measures it.")
    parser.add_argument("--und-projector-mode", type=str, default="linear",
                        choices=["linear", "mlp", "transformer"],
                        help="Capacity of the latent -> merger-input map. Keep it weak for the same "
                             "reason as --und-head-mode; 'transformer' measures a frozen latent's "
                             "ceiling rather than training against it.")
    parser.add_argument("--und-batch-size", type=int, default=0,
                        help="Samples per step scored by the NLL prior (0 = the whole local batch). "
                             "The 2B forward+backward is what makes this necessary; the slice is "
                             "the leading one, already random because the loader shuffles.")
    parser.add_argument("--und-pmi-mode", type=str, default="monitor",
                        choices=["off", "monitor", "token_weight"],
                        help="'monitor' also scores each caption against a mismatched latent so "
                             "und/pmi can be logged (one extra no-grad forward). 'token_weight' "
                             "additionally reweights the trained tokens toward those the null pass "
                             "found surprising. 'off' skips it.")
    parser.add_argument("--und-learning-rate", type=float, default=1e-4,
                        help="Learning rate for the understanding prior's own optimizer.")
    parser.add_argument("--und-warmup-steps", type=int, default=2000,
                        help="Steps during which the understanding gradient updates only the pool "
                             "and heads (latent detached). A cold pool emits a direction "
                             "uncorrelated with anything, and that gradient is noise for the "
                             "encoder.")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if not args.gen_loss and args.vae_diffusion_grad:
        raise ValueError(
            "--vae-diffusion-grad routes the denoising gradient into the VAE, which "
            "is meaningless with --no-gen-loss; drop one of the two."
        )
    if not args.gen_loss and args.latent_reg == "entropy":
        # Entropy maximization is the anti-collapse term for the case where the
        # denoising gradient reaches the encoder. With --no-gen-loss there is no such
        # gradient, so nothing pushes back and the regularizer inflates the posterior
        # variance instead: measured, from 0.0025 to 2.29 within 1000 steps, leaving
        # most of the latent's variance as sampling noise and reconstruction stuck.
        raise ValueError(
            "--latent-reg entropy defends against a denoising gradient that "
            "--no-gen-loss removes, and without it the regularizer inflates the "
            "posterior instead. Use --latent-reg kl with a small "
            "--latent-reg-weight (1e-6) for the reconstruction-only stage."
        )
    if args.und_coeff > 0 and args.conditioning != "text":
        raise ValueError("--und-coeff scores captions, so it needs --conditioning text")
    if args.und_coeff > 0 and args.und_tap == "trunk":
        # Each of these would silently turn the objective into something else: a
        # different encoder has no Qwen tower to read, siglip has no head on this
        # path, and a frozen tower puts the whole branch under no_grad, leaving a
        # loss that is logged but cannot move anything.
        if args.vae != "qwenvit":
            raise ValueError(f"--und-tap trunk reads Qwen3-VL's tower, so it needs "
                             f"--vae qwenvit, not {args.vae}")
        if args.und_type != "nll":
            raise ValueError("--und-tap trunk is the caption-NLL path; pass --und-type nll")
        if args.vae_freeze_encoder:
            raise ValueError("--und-tap trunk supervises the tower, which "
                             "--vae-freeze-encoder holds fixed; drop one of the two.")

    # Backward-compatible alias used by resume logic.
    # If user does not specify --continue-train-exp-dir, resume from current exp dir.
    args.cont_dir = args.continue_train_exp_dir or os.path.join(args.output_dir, args.exp_name)
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
