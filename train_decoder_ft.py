# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

"""Decoder-only VAE finetuning, run after the end-to-end stage.

The latent space moves quickly during end-to-end training, which can leave the
decoder lagging behind the latents it is being asked to decode. This finetunes
ONLY the decoder to catch up. The encoder stays frozen, so the latent distribution
does not move and the latent statistics carry over unchanged -- the finetuned VAE
is a drop-in replacement everywhere the original was used.

There is no SiT and no REPA alignment here; it is a plain reconstruction-GAN
finetune:

    trainable : vae.decoder + vae.post_quant_conv, discriminator
    frozen    : vae.encoder + vae.quant_conv
    loss      : L1 + LPIPS + GAN (the latent regularizer has no effect, since the
                encoder it acts on is frozen)

10k steps is what the paper uses for the ImageNet-256 results.

    torchrun --standalone --nproc_per_node=8 train_decoder_ft.py \
        --init-ckpt exps/<end-to-end-run>/checkpoints/0600000.pt \
        --vae f8d4 --data-dir data --batch-size 256 \
        --max-train-steps 10000 --exp-name decoder-ft-f8d4
"""
import argparse
import copy
import logging
import math
import os
import json
from collections import OrderedDict
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm.auto import tqdm
import wandb

from dataset import CustomINH5Dataset
from loss.losses import ReconstructionLoss_Single_Stage
from models.autoencoder import vae_models
from utils import preprocess_imgs_vae, count_trainable_params
from eval_utils import OnlineEvaluator, run_online_eval

logger = get_logger(__name__)


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return x


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """Step the EMA model towards the current model (params + float buffers)."""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

    ema_buffers = OrderedDict(ema_model.named_buffers())
    model_buffers = OrderedDict(model.named_buffers())
    for name, buffer in model_buffers.items():
        name = name.replace("module.", "")
        if buffer.dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
            ema_buffers[name].mul_(decay).add_(buffer.data, alpha=1 - decay)
        else:
            ema_buffers[name].copy_(buffer)


def create_logger(logging_dir):
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")],
    )
    return logging.getLogger(__name__)


def extract_latents_stats(ckpt):
    """Recover latent statistics (scale/bias) from an end-to-end checkpoint.

    Mirrors save_vae_weights.py: they come from the SiT EMA BatchNorm buffers. They
    are carried over unchanged, because the encoder is frozen and so the latent
    distribution does not move -- which is what makes the finetuned decoder a
    drop-in for the sampling and evaluation pipeline.
    """
    try:
        bn_running_mean = ckpt["ema"]["bn.running_mean"]
        bn_running_var = ckpt["ema"]["bn.running_var"]
    except (KeyError, TypeError):
        return None
    in_channels = bn_running_mean.shape[0]
    return {
        "latents_scale": bn_running_var.rsqrt().view(1, in_channels, 1, 1),
        "latents_bias": bn_running_mean.view(1, in_channels, 1, 1),
    }


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
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

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    if args.vae == "f8d4":
        assert args.resolution % 8 == 0
    elif args.vae == "f16d32":
        assert args.resolution % 16 == 0
    else:
        raise NotImplementedError()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- Build VAE + loss(discriminator) and warm-start from the source ckpt ---
    vae = vae_models[args.vae]().to(device)
    loss_cfg = OmegaConf.load(args.loss_cfg_path)
    vae_loss_fn = ReconstructionLoss_Single_Stage(loss_cfg).to(device)

    latents_stats = None
    global_step = 0
    if args.resume_step > 0:
        # Resume a previous decoder-finetune run (self-contained checkpoint).
        ckpt_path = f"{args.cont_dir}/checkpoints/{str(args.resume_step).zfill(7)}.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu")
        vae.load_state_dict(ckpt["vae"])
        vae_loss_fn.discriminator.load_state_dict(ckpt["discriminator"])
        latents_stats = ckpt.get("latents_stats", None)
        global_step = ckpt["steps"]
        if accelerator.is_main_process:
            logger.info(f"Resumed decoder finetune from {ckpt_path} (step {global_step})")
    else:
        # Fresh finetune: warm-start VAE + discriminator from the end-to-end ckpt.
        src = torch.load(args.init_ckpt, map_location="cpu")
        vae.load_state_dict(src["vae"], strict=False)
        if "discriminator" in src and src["discriminator"] is not None:
            vae_loss_fn.discriminator.load_state_dict(src["discriminator"])
            if accelerator.is_main_process:
                logger.info("Loaded discriminator from source checkpoint.")
        elif args.disc_pretrained_ckpt is not None:
            vae_loss_fn.discriminator.load_state_dict(
                torch.load(args.disc_pretrained_ckpt, map_location="cpu")
            )
        latents_stats = extract_latents_stats(src)
        if accelerator.is_main_process:
            logger.info(f"Warm-started VAE from {args.init_ckpt}")
            if latents_stats is None:
                # Without them the finetuned VAE cannot be used for sampling, and
                # the failure would otherwise only surface much later.
                logger.warning(
                    "No latent statistics found in the source checkpoint (no SiT EMA "
                    "BatchNorm buffers). Carry them over by hand from the end-to-end "
                    "run before using this VAE for sampling."
                )
        del src

    # EMA of the (finetuned) VAE for the final decoder.
    ema_vae = copy.deepcopy(vae).to(device)
    requires_grad(ema_vae, False)
    if args.resume_step > 0 and "vae_ema" in ckpt:
        ema_vae.load_state_dict(ckpt["vae_ema"])

    # --- Freeze the encoder; train only the decoder (+ post_quant_conv) ---------
    requires_grad(vae, False)
    requires_grad(vae.decoder, True)
    requires_grad(vae.post_quant_conv, True)
    trainable_params = [p for p in vae.parameters() if p.requires_grad]
    if accelerator.is_main_process:
        logger.info(
            f"VAE trainable params (decoder only): {count_trainable_params(vae):,} "
            f"/ total {sum(p.numel() for p in vae.parameters()):,}"
        )

    optimizer_vae = torch.optim.AdamW(
        trainable_params,
        lr=args.vae_learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    optimizer_loss_fn = torch.optim.AdamW(
        vae_loss_fn.parameters(),
        lr=args.disc_learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    if args.resume_step > 0:
        optimizer_vae.load_state_dict(ckpt["opt_vae"])
        optimizer_loss_fn.load_state_dict(ckpt["opt_disc"])
        del ckpt

    # --- Data -------------------------------------------------------------------
    train_dataset = CustomINH5Dataset(args.data_dir)
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset, batch_size=local_batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")

    evaluator = None
    eval_dataloader = None
    if args.eval_steps > 0:
        eval_dataset = CustomINH5Dataset(args.eval_data_dir or args.data_dir)
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=args.eval_batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
        )
        evaluator = OnlineEvaluator(device, eval_recon=True, eval_gen=False, fid_feature=args.fid_feature)

    vae, vae_loss_fn, optimizer_vae, optimizer_loss_fn, train_dataloader = accelerator.prepare(
        vae, vae_loss_fn, optimizer_vae, optimizer_loss_fn, train_dataloader
    )
    if eval_dataloader is not None:
        eval_dataloader = accelerator.prepare(eval_dataloader)

    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"{args.exp_name}",
            dir=save_dir,
            config=vars(copy.deepcopy(args)),
            settings={"console": "off"},
        )

    progress_bar = tqdm(
        range(0, args.max_train_steps), initial=global_step, desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # A fixed batch for reconstruction visualization.
    vis_raw, _ = next(iter(train_dataloader))
    vis_raw = vis_raw[: max(1, args.num_vis_samples // accelerator.num_processes)].to(device)

    for epoch in range(args.epochs):
        vae.train()
        for raw_image, _ in train_dataloader:
            raw_image = raw_image.to(device)

            with accelerator.accumulate([vae, vae_loss_fn]), accelerator.autocast():
                processed_image = preprocess_imgs_vae(raw_image)
                # Encoder is frozen; only the decode path carries gradients.
                posterior, z, recon_image = vae(processed_image)

                # 1) Generator (decoder) update: recon + perceptual + GAN.
                vae_loss, vae_loss_dict = vae_loss_fn(
                    processed_image, recon_image, posterior, global_step, "generator",
                    latents_scale=None,
                )
                vae_loss = vae_loss.mean()
                accelerator.backward(vae_loss)
                if accelerator.sync_gradients:
                    grad_norm_vae = accelerator.clip_grad_norm_(vae.parameters(), args.max_grad_norm)
                optimizer_vae.step()
                optimizer_vae.zero_grad(set_to_none=True)

                # 2) Discriminator update.
                d_loss, d_loss_dict = vae_loss_fn(
                    processed_image, recon_image, posterior, global_step, "discriminator"
                )
                d_loss = d_loss.mean()
                accelerator.backward(d_loss)
                if accelerator.sync_gradients:
                    grad_norm_disc = accelerator.clip_grad_norm_(vae_loss_fn.parameters(), args.max_grad_norm)
                optimizer_loss_fn.step()
                optimizer_loss_fn.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema_vae, accelerator.unwrap_model(vae), decay=args.ema_decay)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                logs = {
                    "epoch": epoch,
                    "vae_loss": accelerator.gather(vae_loss).mean().detach().item(),
                    "reconstruction_loss": accelerator.gather(vae_loss_dict["reconstruction_loss"].mean()).mean().detach().item(),
                    "perceptual_loss": accelerator.gather(vae_loss_dict["perceptual_loss"].mean()).mean().detach().item(),
                    "weighted_gan_loss": accelerator.gather(vae_loss_dict["weighted_gan_loss"].mean()).mean().detach().item(),
                    "gan_loss": accelerator.gather(vae_loss_dict["gan_loss"].mean()).mean().detach().item(),
                    "discriminator_factor": accelerator.gather(vae_loss_dict["discriminator_factor"].mean()).mean().detach().item(),
                    "grad_norm_vae": accelerator.gather(grad_norm_vae).mean().detach().item(),
                    "d_loss": accelerator.gather(d_loss).mean().detach().item(),
                    "grad_norm_disc": accelerator.gather(grad_norm_disc).mean().detach().item(),
                    "logits_real": accelerator.gather(d_loss_dict["logits_real"].mean()).mean().detach().item(),
                    "logits_fake": accelerator.gather(d_loss_dict["logits_fake"].mean()).mean().detach().item(),
                    "lecam_loss": accelerator.gather(d_loss_dict["lecam_loss"].mean()).mean().detach().item(),
                }
                progress_bar.set_postfix(**logs)
                if use_wandb:
                    wandb.log(logs, step=global_step)

            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    unwrapped_vae = accelerator.unwrap_model(vae)
                    unwrapped_loss_fn = accelerator.unwrap_model(vae_loss_fn)
                    checkpoint = {
                        "vae": unwrapped_vae.state_dict(),
                        "vae_ema": ema_vae.state_dict(),
                        "discriminator": unwrapped_loss_fn.discriminator.state_dict(),
                        "opt_vae": optimizer_vae.state_dict(),
                        "opt_disc": optimizer_loss_fn.state_dict(),
                        "latents_stats": latents_stats,
                        "args": args,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    # Also drop the latent statistics alongside it, so the finetuned
                    # decoder is a drop-in for the sampling / evaluation pipeline.
                    if latents_stats is not None:
                        torch.save(latents_stats, f"{checkpoint_dir}/{global_step:07d}-latents-stats.pt")
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            if global_step == 1 or (global_step % args.sampling_steps == 0 and global_step > 0):
                vae.eval()
                with torch.no_grad(), accelerator.autocast():
                    unwrapped_vae = accelerator.unwrap_model(vae)
                    proc = preprocess_imgs_vae(vis_raw)
                    _, _, recon = unwrapped_vae(proc)
                    recon = (recon.float() + 1) / 2.
                    real = vis_raw.float() / 255.
                grid = torch.cat([real, recon], dim=0)
                grid = accelerator.gather(grid.to(torch.float32))
                if use_wandb:
                    wandb.log({"recon (top: real, bottom: recon)": wandb.Image(array2grid(grid))}, step=global_step)
                vae.train()

            is_smoke_eval = args.eval_at_start and global_step == 1
            is_regular_eval = args.eval_steps > 0 and global_step % args.eval_steps == 0 and global_step > 0
            if evaluator is not None and (is_smoke_eval or is_regular_eval):
                vae.eval()
                unwrapped_vae = accelerator.unwrap_model(vae)

                def recon_fn(raw, y):
                    with accelerator.autocast():
                        _, _, rec = unwrapped_vae(preprocess_imgs_vae(raw.to(device)))
                    return (rec.float() + 1) / 2.

                num_eval_samples = (
                    min(args.eval_smoke_samples, args.eval_num_samples) if is_smoke_eval else args.eval_num_samples
                )
                eval_tag = "eval_smoke" if is_smoke_eval else "eval"
                metrics = run_online_eval(
                    evaluator, eval_dataloader, device, num_eval_samples, recon_fn=recon_fn, gen_fn=None,
                )
                if accelerator.is_main_process:
                    if use_wandb:
                        wandb.log({f"{eval_tag}/{k}": v for k, v in metrics.items()}, step=global_step)
                    logger.info(f"[{eval_tag} @ step {global_step}] " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
                vae.train()

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

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
    parser = argparse.ArgumentParser(description="Decoder-only VAE finetuning")

    # logging
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb",
                        choices=["wandb", "none"])
    parser.add_argument("--wandb-project", type=str, default="genfirst")
    parser.add_argument("--sampling-steps", type=int, default=2000,
                        help="Log a real-vs-recon image grid every N steps.")
    parser.add_argument("--num-vis-samples", type=int, default=16)

    # init / resume
    parser.add_argument("--init-ckpt", type=str, default=None,
                        help="End-to-end checkpoint to warm-start the VAE + discriminator from.")
    parser.add_argument("--resume-step", type=int, default=0,
                        help="Resume a previous decoder-finetune run from this step.")
    parser.add_argument("--continue-train-exp-dir", type=str, default=None)

    # dataset
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--resolution", type=int, choices=[256], default=256)
    parser.add_argument("--batch-size", type=int, default=256)

    # precision
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=10000)
    parser.add_argument("--checkpointing-steps", type=int, default=10000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-weight-decay", type=float, default=0.)
    parser.add_argument("--adam-epsilon", type=float, default=1e-08)
    parser.add_argument("--max-grad-norm", default=1.0, type=float)
    parser.add_argument("--ema-decay", type=float, default=0.9999)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)

    # vae
    parser.add_argument("--vae", type=str, default="f8d4", choices=["f8d4", "f16d32"])
    parser.add_argument("--vae-learning-rate", type=float, default=1e-4)
    parser.add_argument("--disc-learning-rate", type=float, default=1e-4)
    parser.add_argument("--disc-pretrained-ckpt", type=str, default=None,
                        help="Fallback discriminator ckpt if the source checkpoint has none.")
    parser.add_argument("--loss-cfg-path", type=str, default="configs/l1_lpips_entropy_gan.yaml")

    # eval (reconstruction only; no generation here)
    parser.add_argument("--eval-steps", type=int, default=10000,
                        help="Run rFID/PSNR/SSIM (reconstruction) every N steps (0 disables).")
    parser.add_argument("--eval-num-samples", type=int, default=5000)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-data-dir", type=str, default=None)
    parser.add_argument("--fid-feature", type=int, default=2048, choices=[64, 192, 768, 2048])
    parser.add_argument("--eval-at-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-smoke-samples", type=int, default=64)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    args.cont_dir = args.continue_train_exp_dir or os.path.join(args.output_dir, args.exp_name)
    if args.resume_step == 0 and args.init_ckpt is None:
        raise ValueError("Provide --init-ckpt for a fresh finetune, or --resume-step for a resume.")
    return args


if __name__ == "__main__":
    main(parse_args())
