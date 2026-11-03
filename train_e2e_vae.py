# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

"""Three-stage end-to-end VAE training on aspect-ratio-bucketed web-scale data:

  --mode e2e      stages 1/2: the VAE is trained jointly with an unconditional
                  SiT. The SiT denoising gradient reaches the encoder through the
                  latent (scaled by --vae-diffusion-coeff), the VAE gets a REPA
                  alignment loss through the SiT projector, and an entropy
                  regularizer keeps the posterior from collapsing.
  --mode decoder  stage 3: encoder frozen; only the decoder is trained with
                  L1 + LPIPS + GAN. No SiT / DINO.

Data is streamed from parquet shards and aspect-ratio bucketed at a fixed pixel
area (--side**2), see vae_data.py.
"""

import argparse
import copy
import glob
import json
import logging
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from loss.losses import ReconstructionLoss_Single_Stage
from models.autoencoder import vae_models
from models.sit_e2e import SiT_models
from utils import load_encoders
from vae_data import ParquetImageStream, ensure_hdfs_env

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _GradScale(torch.autograd.Function):
    """Identity forward; scales the gradient flowing back into the VAE latent."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = float(scale)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


def grad_scale(x, scale):
    return x if scale == 1.0 else _GradScale.apply(x, scale)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = dict(ema_model.named_parameters())
    for name, param in model.named_parameters():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
    ema_buffers = dict(ema_model.named_buffers())
    for name, buf in model.named_buffers():
        name = name.replace("module.", "")
        if name in ema_buffers:
            ema_buffers[name].copy_(buf)


def dino_input(x_uint8, patch=16, dino_patch=14):
    """[B,3,H,W] in 0..255 -> DINOv2 input whose token grid equals the SiT grid
    (H/16 x W/16 for an f8 VAE with patch-2 SiT)."""
    h, w = x_uint8.shape[-2:]
    x = x_uint8 / 255.0
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std
    return F.interpolate(x, size=(h // patch * dino_patch, w // patch * dino_patch), mode="bicubic", align_corners=False)


def lr_factor(step, args):
    if args.warmup_steps > 0 and step < args.warmup_steps:
        return (step + 1) / args.warmup_steps
    if args.lr_schedule == "constant":
        return 1.0
    progress = min(1.0, (step - args.warmup_steps) / max(1, args.max_train_steps - args.warmup_steps))
    return args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def sample_uncond(model, shape, num_steps, device, dtype):
    """Euler ODE for the linear path with v-prediction (v = eps - x0), t: 1 -> 0."""
    x = torch.randn(shape, device=device, dtype=torch.float32)
    y = torch.zeros(shape[0], dtype=torch.long, device=device)
    ts = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    for i in range(num_steps):
        t = ts[i].expand(shape[0])
        with torch.autocast("cuda", dtype=dtype):
            v = model.inference(x, t, y).float()
        x = x + (ts[i + 1] - ts[i]) * v
    return x


def latest_checkpoint(ckpt_dir):
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "[0-9]" * 7 + ".pt")))
    return ckpts[-1] if ckpts else None


def warmstart_load(module, sd, name, log):
    cur = module.state_dict()
    filtered = {k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}
    skipped = [k for k in sd if k not in filtered]
    missing, _ = module.load_state_dict(filtered, strict=False)
    log(f"warm-start [{name}]: loaded={len(filtered)} skipped={skipped[:8]}{'...' if len(skipped) > 8 else ''} "
        f"missing={len(missing)}")


def main(args):
    assert args.mixed_precision in ("bf16", "fp16", "no")
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_config=ProjectConfiguration(project_dir=args.output_dir),
    )
    device = accelerator.device
    is_main = accelerator.is_main_process
    save_dir = os.path.join(args.output_dir, args.exp_name)
    ckpt_dir = os.path.join(save_dir, "checkpoints")
    if is_main:
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(os.path.join(save_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        logging.basicConfig(
            level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(save_dir, "log.txt"))],
        )
    log = logger.info if is_main else (lambda *a, **k: None)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    world = accelerator.num_processes
    assert args.batch_size % world == 0, f"--batch-size {args.batch_size} not divisible by {world} processes"
    local_bs = args.batch_size // world
    e2e = args.mode == "e2e"
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}[args.mixed_precision]

    # ---------------- VAE ----------------
    vae = vae_models[args.vae](pretrained_path=args.vae_pretrained_path or None).to(device)
    if args.vae_grad_ckpt:
        vae.vae.enable_gradient_checkpointing()
    latent_ch = vae.embed_dim
    if hasattr(vae, "initial_latents_stats"):
        latents_scale, latents_bias = vae.initial_latents_stats(device=device)
    else:
        latents_scale = torch.full((latent_ch,), float(vae.scaling_factor), device=device)
        latents_bias = torch.full((latent_ch,), float(vae.shift_factor), device=device)

    # ---------------- SiT + DINOv2 (e2e only) ----------------
    model = ema = None
    encoders = []
    if e2e:
        encoders, encoder_types, _ = load_encoders(args.enc_type, device, 256)
        z_dims = [enc.embed_dim for enc in encoders]
        model = SiT_models[args.model](
            in_channels=latent_ch, num_classes=1, class_dropout_prob=0.0, z_dims=z_dims,
            encoder_depth=args.encoder_depth, bn_momentum=args.bn_momentum, path_type=args.path_type,
            qk_norm=args.qk_norm, grad_ckpt=args.sit_grad_ckpt,
        ).to(device)
        model.init_bn(latents_scale=latents_scale, latents_bias=latents_bias)
        ema = copy.deepcopy(model).to(device)
        requires_grad(ema, False)
        log(f"SiT params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    else:
        # Encoder frozen: only post_quant_conv + decoder train.
        vae.vae.encoder.requires_grad_(False)
        if getattr(vae.vae, "quant_conv", None) is not None:
            vae.vae.quant_conv.requires_grad_(False)
    log(f"VAE trainable params: {sum(p.numel() for p in vae.parameters() if p.requires_grad) / 1e6:.1f}M "
        f"/ {sum(p.numel() for p in vae.parameters()) / 1e6:.1f}M")

    # ---------------- losses ----------------
    loss_cfg = OmegaConf.load(args.loss_cfg_path)
    loss_cfg.losses.discriminator_start = args.disc_start
    loss_cfg.losses.discriminator_weight = args.disc_weight
    vae_loss_fn = ReconstructionLoss_Single_Stage(loss_cfg).to(device)
    vae_loss_fn.latent_reg = args.latent_reg
    vae_loss_fn.kl_weight = args.latent_reg_weight
    vae_loss_fn.aux_kl_weight = args.entropy_aux_kl_weight
    log(f"latent_reg={args.latent_reg} w={args.latent_reg_weight} aux_kl={args.entropy_aux_kl_weight}; "
        f"disc start={args.disc_start} w={args.disc_weight}; diffusion->VAE coeff={args.vae_diffusion_coeff}")

    # ---------------- optimizers ----------------
    betas = (args.adam_beta1, args.adam_beta2)
    opt_vae = torch.optim.AdamW([p for p in vae.parameters() if p.requires_grad], lr=args.vae_learning_rate,
                                betas=betas, weight_decay=args.adam_weight_decay)
    opt_disc = torch.optim.AdamW(vae_loss_fn.discriminator.parameters(), lr=args.disc_learning_rate,
                                 betas=betas, weight_decay=args.adam_weight_decay)
    opt_sit = None
    if e2e:
        opt_sit = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=betas,
                                    weight_decay=args.adam_weight_decay)

    # ---------------- resume / warm-start ----------------
    global_step = 0
    carried_bn = None  # latent stats for decoder mode exports (no SiT there)
    resume_path = latest_checkpoint(ckpt_dir) if args.resume == "auto" else (args.resume or None)
    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        vae.load_state_dict(ckpt["vae"])
        vae_loss_fn.discriminator.load_state_dict(ckpt["discriminator"])
        opt_vae.load_state_dict(ckpt["opt_vae"])
        opt_disc.load_state_dict(ckpt["opt_disc"])
        if e2e:
            model.load_state_dict(ckpt["model"])
            ema.load_state_dict(ckpt["ema"])
            opt_sit.load_state_dict(ckpt["opt"])
        else:
            carried_bn = ckpt["ema"]
        global_step = int(ckpt["steps"])
        log(f"resumed from {resume_path} @ step {global_step}")
        del ckpt
    elif args.init_ckpt:
        assert os.path.isfile(args.init_ckpt), f"--init-ckpt not found: {args.init_ckpt}"
        ckpt = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        warmstart_load(vae, ckpt["vae"], "vae", log)
        if "discriminator" in ckpt and not args.reset_disc:
            warmstart_load(vae_loss_fn.discriminator, ckpt["discriminator"], "discriminator", log)
        if e2e and "model" in ckpt:
            warmstart_load(model, ckpt["model"], "model", log)
            warmstart_load(ema, ckpt["ema"], "ema", log)
        carried_bn = {k: v for k, v in ckpt["ema"].items() if k.startswith("bn.")}
        log(f"warm-started from {args.init_ckpt} (optimizers/step reset)")
        del ckpt
    if not e2e:
        assert carried_bn is not None, "--mode decoder needs --init-ckpt (or a resume) from an e2e stage"

    if e2e and world > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # ---------------- data ----------------
    ensure_hdfs_env()
    dataset = ParquetImageStream(
        args.data_json, args.filelist_dir, args.side, local_bs,
        rank=accelerator.process_index, world_size=world, seed=args.seed + global_step,
        min_area_ratio=args.min_area_ratio, max_ratio=args.max_aspect_ratio,
    )
    loader = DataLoader(dataset, batch_size=None, num_workers=args.num_workers, pin_memory=True,
                        prefetch_factor=4 if args.num_workers > 0 else None,
                        persistent_workers=args.num_workers > 0)
    set_seed(args.seed + accelerator.process_index + global_step)

    if e2e:
        model, vae, vae_loss_fn, opt_sit, opt_vae, opt_disc = accelerator.prepare(
            model, vae, vae_loss_fn, opt_sit, opt_vae, opt_disc)
    else:
        vae, vae_loss_fn, opt_vae, opt_disc = accelerator.prepare(vae, vae_loss_fn, opt_vae, opt_disc)
    base_lrs = {"vae": args.vae_learning_rate, "disc": args.disc_learning_rate, "sit": args.learning_rate}

    use_wandb = is_main and args.report_to == "wandb"
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.exp_name, dir=save_dir, config=vars(args),
                   settings={"console": "off"}, resume="allow")

    def state_for_save():
        uv = accelerator.unwrap_model(vae)
        ud = accelerator.unwrap_model(vae_loss_fn).discriminator
        ck = {"vae": uv.state_dict(), "discriminator": ud.state_dict(), "opt_vae": opt_vae.state_dict(),
              "opt_disc": opt_disc.state_dict(), "args": vars(args), "steps": global_step}
        if e2e:
            ck.update(model=accelerator.unwrap_model(model).state_dict(), ema=ema.state_dict(),
                      opt=opt_sit.state_dict())
        else:
            ck["ema"] = carried_bn
        return ck

    def export_vae(ck):
        """Same layout as save_vae_weights.py, loadable by REPA/e2e_vae.py:load_e2e_vae."""
        out = os.path.join(save_dir, "export")
        os.makedirs(out, exist_ok=True)
        c = ck["ema"]["bn.running_mean"].numel()
        stats = {"latents_scale": ck["ema"]["bn.running_var"].rsqrt().view(1, c, 1, 1),
                 "latents_bias": ck["ema"]["bn.running_mean"].view(1, c, 1, 1)}
        torch.save(ck["vae"], os.path.join(out, f"{args.exp_name}.pt"))
        torch.save(stats, os.path.join(out, f"{args.exp_name}-latents-stats.pt"))
        torch.save(ck["discriminator"], os.path.join(out, f"{args.exp_name}-discriminator-ckpt.pt"))
        log(f"exported VAE to {out}")

    loss_kwargs = dict(path_type=args.path_type, prediction="v", weighting=args.weighting)
    log(f"start: step {global_step}/{args.max_train_steps}, global batch {args.batch_size} "
        f"({local_bs}/GPU x {world}), side {args.side}, mode {args.mode}")

    running = {}
    t_log = time.time()
    data_iter = iter(loader)
    while global_step < args.max_train_steps:
        t_data = time.time()
        raw = next(data_iter).to(device, non_blocking=True).float()  # [B,3,H,W] 0..255
        t_data = time.time() - t_data
        proc = raw / 127.5 - 1.0
        f = lr_factor(global_step, args)
        for g in opt_vae.param_groups:
            g["lr"] = base_lrs["vae"] * f
        for g in opt_disc.param_groups:
            g["lr"] = base_lrs["disc"] * f

        vae.train()
        logs = {}
        if e2e:
            with torch.no_grad(), accelerator.autocast():
                zs = [enc.forward_features(dino_input(raw))["x_norm_patchtokens"] for enc in encoders]
            y0 = torch.zeros(raw.shape[0], dtype=torch.long, device=device)
            with accelerator.autocast():
                posterior, z, recon = vae(proc)
                lscale = accelerator.unwrap_model(model).extract_latents_stats()["latents_scale"].detach()
                # Both SiT forwards in train mode (BN stats update between them would otherwise
                # trip autograd's version check); SiT params frozen for the align pass.
                requires_grad(model, False)
                model.train()
                vae_loss, vdict = vae_loss_fn(proc, recon, posterior, global_step, "generator", latents_scale=lscale)
                vae_loss = vae_loss.mean()
                align = model(x=z, y=y0, zs=zs, loss_kwargs={**loss_kwargs, "align_only": True})
                vae_loss = vae_loss + args.vae_align_proj_coeff * align["proj_loss"].mean()
                requires_grad(model, True)
                sit_out = model(x=grad_scale(z, args.vae_diffusion_coeff), y=y0, zs=zs,
                                loss_kwargs={**loss_kwargs, "align_only": False},
                                time_input=align["time_input"], noises=align["noises"])
                sit_loss = sit_out["denoising_loss"].mean() + args.proj_coeff * sit_out["proj_loss"].mean()
            accelerator.backward(vae_loss + sit_loss)
            accelerator.unscale_gradients(optimizer=[opt_vae, opt_sit])
            gn_vae = torch.nn.utils.clip_grad_norm_(vae.parameters(), args.max_grad_norm)
            gn_sit = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt_vae.step()
            opt_sit.step()
            opt_vae.zero_grad(set_to_none=True)
            opt_sit.zero_grad(set_to_none=True)
            logs.update(sit_loss=sit_loss, denoising_loss=sit_out["denoising_loss"].mean(),
                        proj_loss=sit_out["proj_loss"].mean(), vae_align_loss=align["proj_loss"].mean(),
                        grad_norm_sit=gn_sit)
        else:
            with accelerator.autocast():
                posterior, z, recon = vae(proc)
                vae_loss, vdict = vae_loss_fn(proc, recon, posterior, global_step, "generator")
                vae_loss = vae_loss.mean()
            accelerator.backward(vae_loss)
            accelerator.unscale_gradients(optimizer=opt_vae)
            gn_vae = torch.nn.utils.clip_grad_norm_(
                [p for p in vae.parameters() if p.requires_grad], args.max_grad_norm)
            opt_vae.step()
            opt_vae.zero_grad(set_to_none=True)

        d_loss, ddict = vae_loss_fn(proc, recon, posterior, global_step, "discriminator")
        d_loss = d_loss.mean()
        accelerator.backward(d_loss)
        accelerator.unscale_gradients(optimizer=opt_disc)
        gn_disc = torch.nn.utils.clip_grad_norm_(vae_loss_fn.parameters(), args.max_grad_norm)
        opt_disc.step()
        opt_disc.zero_grad(set_to_none=True)
        if e2e:
            update_ema(ema, accelerator.unwrap_model(model), decay=args.ema_decay)
        global_step += 1

        with torch.no_grad():
            mse = F.mse_loss((recon.float().clamp(-1, 1) + 1) / 2, (proc + 1) / 2)
            logs.update(
                vae_loss=vae_loss, rec_loss=vdict["reconstruction_loss"], lpips=vdict["perceptual_loss"],
                latent_reg=vdict["kl_loss"], latent_reg_value=vdict["latent_reg_value"],
                gan_g=vdict["weighted_gan_loss"], d_loss=d_loss, logits_real=ddict["logits_real"],
                logits_fake=ddict["logits_fake"], grad_norm_vae=gn_vae, grad_norm_disc=gn_disc,
                psnr=-10 * torch.log10(mse.clamp_min(1e-10)), z_std=z.float().std(),
                post_std=posterior.std.float().mean(), data_wait=torch.tensor(t_data),
            )
            for k, v in logs.items():
                running[k] = running.get(k, 0.0) + torch.as_tensor(v, device=device).float().detach()

        if global_step % args.log_steps == 0 or global_step == 1:
            keys = sorted(running)
            vec = torch.stack([running[k] for k in keys])
            if dist.is_initialized():
                dist.all_reduce(vec)
            n = (args.log_steps if global_step != 1 else 1) * world
            vals = {k: (vec[i] / n).item() for i, k in enumerate(keys)}
            dt = time.time() - t_log
            vals["img_per_sec"] = args.batch_size * (args.log_steps if global_step != 1 else 1) / dt
            vals["lr_vae"] = base_lrs["vae"] * f
            vals["max_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
            if e2e:
                vals["latents_scale_mean"] = lscale.float().mean().item()
            running = {}
            t_log = time.time()
            log(f"step {global_step} " + " ".join(f"{k}={v:.4g}" for k, v in sorted(vals.items())))
            if use_wandb:
                wandb.log(vals, step=global_step)

        if is_main and (global_step == 1 or global_step % args.sampling_steps == 0):
            with torch.no_grad():
                n = min(4, raw.shape[0])
                vis = torch.cat([(proc[:n] + 1) / 2, (recon[:n].float().clamp(-1, 1) + 1) / 2], 0)
                images = {"recon": make_grid(vis.cpu(), nrow=n)}
                if e2e:
                    uv = accelerator.unwrap_model(vae)
                    ema.eval()
                    h, w = raw.shape[-2] // 8, raw.shape[-1] // 8
                    lat = sample_uncond(ema, (args.num_vis_samples, latent_ch, h, w), 50, device, amp_dtype)
                    st = ema.extract_latents_stats()
                    lat = lat / st["latents_scale"].view(1, -1, 1, 1) + st["latents_bias"].view(1, -1, 1, 1)
                    with accelerator.autocast():
                        samples = uv.decode(lat).sample.float().clamp(-1, 1)
                    images["samples"] = make_grid(((samples + 1) / 2).cpu(), nrow=4)
                if use_wandb:
                    wandb.log({k: wandb.Image(v) for k, v in images.items()}, step=global_step)
                else:
                    from torchvision.utils import save_image
                    os.makedirs(os.path.join(save_dir, "vis"), exist_ok=True)
                    for k, v in images.items():
                        save_image(v, os.path.join(save_dir, "vis", f"{k}_{global_step:07d}.png"))

        if global_step % args.checkpointing_steps == 0 or global_step == args.max_train_steps:
            if is_main:
                ck = state_for_save()
                path = os.path.join(ckpt_dir, f"{global_step:07d}.pt")
                torch.save(ck, path + ".tmp")
                os.replace(path + ".tmp", path)
                log(f"saved {path}")
                old = sorted(glob.glob(os.path.join(ckpt_dir, "[0-9]" * 7 + ".pt")))[:-args.keep_last_ckpts]
                for o in old:
                    if int(os.path.basename(o)[:7]) % args.keep_every_ckpt != 0:
                        os.remove(o)
                if global_step == args.max_train_steps:
                    export_vae(ck)
                del ck
            accelerator.wait_for_everyone()

    log("done")
    if use_wandb:
        wandb.finish()
    accelerator.wait_for_everyone()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["e2e", "decoder"], default="e2e")
    p.add_argument("--exp-name", required=True)
    p.add_argument("--output-dir", default="exps")
    p.add_argument("--report-to", default="wandb", choices=["wandb", "none"])
    p.add_argument("--wandb-project", default="genfirst")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mixed-precision", default="bf16")
    p.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    # data
    p.add_argument("--data-json", required=True)
    p.add_argument("--filelist-dir", default="data/vae_filelists")
    p.add_argument("--side", type=int, default=256, help="bucket area = side**2")
    p.add_argument("--min-area-ratio", type=float, default=0.5625,
                   help="skip images with area < ratio * side**2 (limits upscaling to 4/3)")
    p.add_argument("--max-aspect-ratio", type=float, default=2.0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=512, help="global")
    # VAE
    p.add_argument("--vae", default="flux2", choices=list(vae_models))
    p.add_argument("--vae-pretrained-path", default="")
    p.add_argument("--vae-grad-ckpt", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--loss-cfg-path", default="configs/l1_lpips_entropy_gan.yaml")
    p.add_argument("--disc-start", type=int, default=0)
    p.add_argument("--disc-weight", type=float, default=0.1)
    p.add_argument("--reset-disc", action="store_true", help="do not warm-start the discriminator")
    p.add_argument("--latent-reg", default="entropy", choices=["entropy", "kl"])
    p.add_argument("--latent-reg-weight", type=float, default=0.1)
    p.add_argument("--entropy-aux-kl-weight", type=float, default=1e-6)
    # SiT / REPA
    p.add_argument("--model", default="SiT-XL/2", choices=list(SiT_models))
    p.add_argument("--enc-type", default="dinov2-vit-b")
    p.add_argument("--encoder-depth", type=int, default=8)
    p.add_argument("--proj-coeff", type=float, default=0.5)
    p.add_argument("--vae-align-proj-coeff", type=float, default=1.5)
    p.add_argument("--vae-diffusion-coeff", type=float, default=0.1)
    p.add_argument("--bn-momentum", type=float, default=0.1)
    p.add_argument("--qk-norm", action="store_true")
    p.add_argument("--sit-grad-ckpt", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--path-type", default="linear", choices=["linear", "cosine"])
    p.add_argument("--weighting", default="uniform", choices=["uniform", "lognormal"])
    p.add_argument("--ema-decay", type=float, default=0.9999)
    # optimization
    p.add_argument("--max-train-steps", type=int, default=200000)
    p.add_argument("--learning-rate", type=float, default=1e-4, help="SiT")
    p.add_argument("--vae-learning-rate", type=float, default=1e-4)
    p.add_argument("--disc-learning-rate", type=float, default=1e-4)
    p.add_argument("--lr-schedule", default="constant", choices=["constant", "cosine"],
                   help="applies to VAE + discriminator; SiT stays constant")
    p.add_argument("--min-lr-ratio", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.999)
    p.add_argument("--adam-weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    # bookkeeping
    p.add_argument("--init-ckpt", default="")
    p.add_argument("--resume", default="auto", help="'auto' = latest in exp dir, '' = off, or a path")
    p.add_argument("--checkpointing-steps", type=int, default=10000)
    p.add_argument("--keep-last-ckpts", type=int, default=2, help="older ones are deleted (~12GB each in e2e mode)")
    p.add_argument("--keep-every-ckpt", type=int, default=50000, help="...unless step is a multiple of this")
    p.add_argument("--sampling-steps", type=int, default=5000)
    p.add_argument("--log-steps", type=int, default=50)
    p.add_argument("--num-vis-samples", type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
