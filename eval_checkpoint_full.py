# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

import argparse
import json
import math
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from dataset import CustomINH5Dataset
from eval_utils import OnlineEvaluator
from models.autoencoder import vae_models
from models.sit import SiT_models
from samplers import euler_sampler
from utils import denormalize_latents, preprocess_imgs_vae, load_encoders


def parse_args():
    parser = argparse.ArgumentParser(description="Full offline evaluation for REPA-E checkpoint.")
    parser.add_argument("--exp-path", type=str, required=True, help="Experiment directory containing args.json")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path, e.g. .../0400000.pt")
    parser.add_argument("--eval-data-dir", type=str, default="val_data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--eval-gen-steps", type=int, default=None)
    parser.add_argument("--eval-cfg-scale", type=float, default=None)
    parser.add_argument("--fid-feature", type=int, default=2048, choices=[64, 192, 768, 2048])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--disable-recon", action="store_true")
    parser.add_argument("--disable-gen", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation.")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(args.seed + rank)

    exp_path = Path(args.exp_path)
    with open(exp_path / "args.json", "r") as f:
        train_args = json.load(f)

    eval_gen_steps = args.eval_gen_steps if args.eval_gen_steps is not None else train_args.get("eval_gen_steps", 50)
    eval_cfg_scale = args.eval_cfg_scale if args.eval_cfg_scale is not None else train_args.get("eval_cfg_scale", 1.0)
    path_type = train_args.get("path_type", "linear")

    if train_args["vae"] == "f8d4":
        latent_size = train_args["resolution"] // 8
        in_channels = 4
    elif train_args["vae"] == "f16d32":
        latent_size = train_args["resolution"] // 16
        in_channels = 32
    else:
        raise NotImplementedError(f"Unsupported VAE: {train_args['vae']}")

    encoders, _, _ = load_encoders(train_args["enc_type"], device, train_args["resolution"])
    z_dims = [enc.embed_dim for enc in encoders] if train_args["enc_type"] != "None" else [0]
    del encoders

    model = SiT_models[train_args["model"]](
        input_size=latent_size,
        in_channels=in_channels,
        num_classes=train_args["num_classes"],
        class_dropout_prob=train_args["cfg_prob"],
        z_dims=z_dims,
        encoder_depth=train_args["encoder_depth"],
        bn_momentum=train_args["bn_momentum"],
        fused_attn=train_args.get("fused_attn", True),
        qk_norm=train_args.get("qk_norm", False),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["ema"])
    model.eval()

    vae = vae_models[train_args["vae"]]().to(device)
    vae.load_state_dict(ckpt["vae"])
    vae.eval()

    eval_dataset = CustomINH5Dataset(args.eval_data_dir)
    per_rank_target = math.ceil(args.num_samples / world_size)
    sampler = (
        DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        if distributed else None
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False if sampler is None else None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    evaluator = OnlineEvaluator(
        device=device,
        eval_recon=not args.disable_recon,
        eval_gen=not args.disable_gen,
        fid_feature=args.fid_feature,
    )
    evaluator.reset()

    amp_dtype = torch.float16 if args.mixed_precision == "fp16" else (torch.bfloat16 if args.mixed_precision == "bf16" else None)
    seen = 0
    pbar = tqdm(eval_loader, desc=f"Evaluating (rank {rank})", disable=rank != 0)

    with torch.no_grad():
        for raw_image, labels in pbar:
            raw_image = raw_image.to(device)
            labels = labels.to(device)
            real01 = raw_image.float() / 255.0

            if not args.disable_recon:
                proc = preprocess_imgs_vae(raw_image)
                if amp_dtype is None:
                    _, _, recon = vae(proc)
                else:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        _, _, recon = vae(proc)
                recon01 = (recon.float() + 1) / 2.0
                evaluator.update_recon(real01, recon01)

            if not args.disable_gen:
                zt = torch.randn(labels.shape[0], in_channels, latent_size, latent_size, device=device)
                if amp_dtype is None:
                    samples = euler_sampler(
                        model, zt, labels, num_steps=eval_gen_steps,
                        cfg_scale=eval_cfg_scale, guidance_low=0.0, guidance_high=1.0,
                        path_type=path_type, heun=False,
                    ).to(torch.float32)
                else:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        samples = euler_sampler(
                            model, zt, labels, num_steps=eval_gen_steps,
                            cfg_scale=eval_cfg_scale, guidance_low=0.0, guidance_high=1.0,
                            path_type=path_type, heun=False,
                        ).to(torch.float32)
                ls = model.extract_latents_stats()
                lscale = ls["latents_scale"].view(1, in_channels, 1, 1)
                lbias = ls["latents_bias"].view(1, in_channels, 1, 1)
                img = vae.decode(denormalize_latents(samples, lscale, lbias)).sample
                gen01 = (img.float() + 1) / 2.0
                evaluator.update_gen(real01, gen01)

            seen += raw_image.shape[0]
            if seen >= per_rank_target:
                break

    metrics = evaluator.compute()
    if rank == 0:
        print(json.dumps({"num_samples": min(per_rank_target * world_size, args.num_samples), "metrics": metrics}, indent=2))
    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
