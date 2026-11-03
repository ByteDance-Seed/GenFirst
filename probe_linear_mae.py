# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

"""MAE-protocol ImageNet linear probe for our frozen encoders.

``probe_latent.py`` answers the same question with a closed-form ridge on a
subsampled train split. That is cheap, deterministic and the right tool for
ranking feature spaces against each other, but its absolute numbers land several
points below the published linear-probe protocol, so they cannot be placed next
to MAE's 75.8 or DINOv2's ~86. This script closes that gap by running the recipe
those numbers actually come from (He et al., CVPR 2022, Table 10):

    optimiser        LARS (the MoCo v3 implementation MAE ships), trust coef 1e-3
    base lr          0.1; absolute lr = base_lr * effective_batch / 256
    weight decay     0
    momentum         0.9
    effective batch  16384
    schedule         cosine decay to 0 after 10 linear-warmup epochs
    epochs           90
    data             the full 1.28M-image IN1K train split
    train aug        RandomResizedCrop (the BYOL single-shot variant in MAE's
                     util/crop.py) + horizontal flip
    head             BatchNorm1d(affine=False, eps=1e-6) then Linear
    eval             short side to size/0.875 (bicubic), then centre crop

Three deliberate deviations, each forced and each documented so the paper can
state the protocol honestly:

  * Normalisation is the encoder's own ([-1, 1]), not ImageNet mean/std. The
    frozen stack was trained on its own preprocessing, so feeding it MAE's would
    measure a train/test mismatch instead of the representation. TiTok's authors
    make exactly this substitution when they probe a tokenizer under MAE's recipe
    ("we follow everything in MAE, except that we replace the ViT-Encoder with
    TiTok-Encoder and use global pooling").
  * Input resolution is ours (256 by default), not MAE's 224, because the tap
    points are tied to the autoencoder's grid.
  * Every requested tap point is trained in the same run. The frozen forward
    dominates the cost and is shared across heads, and the heads are independent
    (separate losses, no shared parameters), so N heads cost what one costs.

Tap points and pooling follow ``probe_latent.py`` exactly, so a MAE-protocol
number and a ridge number for the same variant name describe the same feature
space and differ only by the probe.

Usage (scripts/probe_linear_mae.sh wraps this):
    # cost check first: measures throughput and projects the full run, then exits
    torchrun --standalone --nproc_per_node=8 probe_linear_mae.py \
        --imagenet-train-dir=<in1k>/train --imagenet-val-dir=<in1k>/val \
        --vae=qwenvit --vae-ckpt=<ckpt.pt> --bench-steps=30

    # the real thing
    torchrun --standalone --nproc_per_node=8 probe_linear_mae.py \
        --imagenet-train-dir=<in1k>/train --imagenet-val-dir=<in1k>/val \
        --vae=qwenvit --vae-ckpt=<ckpt.pt> --exp-name=mae-probe-nlltrunk

A qwenvit encoder needs transformers>=4.57 for Qwen3-VL; see docs/UND.md.
"""

import argparse
import contextlib
import hashlib
import json
import math
import os
import socket
import time

import numpy as np
import PIL.Image
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset
from torchvision import transforms
from torchvision.transforms import functional as TF

from models.understanding_prior import DEFAULT_QWEN3VL
from probe_latent import load_vae, preprocess_imgs_vae, topk_correct

NUM_CLASSES = 1000
ALL_TARGETS = ("pixel", "latent", "trunk", "merged")
IMG_EXT = (".jpeg", ".jpg", ".png")
IN1K_SIZE = {"train": 1281167, "val": 50000}


def parse_args():
    p = argparse.ArgumentParser()
    # data: original ImageNet JPEGs, ImageFolder layout, sorted synsets == 0-999.
    # Not the c2i parquet shards probe_latent.py reads: MAE's protocol is the full
    # 1.28M-image split with RandomResizedCrop, so it wants the JPEGs.
    p.add_argument("--imagenet-train-dir", type=str, required=True)
    p.add_argument("--imagenet-val-dir", type=str, required=True)
    p.add_argument("--per-class", type=int, default=0,
                   help="cap images per train class (0 = all 1300); a debug knob, "
                        "not a protocol knob -- MAE uses the full split")
    p.add_argument("--allow-partial-imagenet", action="store_true",
                   help="accept a split with missing classes or images; the resulting "
                        "accuracy is not comparable to published numbers")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--crop-pct", type=float, default=0.875,
                   help="eval-time centre-crop fraction, MAE's 224/256")
    p.add_argument("--num-workers", type=int, default=12)
    # optimisation (MAE Table 10 defaults)
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--warmup-epochs", type=int, default=-1,
                   help="-1 keeps MAE's 10/90 fraction, so a shortened run does "
                        "not spend a third of its schedule warming up")
    p.add_argument("--batch-size", type=int, default=512, help="per rank, per micro-step")
    p.add_argument("--eff-batch", type=int, default=16384,
                   help="target effective batch; accumulation is derived from it")
    p.add_argument("--blr", type=float, default=0.1, help="lr = blr * eff_batch / 256")
    p.add_argument("--min-lr", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--trust-coef", type=float, default=0.001)
    # what to probe
    p.add_argument("--variants", type=str,
                   default="latent/full,trunk/pool4,trunk/pool1,merged/pool1",
                   help="comma-separated <target>/<full|poolG>; targets are "
                        f"{ALL_TARGETS}")
    p.add_argument("--max-head-dim", type=int, default=20000,
                   help="reject a variant whose flat dim exceeds this, so a "
                        "260M-parameter head can never be built by accident")
    # vae (flag names match probe_latent.py, whose load_vae handles the non-qwen ones)
    p.add_argument("--vae", type=str, default="qwenvit")
    p.add_argument("--vae-pretrained-path", type=str, default=None,
                   help="for --vae flux: the pretrained FLUX.1 `vae` diffusers folder")
    p.add_argument("--vae-ckpt", type=str, default="",
                   help="a train_e2e.py checkpoint or a save_vae_weights.py extraction; "
                        "empty probes the stock tower (--vae qwenvit only)")
    p.add_argument("--posterior-sample", action="store_true")
    # geometry, used only when the checkpoint is a bare state dict (what
    # save_vae_weights.py writes) and so carries no saved training args
    p.add_argument("--vae-vit-path", type=str, default=DEFAULT_QWEN3VL)
    p.add_argument("--latent-channels", type=int, default=16)
    p.add_argument("--decoder-grid", type=int, default=16,
                   help="what the decoder was built with, not resolution//16")
    p.add_argument("--decoder-width", type=int, default=1024)
    p.add_argument("--decoder-depth", type=int, default=12)
    p.add_argument("--decoder-heads", type=int, default=16)
    # io / control
    p.add_argument("--eval-every", type=int, default=10,
                   help="epochs between val passes; the final epoch always runs")
    p.add_argument("--bench-steps", type=int, default=0,
                   help="time this many train steps, project the full run, exit")
    p.add_argument("--exp-name", type=str, default="mae-probe")
    p.add_argument("--output-dir", type=str, default="exps")
    # Deliberately not under --output-dir: exps/ is often shared storage, and this
    # cache holds *local* paths, so sharing it across machines hands one machine
    # another's staging state. Local disk keeps it per-machine.
    p.add_argument("--index-dir", type=str, default=".cache/imagenet-index",
                   help="where to cache the ImageNet file listing; must be node-local")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
class MaeRandomResizedCrop(transforms.RandomResizedCrop):
    """MAE's util/crop.py: BYOL's single-shot crop sampling, no retry loop.

    torchvision retries up to ten times until the sampled box fits, which biases
    the accepted crops; MAE follows BYOL and instead clamps the box to the image.
    Kept bit-compatible because this is the augmentation the published numbers
    were produced with.
    """

    @staticmethod
    def get_params(img, scale, ratio):
        _, height, width = TF.get_dimensions(img)
        area = height * width
        target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
        log_ratio = torch.log(torch.tensor(ratio))
        aspect = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()
        w = min(int(round(math.sqrt(target_area * aspect))), width)
        h = min(int(round(math.sqrt(target_area / aspect))), height)
        i = torch.randint(0, height - h + 1, size=(1,)).item()
        j = torch.randint(0, width - w + 1, size=(1,)).item()
        return i, j, h, w


def list_imagefolder(root, per_class=0, allow_partial=False):
    """(path, label) pairs; labels are the index of the synset in sorted order.

    A short class list is fatal rather than a warning. Labels are positions in
    the sorted synset list, so a train split missing classes does not merely
    drop them -- it shifts every label above the gap, and train label k then
    names a different synset than val label k. The probe still trains and still
    reports an accuracy; the accuracy is just meaningless. An interrupted
    ``cp -rn`` staging is the usual cause, and it also leaves partially filled
    class dirs, so the image total is checked too.
    """
    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    if len(classes) != NUM_CLASSES and not allow_partial:
        raise RuntimeError(
            f"{root} has {len(classes)} class dirs, expected {NUM_CLASSES}. Labels are "
            f"indices into the sorted synset list, so train and val would disagree on "
            f"what every label means. Finish staging (re-run the STAGE=1 step; cp -rn "
            f"resumes) or pass --allow-partial-imagenet if this is deliberate."
        )
    samples = []
    for label, cls in enumerate(classes):
        cdir = os.path.join(root, cls)
        names = sorted(f for f in os.listdir(cdir) if f.lower().endswith(IMG_EXT))
        if per_class:
            names = names[:per_class]
        samples.extend((os.path.join(cdir, f), label) for f in names)
    if not samples:
        raise FileNotFoundError(f"no images under {root}")
    expect = IN1K_SIZE.get(os.path.basename(root.rstrip("/")))
    if expect and not per_class and len(samples) < 0.99 * expect and not allow_partial:
        raise RuntimeError(
            f"{root} holds {len(samples)} images, expected ~{expect}. Staging looks "
            f"incomplete (class dirs can all exist while still being half filled). "
            f"Re-run the STAGE=1 step, or pass --allow-partial-imagenet."
        )
    return samples


def load_samples(root, per_class, cache_dir, rank, world, is_main, allow_partial=False):
    """List an ImageFolder once and cache it: 1.28M stats over FUSE is not free,
    and without this every rank pays for it on every run."""
    key = hashlib.md5(f"{root}:{per_class}".encode()).hexdigest()[:10]
    path = os.path.join(cache_dir, f"index-{key}.tsv")
    if is_main and not os.path.exists(path):
        os.makedirs(cache_dir, exist_ok=True)
        samples = list_imagefolder(root, per_class, allow_partial)
        # Unique scratch name: if this cache ever does land on shared storage,
        # two jobs rebuilding at once would otherwise interleave writes into one
        # .tmp and each rename the result into place.
        tmp = f"{path}.{socket.gethostname()}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            f.writelines(f"{p}\t{l}\n" for p, l in samples)
        os.replace(tmp, path)
        print(f"    indexed {len(samples)} imgs -> {path}", flush=True)
    if world > 1:
        dist.barrier()
    out = []
    with open(path) as f:
        for line in f:
            p, _, l = line.rstrip("\n").rpartition("\t")
            out.append((p, int(l)))
    # Re-check what was actually loaded. An index cached by an earlier run over a
    # half-staged directory would otherwise sail past the checks in
    # list_imagefolder, which only runs on a cache miss.
    n_lab = len({l for _, l in out})
    if n_lab != NUM_CLASSES and not allow_partial:
        raise RuntimeError(
            f"cached index {path} covers {n_lab} classes, expected {NUM_CLASSES}. It was "
            f"built from an incomplete {root}; delete it and re-index after staging "
            f"finishes, or pass --allow-partial-imagenet."
        )
    expect = IN1K_SIZE.get(os.path.basename(root.rstrip("/")))
    if expect and not per_class and len(out) < 0.99 * expect and not allow_partial:
        raise RuntimeError(
            f"cached index {path} has {len(out)} images, expected ~{expect}. Delete it "
            f"and re-index after staging finishes, or pass --allow-partial-imagenet."
        )
    return out


class JpegDataset(Dataset):
    """ImageNet JPEGs with MAE's transforms, returning uint8 CHW.

    The tensor stays uint8 and un-normalised: the caller applies the encoder's
    own [-1, 1] preprocessing on GPU, which keeps this identical to every other
    path in the repo (``utils.preprocess_imgs_vae``).
    """

    def __init__(self, samples, resolution, train, crop_pct=0.875):
        self.samples = samples
        bic = transforms.InterpolationMode.BICUBIC
        if train:
            self.tf = transforms.Compose([
                MaeRandomResizedCrop(resolution, interpolation=bic),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize(int(round(resolution / crop_pct)), interpolation=bic),
                transforms.CenterCrop(resolution),
            ])
        self.resolution = resolution

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            pil = PIL.Image.open(path).convert("RGB")
            pil = self.tf(pil)
        except Exception:
            pil = PIL.Image.new("RGB", (self.resolution, self.resolution))
        arr = np.asarray(pil, dtype=np.uint8).transpose(2, 0, 1)
        return torch.from_numpy(arr.copy()), torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# probe head, optimiser, schedule -- all straight from MAE
# ---------------------------------------------------------------------------
class LARS(torch.optim.Optimizer):
    """MAE's util/lars.py (from MoCo v3): no scaling or decay on <=1D params."""

    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, trust_coefficient=0.001):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay,
                                      momentum=momentum,
                                      trust_coefficient=trust_coefficient))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad
                if dp is None:
                    continue
                if p.ndim > 1:  # i.e. not a norm gamma/beta or a bias
                    dp = dp.add(p, alpha=g["weight_decay"])
                    pn, un = torch.norm(p), torch.norm(dp)
                    one = torch.ones_like(pn)
                    q = torch.where(pn > 0.,
                                    torch.where(un > 0, g["trust_coefficient"] * pn / un,
                                                one),
                                    one)
                    dp = dp.mul(q)
                st = self.state[p]
                if "mu" not in st:
                    st["mu"] = torch.zeros_like(p)
                st["mu"].mul_(g["momentum"]).add_(dp)
                p.add_(st["mu"], alpha=-g["lr"])


def adjust_lr(optimizer, epoch, args, lr_abs):
    """MAE's util/lr_sched.py: linear warmup then half-cycle cosine."""
    if epoch < args.warmup_epochs:
        lr = lr_abs * epoch / args.warmup_epochs
    else:
        t = (epoch - args.warmup_epochs) / max(1e-8, args.epochs - args.warmup_epochs)
        lr = args.min_lr + (lr_abs - args.min_lr) * 0.5 * (1.0 + math.cos(math.pi * t))
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


class Head(nn.Module):
    """MAE's linear-probe head: a non-affine BN, then a linear classifier."""

    def __init__(self, dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim, affine=False, eps=1e-6)
        self.fc = nn.Linear(dim, NUM_CLASSES)
        nn.init.trunc_normal_(self.fc.weight, std=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        return self.fc(self.bn(x))


class Heads(nn.Module):
    """One head per variant. Independent losses, so DDP sees no shared params."""

    def __init__(self, dims):
        super().__init__()
        # ModuleDict keys cannot contain '.', and variant names carry '/'.
        self.heads = nn.ModuleDict({k.replace("/", "__"): Head(d) for k, d in dims.items()})

    def forward(self, feats):
        return {k.replace("__", "/"): h(feats[k.replace("__", "/")])
                for k, h in self.heads.items()}


# ---------------------------------------------------------------------------
# taps
# ---------------------------------------------------------------------------
def build_encoder(args, device, is_main):
    """(vae, scale, bias), accepting either checkpoint format.

    ``probe_latent.load_vae`` reads the geometry out of a training checkpoint's
    saved args, which a save_vae_weights.py extraction does not have -- it is a
    bare state dict. Both formats are accepted here, with the CLI geometry flags
    as the fallback for the extracted one.
    """
    if args.vae != "qwenvit":
        return load_vae(args, device, is_main)

    from models.qwen_vit_ae import QwenViTAutoencoder

    if not args.vae_ckpt:
        # No checkpoint: the tower is the stock Qwen3-VL one, which is the natural
        # reference for "what joint training cost the encoder". The bottleneck is
        # untrained in this case, so the latent tap would probe a random
        # projection -- refuse it rather than report a meaningless number.
        vae = QwenViTAutoencoder(
            model_path=args.vae_vit_path, latent_channels=args.latent_channels,
            decoder_width=args.decoder_width, decoder_depth=args.decoder_depth,
            decoder_heads=args.decoder_heads, decoder_grid=args.decoder_grid,
            conv_refine=True, freeze_encoder=True,
        ).to(device).eval().requires_grad_(False)
        if is_main:
            print(">>> qwenvit with NO checkpoint: stock Qwen3-VL tower "
                  "(trunk/merged are meaningful, latent is not)", flush=True)
        c = args.latent_channels
        return (vae, torch.ones((c, 1, 1), device=device),
                torch.zeros((c, 1, 1), device=device))

    obj = torch.load(args.vae_ckpt, map_location="cpu", weights_only=False)
    targs = obj.get("args", {}) if isinstance(obj, dict) else {}
    if not isinstance(targs, dict):
        targs = vars(targs)
    sd = obj["vae"] if isinstance(obj, dict) and "vae" in obj else obj
    c = int(targs.get("latent_channels", args.latent_channels))
    vae = QwenViTAutoencoder(
        model_path=targs.get("vae_vit_path") or args.vae_vit_path,
        latent_channels=c,
        decoder_width=int(targs.get("vae_decoder_width", args.decoder_width)),
        decoder_depth=int(targs.get("vae_decoder_depth", args.decoder_depth)),
        decoder_heads=int(targs.get("vae_decoder_heads", args.decoder_heads)),
        decoder_grid=args.decoder_grid,
        conv_refine=bool(targs.get("vae_decoder_conv_refine", True)),
        freeze_encoder=True,
    ).to(device)
    missing, unexpected = vae.load_state_dict(sd, strict=False)

    # The decoder is irrelevant here, but a missing encoder is fatal and silent:
    # strict=False would happily leave the tower at its pretrained-or-random init
    # and the probe would report a number for the wrong weights.
    enc_pref = ("visual.", "pre_norm", "to_mean", "to_logvar")
    critical = [k for k in missing if k.startswith(enc_pref)]
    if critical:
        raise RuntimeError(
            f"{len(critical)} encoder-side tensors missing from {args.vae_ckpt} "
            f"(e.g. {critical[:4]}). This checkpoint does not carry the weights "
            "being probed; point --vae-ckpt at one that does.")
    if is_main:
        n_enc = sum(1 for k in sd if k.startswith(enc_pref))
        print(f">>> qwenvit encoder loaded from {args.vae_ckpt}\n"
              f"    {n_enc} encoder tensors present, step={obj.get('steps', '?') if isinstance(obj, dict) else '?'}, "
              f"missing={len(missing)} unexpected={len(unexpected)} (decoder-side)",
              flush=True)
    vae.eval().requires_grad_(False)
    # Identity: the probe's BatchNorm standardises every dimension anyway, so a
    # per-channel affine on the latent cannot change accuracy.
    scale = torch.ones((c, 1, 1), device=device)
    bias = torch.zeros((c, 1, 1), device=device)
    return vae, scale, bias


def tap_geometry(vae, args, latent_channels):
    """channels and native grid per target, matching probe_latent.py."""
    latent_grid = args.resolution // (16 if args.vae == "f16d32" else 8)
    geo = {"pixel": (3, latent_grid),
           "latent": (int(latent_channels), latent_grid)}
    if hasattr(vae, "vit_forward"):
        # The bottleneck maps one trunk token onto a merge x merge block of
        # latent, and the merger folds a merge x merge block of trunk tokens into
        # one LLM token, so each grid is the previous one divided by merge.
        geo["trunk"] = (vae.vit_dim, latent_grid // vae.merge)
        geo["merged"] = (vae.llm_dim, latent_grid // (vae.merge * vae.merge))
    return geo


@torch.no_grad()
def tap_all(x, targets, vae, scale, bias, posterior_sample, autocast=True):
    """One frozen forward, every requested tap point, as (B, C, g, g) float32.

    ``vit_forward`` already runs the merger, so trunk, merged and latent all come
    out of a single tower pass -- ``probe_latent.py`` pays for two or three.
    """
    amp = (torch.autocast("cuda", dtype=torch.bfloat16) if autocast
           else contextlib.nullcontext())
    out = {}
    if "pixel" in targets:
        out["pixel"] = x
    if not any(t in targets for t in ("latent", "trunk", "merged")):
        return out

    if not hasattr(vae, "vit_forward"):  # FLUX and friends: latent only
        with amp:
            post = vae.encode(x)
        z = (post.sample() if posterior_sample else post.mean).float()
        out["latent"] = (z - bias) * scale
        return out

    with amp:
        feats, vlm = vae.vit_forward(x)
        if "latent" in targets:
            # Mirrors QwenViTAutoencoder.encode; verified against it at startup.
            h = vae.pre_norm(feats)
            mean = vae._unpatchify_latent(vae.to_mean(h))
            if posterior_sample:
                logvar = vae._unpatchify_latent(vae.to_logvar(h)).clamp(-30, 20)
                mean = mean + torch.randn_like(mean) * (0.5 * logvar).exp()
    if "trunk" in targets:
        out["trunk"] = feats.float().permute(0, 3, 1, 2)
    if "merged" in targets:
        gh, gw = vlm.grid
        e = vlm.embeds.float().view(-1, gh, gw, vlm.embeds.shape[-1])
        out["merged"] = e.permute(0, 3, 1, 2)
    if "latent" in targets:
        out["latent"] = (mean.float() - bias) * scale
    return out


def pool_variants(taps, variants):
    """(B, C, g, g) taps -> flat float32 features, one per variant."""
    out = {}
    for name, (target, grid) in variants.items():
        v = taps[target]
        if v.shape[-1] != grid:
            v = F.adaptive_avg_pool2d(v, grid)
        out[name] = v.flatten(1).float()
    return out


@torch.no_grad()
def verify_latent_tap(vae, x, scale, bias):
    """Guard against tap_all drifting from ``encode``'s own latent path.

    Both sides run in fp32: the probe itself runs the tower in bf16, but comparing
    a bf16 tap against an fp32 reference would only measure rounding and would
    have to be given a tolerance loose enough to hide a real mismatch.
    """
    ref = (vae.encode(x).mean.float() - bias) * scale
    got = tap_all(x, ("latent",), vae, scale, bias, False, autocast=False)["latent"]
    rel = (ref - got).abs().max().item() / (ref.abs().max().item() + 1e-8)
    if rel > 1e-4:
        raise RuntimeError(
            f"latent tap disagrees with vae.encode (rel err {rel:.2e}); tap_all "
            "must be updated to match QwenViTAutoencoder.encode")
    return rel


# ---------------------------------------------------------------------------
def setup_dist():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world, local = 0, 1, 0
    torch.cuda.set_device(local)
    return rank, world, local


@torch.no_grad()
def evaluate(loader, heads, variants, targets, vae, scale, bias, args, device, world):
    heads.eval()
    names = list(variants)
    counts = torch.zeros(len(names), 2, device=device)
    total = torch.zeros(1, device=device)
    for imgs, y in loader:
        imgs, y = imgs.to(device, non_blocking=True), y.to(device, non_blocking=True)
        taps = tap_all(preprocess_imgs_vae(imgs), targets, vae, scale, bias, False)
        logits = heads(pool_variants(taps, variants))
        for i, n in enumerate(names):
            hit = topk_correct(logits[n].float(), y)
            counts[i, 0] += hit[1]
            counts[i, 1] += hit[5]
        total += y.numel()
    if world > 1:
        dist.all_reduce(counts)
        dist.all_reduce(total)
    heads.train()
    n = max(1.0, total.item())
    return {nm: {"top1": counts[i, 0].item() / n, "top5": counts[i, 1].item() / n}
            for i, nm in enumerate(names)}, int(n)


def main():
    args = parse_args()
    if args.warmup_epochs < 0:
        args.warmup_epochs = max(1, round(args.epochs * 10 / 90))
    rank, world, local = setup_dist()
    device = torch.device("cuda", local)
    is_main = rank == 0
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.backends.cudnn.benchmark = True

    exp_dir = os.path.join(args.output_dir, args.exp_name)
    if is_main:
        os.makedirs(exp_dir, exist_ok=True)

    # ---- frozen stack ----------------------------------------------------
    if args.vae == "qwenvit" and not args.vae_ckpt and "latent" in args.variants:
        raise ValueError("qwenvit without --vae-ckpt has an untrained bottleneck; "
                         "drop the latent/* variants or pass a checkpoint")
    vae, scale, bias = build_encoder(args, device, is_main)

    # ---- variants --------------------------------------------------------
    geo = tap_geometry(vae, args, scale.numel())
    variants, dims = {}, {}
    for spec in (s for s in args.variants.split(",") if s):
        target, _, pool = spec.partition("/")
        if target not in ALL_TARGETS:
            raise ValueError(f"unknown target '{target}' in '{spec}'; {ALL_TARGETS}")
        if target not in geo:
            raise ValueError(f"target '{target}' needs a qwenvit VAE, got --vae={args.vae}")
        c, native = geo[target]
        g = native if pool in ("full", "") else int(pool.replace("pool", ""))
        if native % g:
            raise ValueError(f"{spec}: grid {g} does not divide native grid {native}")
        dim = c * g * g
        if dim > args.max_head_dim:
            raise ValueError(
                f"{spec} is {dim}d (> --max-head-dim={args.max_head_dim}); pool it "
                f"coarser -- {c}ch at {native}x{native} native")
        variants[spec] = (target, g)
        dims[spec] = dim
    targets = tuple({t for t, _ in variants.values()})
    if is_main:
        print(f">>> variants ({len(variants)}):", flush=True)
        for n, d in dims.items():
            c, native = geo[variants[n][0]]
            print(f"    {n:20s} {d:6d}d  ({c}ch @ {variants[n][1]}x{variants[n][1]}, "
                  f"native {native}x{native})", flush=True)

    # ---- data ------------------------------------------------------------
    if is_main:
        print(">>> indexing ImageNet (minutes over FUSE the first time, cached after)",
              flush=True)
    idx_dir = args.index_dir
    train_ds = JpegDataset(
        load_samples(args.imagenet_train_dir, args.per_class, idx_dir, rank, world, is_main,
                     args.allow_partial_imagenet),
        args.resolution, train=True)
    val_all = JpegDataset(
        load_samples(args.imagenet_val_dir, 0, idx_dir, rank, world, is_main,
                     args.allow_partial_imagenet),
        args.resolution, train=False, crop_pct=args.crop_pct)
    # Strided val shards avoid DistributedSampler's duplicate padding, so the
    # reduced counts are over exactly the 50k images and not 50k + world.
    val_ds = Subset(val_all, list(range(rank, len(val_all), world)))
    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True,
                                 drop_last=True, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True, persistent_workers=args.num_workers > 0,
                              prefetch_factor=4 if args.num_workers > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    accum = max(1, round(args.eff_batch / (args.batch_size * world)))
    eff = args.batch_size * world * accum
    lr_abs = args.blr * eff / 256
    micro_per_epoch = len(train_loader)
    if is_main:
        print(f">>> train {len(train_ds)} imgs | val {len(val_all)} imgs\n"
              f">>> eff batch {eff} (= {args.batch_size} x {world} ranks x {accum} accum)"
              f" | abs lr {lr_abs:.4g} (blr {args.blr})\n"
              f">>> {micro_per_epoch} micro-steps/epoch, {args.epochs} epochs",
              flush=True)
        if eff != args.eff_batch:
            print(f"    note: effective batch {eff} != requested {args.eff_batch}; "
                  f"lr was rescaled to match what is actually used", flush=True)

    # ---- heads -----------------------------------------------------------
    heads = Heads(dims).to(device)
    heads_raw = heads
    if world > 1:
        heads = nn.parallel.DistributedDataParallel(heads, device_ids=[local])
    opt = LARS(heads_raw.parameters(), lr=lr_abs, weight_decay=args.weight_decay,
               momentum=args.momentum, trust_coefficient=args.trust_coef)
    criterion = nn.CrossEntropyLoss()

    # ---- sanity: the replicated latent path must match encode() ----------
    if "latent" in targets and hasattr(vae, "vit_forward"):
        probe_x = torch.zeros(2, 3, args.resolution, args.resolution, device=device)
        probe_x.uniform_(-1, 1)
        rel = verify_latent_tap(vae, probe_x, scale, bias)
        if is_main:
            print(f">>> latent tap matches vae.encode (rel err {rel:.2e})", flush=True)

    # ---- train -----------------------------------------------------------
    history, t_start = [], time.time()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        heads.train()
        t0, seen, loss_sum, nstep = time.time(), 0, 0.0, 0
        opt.zero_grad(set_to_none=True)
        for it, (imgs, y) in enumerate(train_loader):
            if it % accum == 0:
                adjust_lr(opt, epoch + it / micro_per_epoch, args, lr_abs)
            imgs = imgs.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            taps = tap_all(preprocess_imgs_vae(imgs), targets, vae, scale, bias,
                           args.posterior_sample)
            feats = pool_variants(taps, variants)

            last = (it % accum) == (accum - 1)
            ctx = (heads.no_sync() if (world > 1 and not last)
                   else contextlib.nullcontext())
            with ctx:
                logits = heads(feats)
                # Independent heads, so the sum's gradient is each head's own.
                loss = sum(criterion(logits[n], y) for n in variants) / accum
                loss.backward()
            if last:
                opt.step()
                opt.zero_grad(set_to_none=True)
            loss_sum += loss.item() * accum
            nstep += 1
            seen += imgs.shape[0]

            if args.bench_steps and it + 1 >= args.bench_steps:
                dt = time.time() - t0
                ips = seen * world / dt
                total_imgs = len(train_ds) * args.epochs
                if is_main:
                    print(f"\n=== bench ===\n  {ips:.0f} img/s across {world} ranks "
                          f"({ips / world:.0f}/rank)\n"
                          f"  one epoch: {len(train_ds) / ips / 60:.1f} min\n"
                          f"  {args.epochs} epochs: {total_imgs / ips / 3600:.2f} h "
                          f"(+ val passes)", flush=True)
                if world > 1:
                    dist.destroy_process_group()
                return

            if is_main and (it + 1) % (accum * 20) == 0:
                ips = seen * world / (time.time() - t0)
                print(f"    ep{epoch} {it + 1}/{micro_per_epoch} "
                      f"loss={loss_sum / nstep:.3f} lr={opt.param_groups[0]['lr']:.3g} "
                      f"{ips:.0f} img/s", flush=True)

        do_eval = ((epoch + 1) % args.eval_every == 0) or (epoch + 1 == args.epochs)
        line = (f">>> epoch {epoch} done in {(time.time() - t0) / 60:.1f} min "
                f"(loss {loss_sum / max(1, nstep):.3f})")
        if do_eval:
            accs, n_val = evaluate(val_loader, heads, variants, targets, vae, scale,
                                   bias, args, device, world)
            if is_main:
                print(line, flush=True)
                for n in variants:
                    print(f"    {n:20s} top1={accs[n]['top1'] * 100:6.2f} "
                          f"top5={accs[n]['top5'] * 100:6.2f}", flush=True)
                history.append({"epoch": epoch, "n_val": n_val,
                                "acc": {n: accs[n] for n in variants}})
                with open(os.path.join(exp_dir, "mae_probe.json"), "w") as f:
                    json.dump({"args": vars(args), "dims": dims,
                               "eff_batch": eff, "lr": lr_abs,
                               "history": history}, f, indent=2)
        elif is_main:
            print(line, flush=True)

    if is_main:
        print(f"\n=== MAE-protocol linear probe, {args.epochs} epochs, "
              f"{len(train_ds)} train imgs ===")
        final = history[-1]["acc"] if history else {}
        for n in variants:
            best = max((h["acc"][n]["top1"] for h in history), default=0.0)
            print(f"  {n:20s} final top1={final.get(n, {}).get('top1', 0) * 100:6.2f} "
                  f"top5={final.get(n, {}).get('top5', 0) * 100:6.2f}  "
                  f"(best {best * 100:.2f})")
        print(f">>> {(time.time() - t_start) / 3600:.2f} h total; wrote "
              f"{os.path.join(exp_dir, 'mae_probe.json')}", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
