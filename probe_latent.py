# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

"""ImageNet linear probe: is a latent (or a projector's output) linearly semantic?

The caption NLL / PMI that ``--und-type nll`` reports says how much the frozen
LLM can *read*, but not whether the information sits in a smooth semantic
subspace or in a private code that only that particular LLM decodes. A linear
probe answers the second question, because a linear map cannot decode a code.
It is also the only understanding read-out that applies to every configuration,
including ``--und-type siglip`` and runs with no understanding loss at all.

Four probe targets, and the whole point is again their spread:

    pixel      RGB downsampled to the latent's grid -> the floor. A linear map on
               raw pixels is already well above chance on ImageNet, so any latent
               number is meaningless without this.
    latent     the normalised VAE latent -> what the diffusion model consumes.
    trunk      our own encoder's ViT features, tapped before the bottleneck (see
               ``QwenViTAutoencoder.vit_features``) -> splits ``latent`` into its
               two causes: what joint training did to the trunk, and what the
               16-channel bottleneck then throws away. Since ``latent`` defaults
               to the posterior mean, that second part is compression alone,
               with sampling noise excluded unless --posterior-sample. At 256px
               this stores 1024ch at 4x4, i.e. the same 16384 dimensions the
               latent gets, so a trunk-vs-latent gap is not a capacity artefact.
               Probing it on a --vae-freeze-encoder checkpoint gives the
               *untouched* trunk for free.
    merged     our own tower's output *after* Qwen's merger, i.e. bit-for-bit the
               image tokens its language model consumes, and the exact place the
               --und-tap trunk caption NLL is applied. ``trunk`` is one frozen
               merger away from this, so the pair separates "the tower still holds
               the semantics" from "the tower still speaks the language model's
               dialect". Compare ``merged/pool1`` against ``vit/pool1``: same
               layer, same 2048 dims, same global mean -- only the tower weights
               differ, so the gap is exactly what joint training cost the LLM's
               input. A --vae-freeze-encoder checkpoint gives the untouched tower.
    projector  the trained projector's output tokens -> what the LLM consumes.
               Needs a --und-type nll checkpoint, which is where a projector is
               trained; see --projector-ckpt.
    vit        Qwen3-VL's real ViT features, globally pooled -> the ceiling.

With the VAE frozen the projector is a deterministic function of the latent, so
it cannot *add* information; ``probe(projector) - probe(latent)`` is therefore
pure gain in linear accessibility. That gap is the main read-out:

  * projector >> latent: the projector learned to linearly expose semantics that
    were entangled in the latent. Real semantic decoding, and a good warm start.
  * projector ~ latent: whatever bought the PMI is invisible to a linear map --
    consistent with a private code, so treat the PMI gain with suspicion.

A secondary read-out is the accuracy-versus-spatial-frequency curve. ``pool``
variants average the feature map down to a g x g grid (the conventional
protocol, but coarser grids also shrink the dimensionality, which confounds it);
``smooth`` variants average down to g x g and upsample straight back, holding the
dimensionality fixed so only the spatial frequency content changes. Semantics
survives smoothing; a high-frequency private code does not.

Probing is a least-squares (ridge) classifier on cached features: deterministic,
no learning-rate to tune, and the lambda sweep is a few Cholesky solves. Absolute
numbers land below published linear-probe results (fewer images per class, and
ridge rather than logistic), which is fine -- only the comparison between feature
spaces is being asked for.

Usage (see scripts/probe_latent.sh for a parametrized wrapper):
    # a trained latent vs the pixel floor (no VLM loaded, fast)
    torchrun --standalone --nproc_per_node=8 probe_latent.py \
        --imagenet-train-dir=/data/imagenet/c2i-256 \
        --imagenet-val-dir=/data/imagenet/val \
        --label-map=/data/imagenet/classnames.txt \
        --vae=flux --vae-pretrained-path=/path/to/FLUX.1-dev/vae \
        --vae-ckpt=exps/<run>/checkpoints/0400000.pt \
        --targets=latent,pixel

    # add the trained projector and the ViT ceiling (--und-type nll runs)
    torchrun --standalone --nproc_per_node=8 probe_latent.py \
        ... --targets=latent,pixel,projector,vit \
        --projector-ckpt=exps/<run>/checkpoints/0400000.pt

    # re-run the probe on cached features without re-extracting
    python probe_latent.py --skip-extract --targets=latent,pixel
"""

import argparse
import glob
import json
import math
import os
import time

from models.understanding_prior import UnderstandingPrior, DEFAULT_QWEN3VL

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataset import ImageFolderDataset, ParquetClassDataset
from models.autoencoder import FluxVAE, vae_models


NUM_CLASSES = 1000
ALL_TARGETS = ("pixel", "latent", "trunk", "merged", "projector", "vit")


def parse_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--imagenet-train-dir", type=str, required=True,
                   help="c2i parquet shards; labels are class-name strings")
    p.add_argument("--label-map", type=str, required=True,
                   help="one class name per line in canonical 0-999 order")
    p.add_argument("--parquet-rows-per-file", type=int, default=16)
    p.add_argument("--imagenet-val-dir", type=str, required=True,
                   help="root/<synset>/*.JPEG; sorted synsets == canonical ids")
    p.add_argument("--n-train", type=int, default=100000,
                   help="100k is the reported protocol; keep it well above the "
                        "largest probe dimension (~2x) if you change it")
    p.add_argument("--n-val", type=int, default=10000)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64, help="per rank")
    p.add_argument("--num-workers", type=int, default=8)
    # what to probe
    p.add_argument("--targets", type=str, default="latent,pixel")
    # vae
    p.add_argument("--vae", type=str, default="flux",
                   choices=list(vae_models.keys()) + ["qwenvit"])
    p.add_argument("--vae-pretrained-path", type=str, default=None,
                   help="for --vae flux: the pretrained FLUX.1 `vae` diffusers folder")
    p.add_argument("--vae-ckpt", type=str, default="",
                   help="optional end-to-end tuned VAE checkpoint to probe instead of "
                        "the pretrained one")
    p.add_argument("--posterior-sample", action="store_true",
                   help="probe a posterior sample instead of the mean (noisier)")
    # projector / vit
    p.add_argument("--vlm-path", type=str, default=DEFAULT_QWEN3VL)
    p.add_argument("--projector-ckpt", type=str, default="",
                   help="a train_e2e.py checkpoint trained with --und-type nll; its "
                        "saved args define the projector geometry and its 'und' entry "
                        "holds the weights")
    p.add_argument("--latent-patch", type=int, default=2,
                   help="only used when --projector-ckpt carries no args")
    # feature storage / variants
    p.add_argument("--max-feat-dim", type=int, default=20000,
                   help="cap on stored channels*grid^2; the stored grid is halved "
                        "until it fits, and coarser variants are derived from it")
    p.add_argument("--pool-grids", type=str, default="full,4,2,1")
    p.add_argument("--smooth-grids", type=str, default="4,2",
                   help="dimensionality-preserving frequency control; '' to skip")
    # probe
    p.add_argument("--lambdas", type=str, default="1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10",
                   help="ridge strengths, relative to the mean Gram diagonal")
    p.add_argument("--dev-stride", type=int, default=20,
                   help="every k-th train sample is held out to pick lambda; the "
                        "train split is class-ordered, so striding stays balanced")
    p.add_argument("--solve-chunk", type=int, default=8192)
    # io
    p.add_argument("--cache-dir", type=str, default="")
    p.add_argument("--skip-extract", action="store_true",
                   help="reuse cached features (single process is enough)")
    p.add_argument("--out-json", type=str, default="")
    p.add_argument("--exp-name", type=str, default="probe-flux-latent")
    p.add_argument("--output-dir", type=str, default="exps")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def setup_dist():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world, local = 0, 1, 0
    torch.cuda.set_device(local)
    return rank, world, local


def preprocess_imgs_vae(imgs):
    """uint8 [0, 255] -> float [-1, 1]; mirrors utils.preprocess_imgs_vae."""
    return imgs.float() / 127.5 - 1.0


def stored_grid(channels, native, max_dim):
    """Coarsest-but-largest grid whose flat dimension fits the storage budget."""
    g = native
    while g > 1 and channels * g * g > max_dim:
        g //= 2
    return g


def divisor_grids(stored, requested):
    """Requested grids that are actually derivable from the stored one."""
    out = []
    for r in requested:
        g = stored if r == "full" else int(r)
        if g <= stored and stored % g == 0 and g not in out:
            out.append(g)
    return out


# ---------------------------------------------------------------------------
# feature extraction
# ---------------------------------------------------------------------------
def build_vit_batch(image_processor, images_uint8):
    arrs = [img.permute(1, 2, 0).cpu().numpy() for img in images_uint8]
    out = image_processor(images=arrs, return_tensors="pt")
    return out["pixel_values"], out["image_grid_thw"]


def load_vae(args, device, is_main):
    if args.vae == "qwenvit":
        # Geometry comes from the checkpoint's own saved args rather than from
        # flags here, so a probe can never silently be run against a differently
        # shaped decoder than the one that was trained.
        from models.qwen_vit_ae import QwenViTAutoencoder

        ckpt = torch.load(args.vae_ckpt, map_location="cpu")
        targs = ckpt.get("args", {})
        if not isinstance(targs, dict):
            targs = vars(targs)
        # save_vae_weights.py drops these, so an extracted VAE reaches the tower
        # path as None and the error surfaces as a 401 from huggingface.co/None.
        if not targs.get("vae_vit_path"):
            raise ValueError(
                f"{args.vae_ckpt} carries no vae_vit_path in its saved args. "
                f"--vae-ckpt wants a train_e2e.py checkpoint here, not an "
                f"extracted VAE, since the decoder geometry is read from it."
            )
        vae = QwenViTAutoencoder(
            model_path=targs.get("vae_vit_path"),
            latent_channels=targs.get("latent_channels", 16),
            decoder_width=targs.get("vae_decoder_width", 1024),
            decoder_depth=targs.get("vae_decoder_depth", 12),
            decoder_heads=targs.get("vae_decoder_heads", 16),
            decoder_grid=args.resolution // 16,
            conv_refine=targs.get("vae_decoder_conv_refine", True),
            freeze_encoder=True,
        ).to(device)
        missing, unexpected = vae.load_state_dict(ckpt["vae"], strict=False)
        step = ckpt.get("steps")
        del ckpt
        # Identity, not the trained BN statistics. The probe standardises every
        # feature dimension before the ridge solve, so a per-channel affine on
        # the latent cannot change the accuracy -- carrying the MMDiT's running
        # stats here would add a failure mode for no effect.
        c = int(targs.get("latent_channels", 16))
        scale = torch.ones((c, 1, 1), device=device)
        bias = torch.zeros((c, 1, 1), device=device)
        if is_main:
            print(f">>> probing qwenvit step={step} from {args.vae_ckpt} "
                  f"(missing={len(missing)} unexpected={len(unexpected)})", flush=True)
        vae.eval().requires_grad_(False)
        return vae, scale, bias

    if args.vae == "flux":
        vae = vae_models["flux"](pretrained_path=args.vae_pretrained_path).to(device)
        scale = torch.full((16, 1, 1), float(FluxVAE.scaling_factor), device=device)
        bias = torch.full((16, 1, 1), float(FluxVAE.shift_factor), device=device)
    else:
        vae = vae_models[args.vae]().to(device)
        stats = torch.load(args.vae_ckpt.replace(".pt", "-latents-stats.pt"))
        scale = stats["latents_scale"].squeeze().view(-1, 1, 1).to(device)
        bias = stats["latents_bias"].squeeze().view(-1, 1, 1).to(device)
    if args.vae_ckpt and os.path.isfile(args.vae_ckpt):
        ckpt = torch.load(args.vae_ckpt, map_location="cpu")
        if isinstance(ckpt, dict) and "vae" in ckpt and "vae.encoder.conv_in.weight" not in ckpt:
            ckpt = ckpt["vae"]
        vae.load_state_dict(ckpt, strict=False)
        del ckpt
        if is_main:
            print(f">>> probing e2e VAE checkpoint: {args.vae_ckpt}", flush=True)
    vae.eval().requires_grad_(False)
    return vae, scale, bias


def build_prior(args, targets, latent_channels, device, is_main):
    """Qwen3-VL wrapper, only when the projector or ViT target needs it."""
    pargs = {}
    if args.projector_ckpt:
        ckpt = torch.load(args.projector_ckpt, map_location="cpu")
        # train_e2e.py stores its whole argparse namespace under "args"; the
        # understanding knobs are prefixed there, so translate them into this
        # builder's names. `via_merger` has to come along: with the merger the
        # projector's output width is the tower's, without it the language model's,
        # so getting it wrong makes load_state_dict fail on a shape mismatch.
        saved = ckpt.get("args", {})
        saved = saved if isinstance(saved, dict) else vars(saved)
        pargs = {
            "latent_patch": saved.get("und_latent_patch", args.latent_patch),
            "projector_mode": saved.get("und_projector_mode", "linear"),
            "projector_width": saved.get("und_width", 1024),
            "projector_depth": saved.get("und_depth", 4),
            "projector_heads": saved.get("und_heads", 8),
            "via_merger": saved.get("und_via_merger", True),
        }
    prior = UnderstandingPrior(
        model_path=args.vlm_path,
        latent_channels=latent_channels,
        latent_patch=int(pargs.get("latent_patch", args.latent_patch)),
        projector_mode=pargs.get("projector_mode", "linear"),
        projector_width=int(pargs.get("projector_width", 1024)),
        projector_depth=int(pargs.get("projector_depth", 4)),
        projector_heads=int(pargs.get("projector_heads", 8)),
        via_merger=bool(pargs.get("via_merger", False)),
        load_vision="vit" in targets,
        device=device,
        pmi_mode="off",
        grad_checkpointing=False,
    )
    if args.projector_ckpt:
        if "und" not in ckpt:
            raise KeyError(
                f"{args.projector_ckpt} has no 'und' entry: --projector-ckpt expects a "
                f"train_e2e.py checkpoint trained with --und-type nll"
            )
        prior.projector.load_state_dict(ckpt["und"])
        if is_main:
            print(f">>> loaded projector from {args.projector_ckpt} "
                  f"(step {ckpt.get('steps', '?')}, mode={pargs.get('projector_mode')}, "
                  f"via_merger={pargs.get('via_merger')})",
                  flush=True)
        del ckpt
    elif "projector" in targets:
        raise ValueError("--targets includes 'projector' but --projector-ckpt is empty")
    prior.projector.eval()
    return prior


@torch.no_grad()
def extract_split(loader, targets, specs, vae, scale, bias, prior, image_processor,
                  args, device, rank, tag):
    """Run one split through the frozen stack, returning pooled CPU features."""
    chunks = {t: [] for t in targets}
    labels = []
    t0, seen = time.time(), 0
    for images, y in loader:
        images = images.to(device, non_blocking=True)
        labels.append(y.clone())

        x = preprocess_imgs_vae(images)
        z = None
        if "latent" in targets or "projector" in targets:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                posterior = vae.encode(x)
            z = (posterior.sample() if args.posterior_sample else posterior.mean).float()
            z = (z - bias) * scale

        for t in targets:
            if t == "pixel":
                f = F.adaptive_avg_pool2d(x, specs[t]["grid"])
            elif t == "latent":
                f = F.adaptive_avg_pool2d(z, specs[t]["grid"])
            elif t == "trunk":
                # The tap point is before pre_norm and the bottleneck heads, so
                # this is the trunk's own representation. Skipping pre_norm costs
                # nothing: it is a LayerNorm, and the probe standardises every
                # dimension before the solve anyway.
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    h = vae.vit_features(x)
                f = F.adaptive_avg_pool2d(
                    h.float().permute(0, 3, 1, 2), specs[t]["grid"]
                )
            elif t == "merged":
                # vit_forward runs the merger anyway, so this is the same pass
                # ``trunk`` uses. embeds arrive merge-block-major over the token
                # grid, which is raster order on that grid -- the inverse of
                # _to_raster's split -- so the reshape below is the layout Qwen
                # itself assumes. pool1 does not depend on it either way.
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    vlm = vae.vit_forward(x)[1]
                gh, gw = vlm.grid
                f = vlm.embeds.float().view(-1, gh, gw, vlm.embeds.shape[-1])
                f = F.adaptive_avg_pool2d(f.permute(0, 3, 1, 2), specs[t]["grid"])
            elif t == "projector":
                embeds, (hl, wl) = prior.latent_to_embeds(z)
                f = embeds.float().view(-1, hl, wl, embeds.shape[-1]).permute(0, 3, 1, 2)
                f = F.adaptive_avg_pool2d(f, specs[t]["grid"])
            elif t == "vit":
                pixel_values, grid = build_vit_batch(image_processor, images)
                vis = prior.visual(pixel_values.to(device, prior.dtype),
                                   grid_thw=grid.to(device))
                vis = vis[0] if isinstance(vis, (tuple, list)) else vis
                # Global mean only: the merged-token layout is a Qwen internal
                # detail, and the mean is the one pooling that cannot depend on it.
                f = vis.float().view(images.shape[0], -1, vis.shape[-1]).mean(1)
                f = f.view(f.shape[0], -1, 1, 1)
            chunks[t].append(f.flatten(1).half().cpu())

        seen += images.shape[0]
        if rank == 0 and seen % (args.batch_size * 50) == 0:
            ips = seen / (time.time() - t0)
            print(f"    [{tag}] {seen} imgs, {ips:.1f} img/s/rank", flush=True)

    return ({t: torch.cat(v) for t, v in chunks.items()}, torch.cat(labels))


# ---------------------------------------------------------------------------
# ridge probe over streamed feature variants
# ---------------------------------------------------------------------------
def make_variant(channels, grid, kind, g):
    """Chunk transform: stored flat feature -> the variant's flat feature."""
    def fn(x):
        v = x.float().view(-1, channels, grid, grid)
        if g != grid:
            v = F.adaptive_avg_pool2d(v, g)
            if kind == "smooth":
                v = F.interpolate(v, size=(grid, grid), mode="nearest")
        return v.flatten(1)
    return fn


def _stream(x, y, chunk, device, transform, drop_stride=None):
    """Yield transformed (features, labels) chunks from a CPU tensor.

    ``drop_stride=k`` skips every k-th row, which is how the fit split excludes
    the dev rows without materialising a copy of the (large) train tensor.
    """
    n = x.shape[0]
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        xb = x[s:e].to(device, non_blocking=True)
        yb = y[s:e].to(device, non_blocking=True)
        if drop_stride is not None:
            keep = torch.arange(s, e, device=device) % drop_stride != 0
            if not bool(keep.any()):
                continue
            xb, yb = xb[keep], yb[keep]
        yield transform(xb), yb


def topk_correct(scores, y, ks=(1, 5)):
    top = scores.topk(max(ks), dim=1).indices
    hit = top == y[:, None]
    return {k: int(hit[:, :k].any(1).sum().item()) for k in ks}


def ridge_probe(train_x, train_y, val_x, val_y, transform, lambdas, dev_stride,
                chunk, device):
    """Least-squares classifier on standardised features; returns val top-1/top-5."""
    stream_fit = lambda: _stream(train_x, train_y, chunk, device, transform,
                                 drop_stride=dev_stride)
    # The lambda sweep touches the dev rows once per lambda, so materialise them
    # instead of streaming (and discarding) the whole train tensor each time.
    dev_sel = torch.arange(train_x.shape[0]) % dev_stride == 0
    dev_x, dev_y = train_x[dev_sel], train_y[dev_sel]

    # pass 1: standardisation stats on the fit rows only
    d = None
    for xb, _ in stream_fit():
        if d is None:
            d = xb.shape[1]
            ssum = torch.zeros(d, device=device, dtype=torch.float64)
            ssq = torch.zeros(d, device=device, dtype=torch.float64)
            n_fit = 0
        ssum += xb.sum(0).double()
        ssq += xb.pow(2).sum(0).double()
        n_fit += xb.shape[0]
    mu = (ssum / n_fit).float()
    sd = (ssq / n_fit - (ssum / n_fit) ** 2).clamp(min=1e-10).sqrt().float()

    # pass 2: Gram and cross-moment
    gram = torch.zeros(d, d, device=device)
    xty = torch.zeros(d, NUM_CLASSES, device=device)
    prior_counts = torch.zeros(NUM_CLASSES, device=device)
    for xb, yb in stream_fit():
        xb = (xb - mu) / sd
        gram += xb.T @ xb
        xty.index_add_(1, yb, xb.T)
        prior_counts.index_add_(0, yb, torch.ones_like(yb, dtype=torch.float32))
    ybar = prior_counts / n_fit

    def evaluate(w, x, y):
        corr, tot = {1: 0, 5: 0}, 0
        for xb, yb in _stream(x, y, chunk, device, transform):
            scores = ((xb - mu) / sd) @ w + ybar
            for k, c in topk_correct(scores, yb).items():
                corr[k] += c
            tot += yb.shape[0]
        return {k: c / max(tot, 1) for k, c in corr.items()}, tot

    # lambda sweep on the strided dev rows
    diag = gram.diagonal().mean().item()
    best = (-1.0, None, None)
    for lam in lambdas:
        a = gram.clone()
        a.diagonal().add_(lam * diag)
        try:
            w = torch.cholesky_solve(xty, torch.linalg.cholesky(a))
        except Exception:
            a.diagonal().add_(1e-3 * diag)
            w = torch.linalg.solve(a, xty)
        acc, _ = evaluate(w, dev_x, dev_y)
        if acc[1] > best[0]:
            best = (acc[1], lam, w)
    _, lam_best, w_best = best

    val_acc, n_val_used = evaluate(w_best, val_x, val_y)
    return {
        "dim": int(d),
        "n_fit": int(n_fit),
        "n_val": int(n_val_used),
        "lambda": lam_best,
        "dev_top1": best[0],
        "top1": val_acc[1],
        "top5": val_acc[5],
    }


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    rank, world, local = setup_dist()
    device = torch.device("cuda", local)
    is_main = rank == 0
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    targets = [t for t in args.targets.split(",") if t]
    bad = [t for t in targets if t not in ALL_TARGETS]
    if bad:
        raise ValueError(f"unknown targets {bad}; choose from {ALL_TARGETS}")

    exp_dir = os.path.join(args.output_dir, args.exp_name)
    cache_dir = args.cache_dir or os.path.join(exp_dir, "features")
    if is_main:
        os.makedirs(cache_dir, exist_ok=True)
        print(f">>> probe exp: {exp_dir}\n>>> targets: {targets}", flush=True)

    # ---- extract ----------------------------------------------------------
    if not args.skip_extract:
        vae, scale, bias = load_vae(args, device, is_main)
        latent_channels = scale.numel()
        latent_grid = args.resolution // 8 if args.vae != "f16d32" else args.resolution // 16

        prior, image_processor = None, None
        if "projector" in targets or "vit" in targets:
            prior = build_prior(args, targets, latent_channels, device, is_main)
            if "vit" in targets:
                from transformers import AutoImageProcessor
                image_processor = AutoImageProcessor.from_pretrained(args.vlm_path)

        # storage geometry per target
        specs = {}
        for t in targets:
            if t == "pixel":
                c, native = 3, latent_grid
            elif t == "latent":
                c, native = latent_channels, latent_grid
            elif t == "trunk":
                if not hasattr(vae, "vit_features"):
                    raise ValueError(f"--targets trunk needs --vae qwenvit, got {args.vae}")
                # The bottleneck maps one trunk token onto a merge x merge block
                # of latent, so the trunk grid is the latent grid over merge.
                c, native = vae.vit_dim, latent_grid // vae.merge
            elif t == "merged":
                if not hasattr(vae, "vit_forward"):
                    raise ValueError(f"--targets merged needs --vae qwenvit, got {args.vae}")
                # The merger folds a merge x merge block of trunk tokens into one
                # token of llm_dim, so its grid is the trunk grid over merge again.
                c, native = vae.llm_dim, latent_grid // (vae.merge * vae.merge)
            elif t == "projector":
                c = prior.hidden_size
                native = latent_grid // prior.latent_patch
            else:
                c, native = prior.hidden_size, 1
            g = stored_grid(c, native, args.max_feat_dim)
            specs[t] = {"channels": c, "grid": g, "native": native, "dim": c * g * g}
        if is_main:
            for t, s in specs.items():
                note = "" if s["grid"] == s["native"] else \
                    f" (pooled from {s['native']}x{s['native']} to fit --max-feat-dim)"
                print(f"    {t}: {s['channels']}ch @ {s['grid']}x{s['grid']} "
                      f"= {s['dim']}d{note}", flush=True)

        # ---- datasets -----------------------------------------------------
        files = sorted(glob.glob(os.path.join(args.imagenet_train_dir, "*.parquet")))
        if not files:
            raise FileNotFoundError(f"no parquet under {args.imagenet_train_dir}")
        # Shards are class-ordered (shard 0 is all tench), so take an even stride
        # across the whole range rather than a prefix.
        n_files = max(1, min(len(files),
                             math.ceil(args.n_train / args.parquet_rows_per_file)))
        stride = max(1, len(files) // n_files)
        sel = files[::stride][:n_files]
        train_ds = ParquetClassDataset(
            sel, resolution=args.resolution, label_column="class_label",
            label_map_file=args.label_map, rows_per_file=args.parquet_rows_per_file,
        )
        val_ds = ImageFolderDataset(args.imagenet_val_dir, resolution=args.resolution)
        if args.n_val < len(val_ds):
            step = len(val_ds) // args.n_val
            val_ds = Subset(val_ds, list(range(0, len(val_ds), step))[: args.n_val])
        if is_main:
            print(f">>> train: {len(train_ds)} imgs over {len(sel)} shards "
                  f"(stride {stride}) | val: {len(val_ds)} imgs", flush=True)

        results_meta = {"specs": specs}
        for tag, ds in (("train", train_ds), ("val", val_ds)):
            # Strided shards keep every rank's slice class-balanced.
            shard = Subset(ds, list(range(rank, len(ds), world)))
            loader = DataLoader(shard, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True,
                                drop_last=False)
            feats, labels = extract_split(loader, targets, specs, vae, scale, bias,
                                          prior, image_processor, args, device, rank, tag)
            torch.save({"feats": feats, "labels": labels, "specs": specs},
                       os.path.join(cache_dir, f"{tag}-rank{rank}.pt"))
            if is_main:
                print(f">>> extracted {tag}: {labels.numel()} imgs on rank 0", flush=True)
        if is_main:
            with open(os.path.join(cache_dir, "meta.json"), "w") as f:
                json.dump({"specs": specs, "world": world, "args": vars(args)}, f, indent=2)
        if world > 1:
            dist.barrier()
        del vae, prior
        torch.cuda.empty_cache()

    # ---- probe (rank 0) ---------------------------------------------------
    if not is_main:
        if world > 1:
            dist.destroy_process_group()
        return

    torch.backends.cuda.matmul.allow_tf32 = False
    with open(os.path.join(cache_dir, "meta.json")) as f:
        meta = json.load(f)
    specs = meta["specs"]

    def load_split(tag):
        shards = sorted(glob.glob(os.path.join(cache_dir, f"{tag}-rank*.pt")))
        parts = [torch.load(s, map_location="cpu") for s in shards]
        feats = {t: torch.cat([p["feats"][t] for p in parts]) for t in targets}
        labels = torch.cat([p["labels"] for p in parts])
        return feats, labels

    train_f, train_y = load_split("train")
    val_f, val_y = load_split("val")
    print(f">>> probing on {train_y.numel()} train / {val_y.numel()} val features",
          flush=True)

    pool_req = [g for g in args.pool_grids.split(",") if g]
    smooth_req = [g for g in args.smooth_grids.split(",") if g]
    lambdas = [float(x) for x in args.lambdas.split(",") if x]

    results = []
    for t in targets:
        c, g_stored = int(specs[t]["channels"]), int(specs[t]["grid"])
        variants = [("pool", g) for g in divisor_grids(g_stored, pool_req)]
        variants += [("smooth", g) for g in divisor_grids(g_stored, smooth_req)
                     if g != g_stored]
        for kind, g in variants:
            name = f"{t}/{kind}{g}"
            t0 = time.time()
            r = ridge_probe(train_f[t], train_y, val_f[t], val_y,
                            make_variant(c, g_stored, kind, g), lambdas,
                            args.dev_stride, args.solve_chunk, device)
            r.update({"target": t, "variant": f"{kind}{g}", "name": name,
                      "secs": round(time.time() - t0, 1)})
            results.append(r)
            print(f"  {name:24s} dim={r['dim']:6d} lam={r['lambda']:<8g} "
                  f"top1={r['top1']*100:6.2f} top5={r['top5']*100:6.2f} "
                  f"({r['secs']}s)", flush=True)

    print("\n=== ImageNet linear probe (val top-1 %) ===")
    for r in sorted(results, key=lambda r: -r["top1"]):
        print(f"  {r['name']:24s} {r['top1']*100:6.2f}  (dim {r['dim']})")

    out = args.out_json or os.path.join(exp_dir, "probe.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump({"args": vars(args), "specs": specs, "results": results}, f, indent=2)
    print(f"\n>>> wrote {out}", flush=True)

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
