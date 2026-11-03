# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

"""Lightweight in-training evaluation: rFID / PSNR / SSIM (reconstruction) and
gFID (generation).

Built on ``torchmetrics`` (FrechetInceptionDistance, PeakSignalNoiseRatio,
StructuralSimilarityIndexMeasure). All metrics are computed over a *subset* of
images sampled periodically during training, so the FID values are a
**relative monitoring proxy** rather than the official 50k-sample FID — use
``Prior_only/generate.py`` + ``Prior_only/eval_fid.py`` for final numbers.

Distributed: torchmetrics metrics synchronize their internal state across ranks
on ``compute()`` when ``torch.distributed`` is initialized (which Accelerate
sets up), so each rank just updates with its own shard.

Convention: all images passed to the evaluator are float tensors in ``[0, 1]``
with shape ``(N, 3, H, W)``.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import torch


class OnlineEvaluator:
    """Accumulates rFID/PSNR/SSIM (reconstruction) and gFID (generation)."""

    def __init__(self, device, eval_recon: bool = True, eval_gen: bool = True,
                 fid_feature: int = 2048):
        # Imported lazily so importing this module never triggers the
        # Inception-weight download unless evaluation is actually used.
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image import (
            PeakSignalNoiseRatio,
            StructuralSimilarityIndexMeasure,
        )

        self.device = device
        self.eval_recon = eval_recon
        self.eval_gen = eval_gen

        if eval_recon:
            self.rfid = FrechetInceptionDistance(feature=fid_feature, normalize=True).to(device)
            self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
            self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        if eval_gen:
            self.gfid = FrechetInceptionDistance(feature=fid_feature, normalize=True).to(device)
        self.reset()

    def reset(self):
        if self.eval_recon:
            self.rfid.reset()
            self.psnr.reset()
            self.ssim.reset()
        if self.eval_gen:
            self.gfid.reset()

    @torch.no_grad()
    def update_recon(self, real01: torch.Tensor, recon01: torch.Tensor):
        real = real01.clamp(0, 1).float()
        recon = recon01.clamp(0, 1).float()
        self.rfid.update(real, real=True)
        self.rfid.update(recon, real=False)
        self.psnr.update(recon, real)
        self.ssim.update(recon, real)

    @torch.no_grad()
    def update_gen(self, real01: torch.Tensor, gen01: torch.Tensor):
        self.gfid.update(real01.clamp(0, 1).float(), real=True)
        self.gfid.update(gen01.clamp(0, 1).float(), real=False)

    @torch.no_grad()
    def compute(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if self.eval_recon:
            out["rfid"] = float(self.rfid.compute())
            out["psnr"] = float(self.psnr.compute())
            out["ssim"] = float(self.ssim.compute())
        if self.eval_gen:
            out["gfid"] = float(self.gfid.compute())
        return out


@torch.no_grad()
def run_online_eval(
    evaluator: OnlineEvaluator,
    eval_loader,
    device,
    num_samples: int,
    recon_fn: Optional[Callable] = None,
    gen_fn: Optional[Callable] = None,
) -> Dict[str, float]:
    """Run one evaluation pass over up to ``num_samples`` images per rank.

    Args:
        recon_fn: ``(raw_image_uint8, labels) -> recon01`` (float [0,1]) or None.
        gen_fn:   ``(labels) -> gen01`` (float [0,1]) or None.
    The ``raw_image`` from the loader is uint8/float in [0, 255]; the real image
    used for the metrics is ``raw_image / 255``.
    """
    evaluator.reset()
    seen = 0
    for raw_image, y in eval_loader:
        raw_image = raw_image.to(device)
        y = y.to(device)
        real01 = raw_image.float() / 255.0
        if recon_fn is not None and evaluator.eval_recon:
            recon01 = recon_fn(raw_image, y)
            evaluator.update_recon(real01, recon01)
        if gen_fn is not None and evaluator.eval_gen:
            gen01 = gen_fn(y)
            evaluator.update_gen(real01, gen01)
        seen += raw_image.shape[0]
        if seen >= num_samples:
            break
    return evaluator.compute()
