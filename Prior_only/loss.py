# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Sihyun Yu
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT License. The copy in Prior_only/LICENSE is the
# governing text for this directory.
#
# This module is the stochastic-interpolant training loss used by the
# prior-only trainers, including the representation-alignment term from
# REPA (https://github.com/sihyun-yu/REPA). The algorithm is REPA's; the
# implementation below is written for this repository and is not a copy of
# any third-party source file.

"""Stochastic-interpolant denoising loss plus optional REPA alignment."""

import math

import torch
import torch.nn.functional as F


def _mean_over_spatial(tensor):
    """Average every axis after the batch axis. Result keeps shape (B,)."""
    if tensor.ndim == 1:
        return tensor
    spatial = tuple(range(1, tensor.ndim))
    return tensor.mean(dim=spatial)


def _linear_schedule(t):
    # x_t = (1 - t) x_0 + t eps, velocity target = eps - x_0
    return 1.0 - t, t, -1.0, 1.0


def _cosine_schedule(t):
    half_pi = 0.5 * math.pi
    alpha = torch.cos(t * half_pi)
    sigma = torch.sin(t * half_pi)
    return alpha, sigma, -half_pi * torch.sin(t * half_pi), half_pi * torch.cos(t * half_pi)


_SCHEDULES = {
    "linear": _linear_schedule,
    "cosine": _cosine_schedule,
}


def _draw_times(batch, weighting, path_type, device, dtype):
    # Sample on CPU, then move. Matches the original REPA trainer so a seeded
    # run still draws the same times before the on-device noise.
    if weighting == "uniform":
        times = torch.rand(batch, 1, 1, 1)
    elif weighting == "lognormal":
        # EDM-style log-normal sigma, then invert the chosen interpolant for t.
        sigma = torch.randn(batch, 1, 1, 1).exp()
        if path_type == "linear":
            times = sigma / (1.0 + sigma)
        elif path_type == "cosine":
            times = (2.0 / math.pi) * torch.atan(sigma)
        else:
            raise NotImplementedError(f"unknown interpolant: {path_type}")
    else:
        raise NotImplementedError(f"unknown time weighting: {weighting}")
    return times.to(device=device, dtype=dtype)


def _repa_alignment(teacher_tokens, student_tokens):
    """Mean negative cosine between teacher and student patch tokens.

    REPA's original loop zipped the batch axis of each ``(B, T, D)`` feature
    and then divided the summed per-sample token-means by
    ``(n_encoders * B)``. For a fixed token length that is exactly the mean
    over batch and tokens, then over encoders.
    """
    if not teacher_tokens:
        raise ValueError("zs must contain at least one encoder feature")
    acc = None
    for teacher, student in zip(teacher_tokens, student_tokens):
        teacher = F.normalize(teacher, dim=-1)
        student = F.normalize(student, dim=-1)
        per_encoder = (-(teacher * student).sum(dim=-1)).mean()
        acc = per_encoder if acc is None else acc + per_encoder
    return acc / len(teacher_tokens)


class SILoss:
    """Velocity-prediction loss on a linear or cosine interpolant.

    ``encoders``, ``accelerator``, ``latents_scale`` and ``latents_bias`` are
    accepted so existing trainer call sites keep working; they are not used
    inside the forward. Alignment features are passed in as ``zs`` instead.
    """

    def __init__(
        self,
        prediction="v",
        path_type="linear",
        weighting="uniform",
        encoders=None,
        accelerator=None,
        latents_scale=None,
        latents_bias=None,
    ):
        if prediction != "v":
            raise NotImplementedError("only velocity prediction is implemented")
        if path_type not in _SCHEDULES:
            raise NotImplementedError(f"unknown interpolant: {path_type}")
        if weighting not in ("uniform", "lognormal"):
            raise NotImplementedError(f"unknown time weighting: {weighting}")
        self.prediction = prediction
        self.path_type = path_type
        self.weighting = weighting
        self.encoders = encoders or []
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias

    def interpolant(self, t):
        return _SCHEDULES[self.path_type](t)

    def __call__(self, model, images, model_kwargs=None, zs=None):
        if model_kwargs is None:
            model_kwargs = {}

        t = _draw_times(
            images.shape[0], self.weighting, self.path_type, images.device, images.dtype
        )
        noise = torch.randn_like(images)
        alpha, sigma, d_alpha, d_sigma = self.interpolant(t)
        x_t = alpha * images + sigma * noise
        target = d_alpha * images + d_sigma * noise

        pred, zs_hat = model(x_t, t.flatten(), **model_kwargs)
        denoise = _mean_over_spatial((pred - target).square())
        align = _repa_alignment(zs, zs_hat)
        return denoise, align
