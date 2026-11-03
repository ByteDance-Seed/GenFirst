# Copyright (c) 2024 Sihyun Yu
#
# From REPA (https://github.com/sihyun-yu/REPA), of which this directory is a
# fork. Licensed under the MIT license in Prior_only/LICENSE.

import torch
import numpy as np


def expand_t_like_x(t, x_cur):
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim,], time vector
      x: [batch_dim,...], data point
    """
    dims = [1] * (len(x_cur.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t

def get_score_from_velocity(vt, xt, t, path_type="linear"):
    """Wrapper function: transfrom velocity prediction model to score
    Args:
        velocity: [batch_dim, ...] shaped tensor; velocity model output
        x: [batch_dim, ...] shaped tensor; x_t data point
        t: [batch_dim,] time tensor
    """
    t = expand_t_like_x(t, xt)
    if path_type == "linear":
        alpha_t, d_alpha_t = 1 - t, torch.ones_like(xt, device=xt.device) * -1
        sigma_t, d_sigma_t = t, torch.ones_like(xt, device=xt.device)
    elif path_type == "cosine":
        alpha_t = torch.cos(t * np.pi / 2)
        sigma_t = torch.sin(t * np.pi / 2)
        d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
        d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
    else:
        raise NotImplementedError

    mean = xt
    reverse_alpha_ratio = alpha_t / d_alpha_t
    var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
    score = (reverse_alpha_ratio * vt - mean) / var

    return score


def compute_diffusion(t_cur):
    return 2 * t_cur


def _eval_drift(model, x_cur, t_cur, y, y_null, doubled, dtype, sde, path_type):
    """One model evaluation at ``t_cur``, returned as a float64 drift.

    ``doubled`` stacks [cond, uncond] into one batch for classifier-free guidance.
    For the SDE sampler the velocity is converted to the reverse-time drift here,
    which is affine in the velocity, so guidance may be applied either before or
    after the conversion -- the weights of every combination below sum to one, so
    the affine constant cancels.
    """
    if doubled:
        model_input = torch.cat([x_cur] * 2, dim=0)
        y_cur = torch.cat([y, y_null], dim=0)
    else:
        model_input = x_cur
        y_cur = y
    time_input = torch.ones(model_input.size(0)).to(
        device=x_cur.device, dtype=torch.float64
        ) * t_cur
    v_cur = model(
        model_input.to(dtype=dtype), time_input.to(dtype=dtype), y=y_cur
        )[0].to(torch.float64)
    if not sde:
        return v_cur
    s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
    return v_cur - 0.5 * compute_diffusion(t_cur) * s_cur


def _apply_guidance(d_main, d_weak, cfg_scale, autoguidance_scale, do_cfg):
    """Fold autoguidance and classifier-free guidance into one drift.

    Autoguidance (Karras et al., "Guiding a Diffusion Model with a Bad Version of
    Itself") points the guidance direction away from a deliberately weaker model
    -- here an earlier checkpoint of the same run -- instead of away from the
    unconditional branch. It is applied inside each CFG branch before the branches
    are mixed, so the two scales compose instead of competing.

    The two are gated on independent time intervals, so all three combinations
    reach here. With CFG off but autoguidance on, ``d_main`` and ``d_weak`` are
    both plain conditional drifts and the result is pure autoguidance.
    """
    if d_weak is not None:
        d_main = d_weak + autoguidance_scale * (d_main - d_weak)
    if do_cfg:
        d_cond, d_uncond = d_main.chunk(2)
        d_main = d_uncond + cfg_scale * (d_cond - d_uncond)
    return d_main


def euler_sampler(
        model,
        latents,
        y,
        num_steps=20,
        heun=False,
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        path_type="linear", # not used, just for compatability
        weak_model=None,
        autoguidance_scale=1.0,
        ag_low=0.0,
        ag_high=1.0,
        ):
    # setup conditioning
    if cfg_scale > 1.0:
        y_null = torch.tensor([1000] * y.size(0), device=y.device)
    else:
        y_null = None
    use_autoguidance = weak_model is not None and autoguidance_scale != 1.0
    _dtype = latents.dtype
    t_steps = torch.linspace(1, 0, num_steps+1, dtype=torch.float64)
    x_next = latents.to(torch.float64)

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            do_cfg = cfg_scale > 1.0 and guidance_low <= t_cur <= guidance_high
            do_ag = use_autoguidance and ag_low <= t_cur <= ag_high
            drift_kwargs = dict(
                y=y, y_null=y_null, doubled=do_cfg, dtype=_dtype,
                sde=False, path_type=path_type,
                )
            d_cur = _eval_drift(model, x_cur, t_cur, **drift_kwargs)
            d_weak = _eval_drift(weak_model, x_cur, t_cur, **drift_kwargs) if do_ag else None
            d_cur = _apply_guidance(d_cur, d_weak, cfg_scale, autoguidance_scale, do_cfg)
            x_next = x_cur + (t_next - t_cur) * d_cur
            if heun and (i < num_steps - 1):
                d_prime = _eval_drift(model, x_next, t_next, **drift_kwargs)
                d_prime_weak = (
                    _eval_drift(weak_model, x_next, t_next, **drift_kwargs) if do_ag else None
                    )
                d_prime = _apply_guidance(
                    d_prime, d_prime_weak, cfg_scale, autoguidance_scale, do_cfg
                    )
                x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


def euler_maruyama_sampler(
        model,
        latents,
        y,
        num_steps=20,
        heun=False,  # not used, just for compatability
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        path_type="linear",
        weak_model=None,
        autoguidance_scale=1.0,
        ag_low=0.0,
        ag_high=1.0,
        ):
    # setup conditioning
    if cfg_scale > 1.0:
        y_null = torch.tensor([1000] * y.size(0), device=y.device)
    else:
        y_null = None
    use_autoguidance = weak_model is not None and autoguidance_scale != 1.0

    _dtype = latents.dtype

    t_steps = torch.linspace(1., 0.04, num_steps, dtype=torch.float64)
    t_steps = torch.cat([t_steps, torch.tensor([0.], dtype=torch.float64)])
    x_next = latents.to(torch.float64)
    device = x_next.device

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-2], t_steps[1:-1])):
            dt = t_next - t_cur
            x_cur = x_next
            do_cfg = cfg_scale > 1.0 and guidance_low <= t_cur <= guidance_high
            do_ag = use_autoguidance and ag_low <= t_cur <= ag_high
            diffusion = compute_diffusion(t_cur)
            eps_i = torch.randn_like(x_cur).to(device)
            deps = eps_i * torch.sqrt(torch.abs(dt))

            # compute drift
            drift_kwargs = dict(
                y=y, y_null=y_null, doubled=do_cfg, dtype=_dtype,
                sde=True, path_type=path_type,
                )
            d_cur = _eval_drift(model, x_cur, t_cur, **drift_kwargs)
            d_weak = _eval_drift(weak_model, x_cur, t_cur, **drift_kwargs) if do_ag else None
            d_cur = _apply_guidance(d_cur, d_weak, cfg_scale, autoguidance_scale, do_cfg)

            x_next =  x_cur + d_cur * dt + torch.sqrt(diffusion) * deps

    # last step
    t_cur, t_next = t_steps[-2], t_steps[-1]
    dt = t_next - t_cur
    x_cur = x_next
    do_cfg = cfg_scale > 1.0 and guidance_low <= t_cur <= guidance_high
    do_ag = use_autoguidance and ag_low <= t_cur <= ag_high

    # compute drift
    drift_kwargs = dict(
        y=y, y_null=y_null, doubled=do_cfg, dtype=_dtype,
        sde=True, path_type=path_type,
        )
    d_cur = _eval_drift(model, x_cur, t_cur, **drift_kwargs)
    d_weak = _eval_drift(weak_model, x_cur, t_cur, **drift_kwargs) if do_ag else None
    d_cur = _apply_guidance(d_cur, d_weak, cfg_scale, autoguidance_scale, do_cfg)

    mean_x = x_cur + dt * d_cur

    return mean_x
