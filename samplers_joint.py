# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

"""Samplers for the block-causal joint MMDiT.

Both directions integrate the same flow-matching ODE that trains the model,
``dx/dt = v(x, t)`` from ``t=1`` (noise) to ``t=0`` (data), with the step
convention of :mod:`samplers` (``x += (t_next - t_cur) * v``).

The two directions differ in what is held clean and in how many ODEs are solved:

* ``t2i`` — the text latent is the clean condition at ``t=0`` and one ODE
  produces the whole image latent.
* ``i2t`` — the image latent is the clean condition and one ODE is solved per
  text block, in order, because block ``b`` conditions on a *clean* history of
  blocks ``0..b-1``. This is the inference-time counterpart of the 2L training
  layout.

Classifier-free guidance needs the model to have been trained with
``cond_dropout > 0``; the unconditional branch is obtained by hiding the
condition modality with ``cond_drop`` rather than by a null token. Passing
``cfg_scale != 1`` to a checkpoint trained without dropout will produce garbage,
so callers should check the training arg.
"""

import torch

from models.joint_mmdit import TASK_I2T, TASK_T2I


def _steps(num_steps, device):
    return torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float32, device=device)


def _cfg_drop(B, device):
    """``(2B,)`` mask: first half keeps the condition, second half hides it."""
    return torch.cat([
        torch.zeros(B, dtype=torch.bool, device=device),
        torch.ones(B, dtype=torch.bool, device=device),
    ])


def _combine(v, cfg_scale):
    v_cond, v_uncond = v.chunk(2)
    return v_uncond + cfg_scale * (v_cond - v_uncond)


@torch.no_grad()
def t2i_sample(
    model, z_txt, text_valid, num_steps=50, cfg_scale=1.0, heun=False, generator=None,
    x_init=None,
):
    """Sample image latents from clean text latents.

    Args:
        model: :class:`~models.joint_mmdit.BlockCausalJointMMDiT` (unwrapped).
        z_txt: clean text latent ``(B, L, C_txt)`` from the text VAE.
        text_valid: ``(B, L)`` bool over real token positions.
        cfg_scale: 1.0 disables guidance; above that the batch is doubled and
            the text-blind branch is used as the unconditional prediction.
        x_init: optional starting noise ``(B, C, H, W)``. Batch evaluation needs
            each sample's noise to depend only on its own prompt, so that how
            prompts happen to be grouped into batches -- or which ones a resumed
            run skips -- cannot change the images.

    Returns:
        Image latent ``(B, C, H, W)`` in **unnormalized** VAE space, ready for
        ``vae.decode``.
    """
    B = z_txt.shape[0]
    device = z_txt.device
    g = model.grid * model.patch_size
    if x_init is not None:
        x = x_init.to(device=device, dtype=torch.float32).clone()
    else:
        x = torch.randn(
            B, model.image_channels, g, g, device=device, dtype=torch.float32,
            generator=generator,
        )

    use_cfg = cfg_scale != 1.0
    n = 2 * B if use_cfg else B
    z_txt_in = torch.cat([z_txt, z_txt]) if use_cfg else z_txt
    valid_in = torch.cat([text_valid, text_valid]) if use_cfg else text_valid
    drop = _cfg_drop(B, device) if use_cfg else None
    t_txt = torch.zeros((n, z_txt.shape[1]), device=device, dtype=torch.float32)
    t_steps = _steps(num_steps, device)

    def velocity(x_in, t_scalar):
        t_img = torch.full((n,), float(t_scalar), device=device, dtype=torch.float32)
        v, _, _ = model.denoise(
            torch.cat([x_in, x_in]) if use_cfg else x_in,
            z_txt_in, valid_in, TASK_T2I,
            t_img=t_img, t_txt_tok=t_txt, cond_drop=drop,
        )
        return _combine(v.float(), cfg_scale) if use_cfg else v.float()

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        dt = t_next - t_cur
        v = velocity(x, t_cur)
        x_next = x + dt * v
        if heun and i < num_steps - 1:
            x_next = x + dt * 0.5 * (v + velocity(x_next, t_next))
        x = x_next

    return model.denormalize_image_latents(x)


@torch.no_grad()
def i2t_sample(
    model,
    z_img,
    n_blocks,
    num_steps=20,
    cfg_scale=1.0,
    heun=False,
    generator=None,
    eos_id=None,
    text_vae=None,
):
    """Generate text latents block by block from a clean image latent.

    Args:
        z_img: **unnormalized** image latent ``(B, C, H, W)``.
        n_blocks: how many text blocks to generate (upper bound on length).
        cfg_scale: 1.0 disables guidance; above that the image-blind branch (a
            pure text prior) is the unconditional prediction.
        eos_id, text_vae: when both are given, generation stops early once every
            sample has emitted an end-of-text token.

    Returns:
        ``(z_txt, valid)``: latents ``(B, L, C_txt)`` and the bool mask of
        positions that were actually generated, ``L = n_blocks * block_size``.
    """
    B = z_img.shape[0]
    device = z_img.device
    blk = model.text_block_size
    L = n_blocks * blk
    C = model.text_channels

    use_cfg = cfg_scale != 1.0
    n = 2 * B if use_cfg else B
    drop = _cfg_drop(B, device) if use_cfg else None

    img_n = model.normalize_image_latents(z_img).float()
    if use_cfg:
        img_n = torch.cat([img_n, img_n])
    t_img = torch.zeros((n,), device=device, dtype=torch.float32)

    z_clean = torch.zeros(B, L, C, device=device, dtype=torch.float32)
    valid = torch.zeros(B, L, dtype=torch.bool, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    t_steps = _steps(num_steps, device)

    def dup(t):
        return torch.cat([t, t]) if use_cfg else t

    for b in range(n_blocks):
        sl = slice(b * blk, (b + 1) * blk)
        valid[:, sl] = True
        x = torch.randn(B, blk, C, device=device, dtype=torch.float32, generator=generator)

        def velocity(x_blk, t_scalar):
            # Only this block's slice is read back. Under the 2L mask a noisy
            # query sees clean blocks strictly before it plus its own noisy
            # block, so neither the other noisy slots nor the not-yet-written
            # clean tail can influence the result.
            txt_noisy = torch.zeros(B, L, C, device=device, dtype=torch.float32)
            txt_noisy[:, sl] = x_blk
            t_txt = torch.zeros(B, L, device=device, dtype=torch.float32)
            t_txt[:, sl] = float(t_scalar)
            _, v, _ = model.denoise(
                img_n, dup(txt_noisy), dup(valid), TASK_I2T,
                t_img=t_img, t_txt_tok=dup(t_txt), txt_clean=dup(z_clean),
                cond_drop=drop,
            )
            v = v[:, sl].float()
            return _combine(v, cfg_scale) if use_cfg else v

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            dt = t_next - t_cur
            v = velocity(x, t_cur)
            x_next = x + dt * v
            if heun and i < num_steps - 1:
                x_next = x + dt * 0.5 * (v + velocity(x_next, t_next))
            x = x_next

        z_clean[:, sl] = x

        if eos_id is not None and text_vae is not None:
            ids = decode_text_ids(text_vae, z_clean[:, : (b + 1) * blk])
            finished |= (ids[:, b * blk :] == eos_id).any(dim=1)
            if bool(finished.all()):
                valid[:, (b + 1) * blk :] = False
                break

    return z_clean, valid


@torch.no_grad()
def decode_text_ids(text_vae, z_txt):
    """Decode padded text latents ``(B, L, C)`` to token ids ``(B, L)``."""
    B, L = z_txt.shape[0], z_txt.shape[1]
    txt_shape = torch.full((B, 1), L, dtype=torch.long, device=z_txt.device)
    z_na = z_txt.reshape(B * L, -1)
    logits = text_vae.decode(z_na, txt_shape)
    return logits.argmax(-1).reshape(B, L)


def ids_to_text(tokenizer, ids, valid=None, eos_id=None):
    """Detokenize ``(B, L)`` ids, cutting each sample at its first EOS."""
    out = []
    for i in range(ids.shape[0]):
        row = ids[i]
        if valid is not None:
            row = row[valid[i]]
        row = row.tolist()
        if eos_id is not None and eos_id in row:
            row = row[: row.index(eos_id)]
        out.append(tokenizer.decode(row))
    return out
