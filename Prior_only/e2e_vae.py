# Copyright (c) 2022 Machine Vision and Learning Group, LMU Munich
#
# From CompVis/latent-diffusion (https://github.com/CompVis/latent-diffusion),
# MIT. See NOTICE.

"""Self-contained copy of the AutoencoderKL used in the end-to-end stage.

This mirrors the root `models/autoencoder.py` -- the VAE that was trained
end-to-end together with the SiT prior -- so that the prior-only stage encodes
images and decodes latents with the *exact same* VAE. Keeping a copy here lets
this directory run standalone, without importing from the end-to-end trainer.

Latent shape for 256x256 inputs is [B, 4, 32, 32] (f8, d4), which matches what
the Stable-Diffusion VAE produces, so the rest of the pipeline is unchanged.

Ref: https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/autoencoder.py
"""

import os

import numpy as np
import torch
import torch.nn as nn


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch=128,
        out_ch=3,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=256,
        z_channels=16,
        double_z=True,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch=128,
        out_ch=3,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=256,
        z_channels=16,
        give_pre_end=False,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = torch.nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z):
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device
            )

    def sample(self):
        x = self.mean + self.std * torch.randn(
            self.mean.shape, device=self.parameters.device
        )
        return x

    def mode(self):
        return self.mean


class _DecodeOutput:
    """Tiny container so callers can use `vae.decode(z).sample`, matching the
    diffusers `AutoencoderKL.decode(...).sample` interface used by REPA."""

    def __init__(self, sample):
        self.sample = sample


class AutoencoderKL(nn.Module):
    def __init__(self, embed_dim, ch_mult, use_variational=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = Encoder(ch_mult=ch_mult, z_channels=embed_dim)
        self.decoder = Decoder(ch_mult=ch_mult, z_channels=embed_dim)
        self.use_variational = use_variational
        mult = 2 if self.use_variational else 1
        self.quant_conv = torch.nn.Conv2d(2 * embed_dim, mult * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, embed_dim, 1)

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        if not self.use_variational:
            moments = torch.cat((moments, torch.ones_like(moments)), 1)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return _DecodeOutput(sample=dec)

    def forward(self, x, return_recon=True):
        posterior = self.encode(x)
        z = posterior.sample()
        recon = None
        if return_recon:
            recon = self.decode(z).sample
        return posterior, z, recon


# Predefined VAE architectures (matching the end-to-end trainer)
def VAE_F8D4(**kwargs):
    # [B, 4, 32, 32]
    return AutoencoderKL(embed_dim=4, ch_mult=[1, 2, 4, 4], use_variational=True, **kwargs)


def VAE_F16D32(**kwargs):
    # [B, 32, 16, 16]
    return AutoencoderKL(
        embed_dim=32, ch_mult=[1, 1, 2, 2, 4], use_variational=True, **kwargs
    )


FLUX_VAE_CONFIG = dict(
    in_channels=3,
    out_channels=3,
    latent_channels=16,
    block_out_channels=[128, 256, 512, 512],
    down_block_types=["DownEncoderBlock2D"] * 4,
    up_block_types=["UpDecoderBlock2D"] * 4,
    layers_per_block=2,
    norm_num_groups=32,
    act_fn="silu",
    sample_size=1024,
    scaling_factor=0.3611,
    shift_factor=0.1159,
    use_quant_conv=False,
    use_post_quant_conv=False,
    force_upcast=True,
)


class FluxVAE(nn.Module):
    """diffusers FLUX.1 AutoencoderKL adapted to this repository's VAE interface."""

    scaling_factor = FLUX_VAE_CONFIG["scaling_factor"]
    shift_factor = FLUX_VAE_CONFIG["shift_factor"]

    def __init__(self, pretrained_path=None):
        super().__init__()
        from diffusers import AutoencoderKL as _DiffusersAutoencoderKL

        if pretrained_path is not None:
            self.vae = _DiffusersAutoencoderKL.from_pretrained(pretrained_path)
        else:
            self.vae = _DiffusersAutoencoderKL(**FLUX_VAE_CONFIG)
        self.embed_dim = self.vae.config.latent_channels

    def encode(self, x):
        moments = self.vae.encode(x).latent_dist.parameters
        return DiagonalGaussianDistribution(moments)

    def decode(self, z):
        dec = self.vae.decode(z).sample
        return _DecodeOutput(sample=dec)

    def forward(self, x, return_recon=True):
        posterior = self.encode(x)
        z = posterior.sample()
        recon = self.decode(z).sample if return_recon else None
        return posterior, z, recon


def VAE_FLUX(pretrained_path=None, **kwargs):
    return FluxVAE(pretrained_path=pretrained_path)


FLUX2_VAE_CONFIG = dict(
    in_channels=3,
    out_channels=3,
    latent_channels=32,
    block_out_channels=[128, 256, 512, 512],
    down_block_types=["DownEncoderBlock2D"] * 4,
    up_block_types=["UpDecoderBlock2D"] * 4,
    layers_per_block=2,
    norm_num_groups=32,
    act_fn="silu",
    sample_size=1024,
    scaling_factor=1.0,
    shift_factor=0.0,
    use_quant_conv=True,
    use_post_quant_conv=True,
    force_upcast=True,
)


def _read_diffusers_vae_state_dict(path):
    """Read a diffusers VAE state_dict from a model folder or a single file."""
    if os.path.isdir(path):
        for name in (
            "diffusion_pytorch_model.safetensors",
            "diffusion_pytorch_model.bin",
            "e2e-vae-flux2-sitxl2-auxkl2p5e8-s3-20k.safetensors",
        ):
            candidate = os.path.join(path, name)
            if os.path.isfile(candidate):
                path = candidate
                break
        else:
            raise FileNotFoundError(f"No diffusers VAE weights found in {path}")
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path)
    return torch.load(path, map_location="cpu", weights_only=False)


class Flux2VAE(nn.Module):
    """diffusers FLUX.2 `AutoencoderKL` (32ch, f8) adapted to this repository's VAE interface."""

    scaling_factor = FLUX2_VAE_CONFIG["scaling_factor"]
    shift_factor = FLUX2_VAE_CONFIG["shift_factor"]
    patch_size = 2

    def __init__(self, pretrained_path=None):
        super().__init__()
        from diffusers import AutoencoderKL as _DiffusersAutoencoderKL

        self.vae = _DiffusersAutoencoderKL(**FLUX2_VAE_CONFIG)
        self.embed_dim = self.vae.config.latent_channels
        folded_channels = self.embed_dim * self.patch_size ** 2
        self.register_buffer("bn_running_mean", torch.zeros(folded_channels))
        self.register_buffer("bn_running_var", torch.ones(folded_channels))
        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)

    def load_pretrained(self, pretrained_path):
        sd = _read_diffusers_vae_state_dict(pretrained_path)
        if isinstance(sd, dict) and "vae" in sd and not any(k.startswith("vae.") for k in sd):
            sd = sd["vae"]
        if any(k.startswith("vae.") for k in sd):
            missing, unexpected = self.load_state_dict(sd, strict=False)
        else:
            sd = dict(sd)
            bn_mean = sd.pop("bn.running_mean", None)
            bn_var = sd.pop("bn.running_var", None)
            sd.pop("bn.num_batches_tracked", None)
            missing, unexpected = self.vae.load_state_dict(sd, strict=False)
            if bn_mean is not None:
                self.bn_running_mean.copy_(bn_mean)
            if bn_var is not None:
                self.bn_running_var.copy_(bn_var)
        if len(missing) > 0 or len(unexpected) > 0:
            print(f"[Flux2VAE] pretrained load: missing={missing} unexpected={unexpected}")

    def initial_latents_stats(self, device=None):
        """Per-channel ``(scale, bias)`` in E2E convention (`(z - bias) * scale`),
        averaging each latent channel's 4 patch-folded bn entries."""
        folded = self.patch_size ** 2
        mean = self.bn_running_mean.detach().float().view(self.embed_dim, folded).mean(dim=1)
        var = self.bn_running_var.detach().float().view(self.embed_dim, folded).mean(dim=1)
        scale = var.clamp_min(1e-8).rsqrt()
        if device is not None:
            mean, scale = mean.to(device), scale.to(device)
        return scale, mean

    def encode(self, x):
        moments = self.vae.encode(x).latent_dist.parameters
        return DiagonalGaussianDistribution(moments)

    def decode(self, z):
        dec = self.vae.decode(z).sample
        return _DecodeOutput(sample=dec)

    def forward(self, x, return_recon=True):
        posterior = self.encode(x)
        z = posterior.sample()
        recon = self.decode(z).sample if return_recon else None
        return posterior, z, recon


def VAE_FLUX2(pretrained_path=None, **kwargs):
    return Flux2VAE(pretrained_path=pretrained_path)


vae_models = {
    "f8d4": VAE_F8D4,  # [B, 4, 32, 32]
    "f16d32": VAE_F16D32,  # [B, 32, 16, 16]
    "flux": VAE_FLUX,  # [B, 16, 32, 32]
    "flux2": VAE_FLUX2,  # [B, 32, 32, 32]
}


# Latent spec per VAE arch: (spatial downsample factor, latent channels).
_vae_arch_specs = {
    "f8d4": {"downsample": 8, "channels": 4},
    "f16d32": {"downsample": 16, "channels": 32},
    "flux": {"downsample": 8, "channels": 16},
    "flux2": {"downsample": 8, "channels": 32},
    "qwenvit": {"downsample": 8, "channels": 16},
}
VAE_LATENT_CHANNELS = {k: v["channels"] for k, v in _vae_arch_specs.items()}


def vae_arch_spec(vae_arch):
    """Return (downsample_factor, latent_channels) for a given VAE arch."""
    if vae_arch not in _vae_arch_specs:
        raise ValueError(
            f"Unknown vae_arch '{vae_arch}'. Known: {list(_vae_arch_specs.keys())}"
        )
    spec = _vae_arch_specs[vae_arch]
    return spec["downsample"], spec["channels"]


def _load_qwenvit_class():
    """Load the parent checkout's Qwen ViT VAE without colliding with Prior_only.models."""
    import importlib
    import sys
    import types
    from pathlib import Path

    package = "_genfirst_parent_models"
    models_dir = Path(__file__).resolve().parents[1] / "models"
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(models_dir)]
        module.__package__ = package
        sys.modules[package] = module
    return importlib.import_module(f"{package}.qwen_vit_ae").QwenViTAutoencoder


def load_e2e_vae(vae_arch, vae_ckpt, device="cpu", eval_mode=True, **kwargs):
    """Build an end-to-end VAE and load extracted weights."""
    if vae_arch == "qwenvit":
        QwenViTAutoencoder = _load_qwenvit_class()
        model_path = kwargs.get("model_path") or "Qwen/Qwen3-VL-2B-Instruct"
        vae = QwenViTAutoencoder(
            model_path=model_path,
            latent_channels=int(kwargs.get("latent_channels", 16)),
            decoder_width=int(kwargs.get("decoder_width", 1024)),
            decoder_depth=int(kwargs.get("decoder_depth", 12)),
            decoder_heads=int(kwargs.get("decoder_heads", 16)),
            decoder_grid=int(kwargs.get("decoder_grid", 16)),
            conv_refine=bool(kwargs.get("conv_refine", True)),
            freeze_encoder=True,
        )
        sd = torch.load(vae_ckpt, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "vae" in sd:
            sd = sd["vae"]
        missing, unexpected = vae.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Qwen ViT VAE checkpoint is incompatible: "
                f"missing={missing[:12]} unexpected={unexpected[:12]}"
            )
        vae = vae.to(device)
        if eval_mode:
            vae.eval()
            vae.requires_grad_(False)
        return vae

    if vae_arch == "flux2":
        vae = Flux2VAE(pretrained_path=vae_ckpt)
        vae = vae.to(device)
        if eval_mode:
            vae.eval()
            vae.requires_grad_(False)
        return vae

    if vae_arch == "flux":
        if os.path.isdir(vae_ckpt):
            vae = FluxVAE(pretrained_path=vae_ckpt)
        else:
            vae = FluxVAE(pretrained_path=None)
            sd = _read_diffusers_vae_state_dict(vae_ckpt)
            if isinstance(sd, dict) and "vae" in sd and "vae.encoder.conv_in.weight" not in sd:
                sd = sd["vae"]
            missing, unexpected = vae.load_state_dict(sd, strict=False)
            if len(missing) > 0 or len(unexpected) > 0:
                print(
                    f"[load_e2e_vae] flux loaded with missing={len(missing)} "
                    f"unexpected={len(unexpected)} keys"
                )
        vae = vae.to(device)
        if eval_mode:
            vae.eval()
            vae.requires_grad_(False)
        return vae

    vae = vae_models[vae_arch]()
    sd = torch.load(vae_ckpt, map_location="cpu", weights_only=False)
    # support either a raw state_dict or a checkpoint dict containing "vae"
    if isinstance(sd, dict) and "vae" in sd and "encoder.conv_in.weight" not in sd:
        sd = sd["vae"]
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    if len(missing) > 0 or len(unexpected) > 0:
        print(
            f"[load_e2e_vae] loaded with missing={len(missing)} "
            f"unexpected={len(unexpected)} keys"
        )
    vae = vae.to(device)
    if eval_mode:
        vae.eval()
        vae.requires_grad_(False)
    return vae


def latents_stats_to_repa(latents_stats):
    """Convert end-to-end latent stats to REPA's (scale, bias) convention.

    The end-to-end trainer normalizes latents as `(z - bias_e) * scale_e`
    (`scale_e = 1/std`, `bias_e = mean`), see the root `utils.py`
    `normalize_latents` / `denormalize_latents`.

    REPA normalizes as `z * scale_r + bias_r` and decodes with
    `(z - bias_r) / scale_r` (train_prior_only.py `sample_posterior`). To make REPA's
    formula identical to that standardization we set:
        scale_r = scale_e
        bias_r  = -bias_e * scale_e
    """
    scale_e = latents_stats["latents_scale"].float()
    bias_e = latents_stats["latents_bias"].float()
    scale_r = scale_e
    bias_r = -bias_e * scale_e
    return scale_r, bias_r
