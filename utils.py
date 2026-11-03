# Copyright (c) 2025 Xingjian Leng and Jaskirat Singh
#
# From REPA-E (https://github.com/End2End-Diffusion/REPA-E), which this tree
# derives from. Licensed under the MIT license in LICENSE.

import math
import os
import warnings

import numpy as np
from PIL import Image
import timm
import torch

from models import mocov3_vit


def _encoder_ckpt_dir():
    """Directory holding the REPA alignment-encoder checkpoints.

    Only DINOv2 is needed for the recipes in this repository; the other encoders
    are kept for parity with REPA and expect their weights here. Override with
    ENCODER_CKPT_DIR.
    """
    return os.environ.get("ENCODER_CKPT_DIR", "ckpts")


def fix_mocov3_state_dict(state_dict):
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.base_encoder'):
            # fix naming bug in checkpoint
            new_k = k[len("module.base_encoder."):]
            if "blocks.13.norm13" in new_k:
                new_k = new_k.replace("norm13", "norm1")
            if "blocks.13.mlp.fc13" in k:
                new_k = new_k.replace("fc13", "fc1")
            if "blocks.14.norm14" in k:
                new_k = new_k.replace("norm14", "norm2")
            if "blocks.14.mlp.fc14" in k:
                new_k = new_k.replace("fc14", "fc2")
            # remove prefix
            if 'head' not in new_k and new_k.split('.')[0] != 'fc':
                state_dict[new_k] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    if 'pos_embed' in state_dict.keys():
        state_dict['pos_embed'] = timm.layers.pos_embed.resample_abs_pos_embed(
            state_dict['pos_embed'], [16, 16],
        )
    return state_dict

@torch.no_grad()
def load_encoders(enc_type, device, resolution=256):
    assert (resolution == 256) or (resolution == 512)
    
    enc_names = enc_type.split(',')
    encoders, architectures, encoder_types = [], [], []
    for enc_name in enc_names:
        encoder_type, architecture, model_config = enc_name.split('-')
        # Currently, we only support 512x512 experiments with DINOv2 encoders.
        if resolution == 512:
            if encoder_type != 'dinov2':
                raise NotImplementedError(
                    "Currently, we only support 512x512 experiments with DINOv2 encoders."
                    )

        architectures.append(architecture)
        encoder_types.append(encoder_type)
        if encoder_type == 'mocov3':
            if architecture == 'vit':
                if model_config == 's':
                    encoder = mocov3_vit.vit_small()
                elif model_config == 'b':
                    encoder = mocov3_vit.vit_base()
                elif model_config == 'l':
                    encoder = mocov3_vit.vit_large()
                ckpt = torch.load(os.path.join(_encoder_ckpt_dir(), f'mocov3_vit{model_config}.pth'))
                state_dict = fix_mocov3_state_dict(ckpt['state_dict'])
                del encoder.head
                encoder.load_state_dict(state_dict, strict=True)
                encoder.head = torch.nn.Identity()
            elif architecture == 'resnet':
                raise NotImplementedError()
 
            encoder = encoder.to(device)
            encoder.eval()

        elif 'dinov2' in encoder_type:
            import timm
            # Build the DINOv2 ViT architecture locally via timm and load the
            # pretrained weights from disk. This avoids torch.hub.load(), which
            # fetches the repo source code from GitHub at runtime (and fails with
            # an HTTP 403 rate-limit / connection error in offline environments).
            size_map = {"s": "small", "b": "base", "l": "large", "g": "giant"}
            reg = "reg" in encoder_type
            reg_tag = "_reg4" if reg else ""
            model_name = f"vit_{size_map[model_config]}_patch14{reg_tag}_dinov2"
            # dynamic_img_size lets timm interpolate the positional embedding to
            # whatever input resolution we feed (224 for 256px, 448 for 512px).
            encoder = timm.create_model(
                model_name, pretrained=False, num_classes=0, dynamic_img_size=True,
            )
            weights = os.environ.get(
                "DINOV2_WEIGHTS",
                f"pretrained/dinov2_vit{model_config}14{'_reg4' if reg else ''}_pretrain.pth",
            )
            state_dict = torch.load(weights, map_location="cpu", weights_only=False)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            # `mask_token` is present in the official checkpoint but unused here.
            encoder.load_state_dict(state_dict, strict=False)
            encoder.head = torch.nn.Identity()
            # Expose a DINOv2-hub-style forward_features returning a dict so the
            # rest of the codebase (z['x_norm_patchtokens']) keeps working.
            _num_prefix = encoder.num_prefix_tokens
            _timm_forward_features = encoder.forward_features

            def _dinov2_forward_features(x, _ff=_timm_forward_features, _np=_num_prefix):
                tokens = _ff(x)
                return {
                    "x_norm_clstoken": tokens[:, 0],
                    "x_norm_patchtokens": tokens[:, _np:],
                }

            encoder.forward_features = _dinov2_forward_features
            encoder = encoder.to(device)
            encoder.eval()
        
        elif 'dinov1' == encoder_type:
            # The DINOv1 backbone this branch needs (models/dinov1.py) is not part
            # of this repository. DINOv2 is what the paper uses.
            raise NotImplementedError(
                "The 'dinov1' encoder is not available in this release; "
                "use --enc-type dinov2-vit-b."
            )

        elif encoder_type == 'clip':
            import clip
            from models.clip_vit import UpdatedVisionTransformer
            encoder_ = clip.load(f"ViT-{model_config}/14", device='cpu')[0].visual
            encoder = UpdatedVisionTransformer(encoder_).to(device)
             #.to(device)
            encoder.embed_dim = encoder.model.transformer.width
            encoder.forward_features = encoder.forward
            encoder.eval()
        
        elif encoder_type == 'mae':
            from models.mae_vit import vit_large_patch16
            import timm
            kwargs = dict(img_size=256)
            encoder = vit_large_patch16(**kwargs).to(device)
            with open(os.path.join(_encoder_ckpt_dir(), f"mae_vit{model_config}.pth"), "rb") as f:
                state_dict = torch.load(f)
            if 'pos_embed' in state_dict["model"].keys():
                state_dict["model"]['pos_embed'] = timm.layers.pos_embed.resample_abs_pos_embed(
                    state_dict["model"]['pos_embed'], [16, 16],
                )
            encoder.load_state_dict(state_dict["model"])

            encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
                encoder.pos_embed.data, [16, 16],
            )

        elif encoder_type == 'jepa':
            from models.jepa import vit_huge
            kwargs = dict(img_size=[224, 224], patch_size=14)
            encoder = vit_huge(**kwargs).to(device)
            with open(os.path.join(_encoder_ckpt_dir(), f"ijepa_vit{model_config}.pth"), "rb") as f:
                state_dict = torch.load(f, map_location=device)
            new_state_dict = dict()
            for key, value in state_dict['encoder'].items():
                new_state_dict[key[7:]] = value
            encoder.load_state_dict(new_state_dict)
            encoder.forward_features = encoder.forward

        encoders.append(encoder)
    
    return encoders, encoder_types, architectures


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def center_crop_arr(image_arr, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    pil_image = Image.fromarray(image_arr)
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


def preprocess_imgs_vae(imgs):
    # imgs: (B, C, H, W) -> (B, C, H, W), [0, 255] uint8 -> [-1, 1] float32
    return imgs.float() / 127.5 - 1.


def count_trainable_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def normalize_latents(latents, latents_scale, latents_bias):
    return (latents - latents_bias) * latents_scale


def denormalize_latents(latents, latents_scale, latents_bias):
    return latents / latents_scale + latents_bias


@torch.no_grad()
def compute_latent_stats(z, posterior=None):
    """Diagnostics for the VAE latent, used to monitor latent collapse.

    Args:
        z: sampled latent, shape ``(B, C, H, W)``.
        posterior: optional ``DiagonalGaussianDistribution`` exposing ``.mean``
            and ``.var`` (same shape as ``z``).

    Returns:
        (stats, per_dim_std) where ``stats`` is a dict of 0-dim scalar tensors
        (still on device, un-reduced across ranks) and ``per_dim_std`` is the
        per-channel std over ``(B, H, W)``, shape ``(C,)``.

    The dict keys follow a ``*_min`` / ``*_max`` / ``*_mean`` / ``*_std`` suffix
    convention so callers can reduce them across processes appropriately.
    """
    zf = z.detach().float()
    c = zf.shape[1]
    # per-channel std over batch + spatial dims -> (C,)
    per_dim_std = zf.transpose(0, 1).reshape(c, -1).std(dim=1)

    stats = {
        "latent/z_min": zf.min(),
        "latent/z_max": zf.max(),
        "latent/z_mean": zf.mean(),
        "latent/z_std": zf.std(),
        "latent/z_perdim_std_min": per_dim_std.min(),
        "latent/z_perdim_std_max": per_dim_std.max(),
        "latent/z_perdim_std_mean": per_dim_std.mean(),
        "latent/z_perdim_std_std": per_dim_std.std(),
    }

    if posterior is not None:
        m = posterior.mean.detach().float()
        v = posterior.var.detach().float()
        stats.update({
            "latent/mean_min": m.min(),
            "latent/mean_max": m.max(),
            "latent/mean_mean": m.mean(),
            "latent/mean_std": m.std(),
            "latent/var_min": v.min(),
            "latent/var_max": v.max(),
            "latent/var_mean": v.mean(),
            "latent/var_std": v.std(),
        })

    return stats, per_dim_std


def reduce_latent_stats(accelerator, stats):
    """Reduce the per-rank scalar latent stats into Python floats.

    ``*_min`` -> global min, ``*_max`` -> global max, everything else
    (``*_mean`` / ``*_std``) -> mean across ranks (an approximation that is fine
    for monitoring, since all ranks use the same local batch size).
    """
    out = {}
    for key, val in stats.items():
        gathered = accelerator.gather(val.reshape(1))
        if key.endswith("_min"):
            out[key] = gathered.min().item()
        elif key.endswith("_max"):
            out[key] = gathered.max().item()
        else:
            out[key] = gathered.mean().item()
    return out
