# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

"""Held-out reconstruction + per-channel latent SNR check for a VAE.

    python scripts/eval_vae_recon.py --label flux2_s3_20k --arch flux2 \
        --vae pretrained/extracted_vae/e2e-vae-flux2-sitxl2-auxkl2p5e8-s3-20k/e2e-vae-flux2-sitxl2-auxkl2p5e8-s3-20k.safetensors \
        --data /path/to/imagenet/val --res 256 512 --out recon_eval/subset

--vae is an exported state_dict (.safetensors / .pt) or a diffusers `vae` folder.
Images: the first --per-class images of every ImageNet val class, short-side
resize + center crop. Reports PSNR / SSIM / LPIPS / rFID (posterior mode) and
per-channel latent stats: std of the posterior mean over data, mean posterior
std, and their ratio (channel SNR).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loss.lpips import LPIPS  # noqa: E402
from models.autoencoder import vae_models  # noqa: E402


class ValImages(Dataset):
    def __init__(self, root, per_class, res):
        self.paths = []
        for c in sorted(os.listdir(root)):
            cdir = os.path.join(root, c)
            if not os.path.isdir(cdir):
                continue
            files = sorted(os.listdir(cdir))[:per_class]
            self.paths += [os.path.join(cdir, f) for f in files]
        self.res = res

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB")
        w, h = im.size
        s = self.res / min(w, h)
        im = im.resize((max(self.res, round(w * s)), max(self.res, round(h * s))), Image.BICUBIC)
        w, h = im.size
        l, t = (w - self.res) // 2, (h - self.res) // 2
        im = im.crop((l, t, l + self.res, t + self.res))
        return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)


def load_vae(arch, path, device):
    if arch in ("flux", "flux2") and (os.path.isdir(path) or path.endswith(".safetensors")):
        vae = vae_models[arch](pretrained_path=path)
    else:
        vae = vae_models[arch](pretrained_path=None)
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "vae" in sd and not any(k.startswith("vae.") for k in sd):
            sd = sd["vae"]
        missing, unexpected = vae.load_state_dict(sd, strict=False)
        assert not [k for k in missing if not k.startswith("bn_running")], f"missing keys: {missing[:5]}"
        assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    return vae.to(device).eval().requires_grad_(False)


def save_pngs(x_uint8, out_dir, start):
    os.makedirs(out_dir, exist_ok=True)
    for j, img in enumerate(x_uint8.permute(0, 2, 3, 1).cpu().numpy()):
        Image.fromarray(img).save(os.path.join(out_dir, f"{start + j:05d}.png"))


@torch.no_grad()
def evaluate(vae, loader, res, out_dir, device, lpips_fn):
    ref_dir, rec_dir = os.path.join(out_dir, f"ref_{res}"), os.path.join(out_dir, f"rec_{res}")
    psnr, ssim, lp, n = 0.0, 0.0, 0.0, 0
    s_mu = s_mu2 = s_var = None
    cnt = 0
    from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
    for x in loader:
        x = x.to(device)
        xin = x.float() / 127.5 - 1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            post = vae.encode(xin)
            rec = vae.decode(post.mode()).sample
        rec = rec.float().clamp(-1, 1)
        mu, var = post.mean.float(), post.var.float()
        if s_mu is None:
            c = mu.shape[1]
            s_mu, s_mu2, s_var = (torch.zeros(c, device=device, dtype=torch.float64) for _ in range(3))
        s_mu += mu.sum((0, 2, 3)).double()
        s_mu2 += (mu ** 2).sum((0, 2, 3)).double()
        s_var += var.sum((0, 2, 3)).double()
        cnt += mu.shape[0] * mu.shape[2] * mu.shape[3]

        a, b = (xin + 1) / 2, (rec + 1) / 2
        mse = ((a - b) ** 2).mean((1, 2, 3))
        psnr += (-10 * torch.log10(mse.clamp_min(1e-10))).sum().item()
        ssim += ssim_fn(b, a, data_range=1.0, reduction="sum").item()
        lp += lpips_fn(xin, rec).sum().item()
        rec_u8 = ((rec + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
        save_pngs(x, ref_dir, n)
        save_pngs(rec_u8, rec_dir, n)
        n += x.shape[0]

    import torch_fidelity
    fid = torch_fidelity.calculate_metrics(input1=rec_dir, input2=ref_dir, cuda=True, fid=True,
                                           verbose=False)["frechet_inception_distance"]
    mean_c = s_mu / cnt
    data_std = (s_mu2 / cnt - mean_c ** 2).clamp_min(0).sqrt()
    post_std = (s_var / cnt).sqrt()
    snr = data_std / post_std
    return {
        "n": n, "psnr": psnr / n, "ssim": ssim / n, "lpips": lp / n, "rfid": fid,
        "latent_mean_abs_max": mean_c.abs().max().item(),
        "data_std_mean": data_std.mean().item(), "data_std_min": data_std.min().item(),
        "post_std_mean": post_std.mean().item(),
        "snr_min": snr.min().item(), "snr_median": snr.median().item(), "snr_max": snr.max().item(),
        "snr_per_channel": [round(v, 3) for v in snr.tolist()],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--arch", required=True, choices=list(vae_models))
    p.add_argument("--vae", required=True)
    p.add_argument("--data", required=True, help="ImageNet validation root directory.")
    p.add_argument("--per-class", type=int, default=5)
    p.add_argument("--res", type=int, nargs="+", default=[256, 512])
    p.add_argument("--batch", type=int, default=25)
    p.add_argument("--out", default="recon_eval/subset")
    args = p.parse_args()

    device = "cuda"
    vae = load_vae(args.arch, args.vae, device)
    lpips_fn = LPIPS().to(device).eval()
    results = {}
    for res in args.res:
        loader = DataLoader(ValImages(args.data, args.per_class, res), batch_size=max(1, args.batch // (res // 256) ** 2),
                            num_workers=8)
        r = evaluate(vae, loader, res, os.path.join(args.out, args.label), device, lpips_fn)
        results[res] = r
        print(f"[{args.label}] res {res}: " + " ".join(
            f"{k}={v:.4f}" for k, v in r.items() if isinstance(v, float)), flush=True)
    with open(os.path.join(args.out, f"{args.label}.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
