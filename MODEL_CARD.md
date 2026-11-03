---
license: cc-by-nc-4.0
library_name: pytorch
pipeline_tag: text-to-image
tags:
  - diffusion
  - latent-diffusion
  - end-to-end-training
  - variational-autoencoder
  - text-to-image
  - class-conditional-image-generation
  - sit
  - mmdit
---

# GenFirst

Generation Before Reconstruction for Stable End-to-End Latent Generative Modeling.

- Paper: [arXiv:2608.29335](https://arxiv.org/abs/2608.29335)
- Code: [github.com/ByteDance-Seed/GenFirst](https://github.com/ByteDance-Seed/GenFirst)
- Project page: [tom-zgt.github.io/GenFirst](https://tom-zgt.github.io/GenFirst/)

These weights are released under **CC BY-NC 4.0: non-commercial research use
only**, which is stricter than the MIT licence on the code. See
[Licence and third-party terms](#licence-and-third-party-terms) before use.

## What this is

GenFirst trains a latent VAE jointly with a generative prior without the latent
collapse that usually makes such end-to-end training unstable. The method is a
two-phase schedule over the prior and entropy weights: phase 1 lets the prior
shape the latent geometry, phase 2 drops both weights tenfold so the decoder
recovers detail. The result is a latent space that a diffusion prior fits well
while remaining a good autoencoder.

This release covers the **EiT** (diffusion-prior) instantiation. The
autoregressive **EAR** instantiation from the paper is not released.

Two kinds of artifact are published, and they are used together:

1. **End-to-end VAEs** — the encoder/decoder produced by the end-to-end stage,
   each shipped with its latent normalization statistics.
2. **Generative priors** — SiT models for class-conditional generation, and
   MMDiT models for text-to-image, each trained on a frozen VAE from (1).

## Model details

| | |
|---|---|
| Developed by | Guangting Zheng, Yiyuan Zhang, Tao Yang, Yunpeng Chen, Rui Zhu, Jiajun Deng, Yanyong Zhang |
| Model type | Latent diffusion (flow matching / stochastic interpolant) with a jointly trained VAE |
| Prior architecture | SiT-XL and MMDiT-XL: 28 blocks, width 1152, 16 attention heads (~675M parameters) |
| Text encoder (text-to-image) | CLIP-L, frozen, not modified by us |
| Precision | Weights released in fp32; sampling tested in bf16 |
| Framework | PyTorch |
| Inference code | Required — see [Usage](#usage). These are raw `.pt` checkpoints, not `diffusers` pipelines. |

### Class-conditional priors

`SiT-XL` on a frozen end-to-end VAE. Conditioning is a single integer class
label in `[0, 999]` following the standard 1000-class label ordering; there is
no text input. Epoch counts are prior-only training epochs.

| Checkpoint | Resolution | VAE | gFID | IS | gFID (guided) | IS (guided) | Download |
|---|---|---|---|---|---|---|---|
| EiT, 80 epochs | 256x256 | `f16d32` | 2.79 | 175.4 | 1.38 | 271.3 | [checkpoint](https://huggingface.co/ByteDance-Seed/GenFirst/blob/main/class-conditional/imagenet256/eit-imagenet256-80ep.pt) · [weak-20ep](https://huggingface.co/ByteDance-Seed/GenFirst/blob/main/class-conditional/imagenet256/eit-imagenet256-weak-20ep.pt) |
| EiT, 480 epochs | 256x256 | `f16d32` | 1.60 | 216.8 | 0.988 | 270.7 | [checkpoint](https://huggingface.co/ByteDance-Seed/GenFirst/blob/main/class-conditional/imagenet256/eit-imagenet256-480ep.pt) · [weak-20ep](https://huggingface.co/ByteDance-Seed/GenFirst/blob/main/class-conditional/imagenet256/eit-imagenet256-weak-20ep.pt) |
| EiT, 800 epochs | 256x256 | `f16d32` | **1.45** | 226.7 | **0.969** | 283.4 | [checkpoint](https://huggingface.co/ByteDance-Seed/GenFirst/blob/main/class-conditional/imagenet256/eit-imagenet256-800ep.pt) · [weak-20ep](https://huggingface.co/ByteDance-Seed/GenFirst/blob/main/class-conditional/imagenet256/eit-imagenet256-weak-20ep.pt) |
| EiT, 80 epochs | 512x512 | `f16d32` | – | – | 1.63 | 278.9 | [checkpoint](https://huggingface.co/ByteDance-Seed/GenFirst/blob/main/class-conditional/imagenet512/eit-imagenet512-80ep.pt) |
| EiT, 400 epochs | 512x512 | `f16d32` | – | – | **1.23** | 286.1 | [checkpoint](https://huggingface.co/ByteDance-Seed/GenFirst/blob/main/class-conditional/imagenet512/eit-imagenet512-400ep.pt) |

The 512 models reuse the 256 `f16d32` VAE unchanged and frozen; only the prior
is trained at 512.

### Text-to-image priors

`MMDiT-XL/2` with a frozen CLIP-L text encoder, on a frozen end-to-end FLUX
VAE, at 512x512. Scores are overall GenEval and DPG-Bench, with no prompt
rewriting. The two variants differ only in which VAE they were trained on.

| Checkpoint | Resolution | GenEval | DPG-Bench | Download |
|---|---|---|---|---|
| EiT T2I, variant A | 512x512 | **0.8987** | **82.60** | [checkpoint](https://huggingface.co/ByteDance-Seed/GenFirst/blob/main/text-to-image/variant-a/eit-t2i-variant-a-512.pt) · [VAE](https://huggingface.co/ByteDance-Seed/GenFirst/tree/main/vae/e2e-flux-vae-sit-imagenet-s2) |
| EiT T2I, variant B | 512x512 | 0.8920 | 82.06 | [checkpoint](https://huggingface.co/ByteDance-Seed/GenFirst/blob/main/text-to-image/variant-b/eit-t2i-variant-b-512.pt) · [VAE](https://huggingface.co/ByteDance-Seed/GenFirst/tree/main/vae/e2e-flux-vae-mmdit-l2-t2i-s2) |

### Shared visual latents (generation + understanding)

`MMDiT-XL/2` on a frozen Qwen3-VL ViT autoencoder (`--vae-arch qwenvit`) shaped end-to-end with a generative prior plus representation supervision (`SigLIP` or `VLM-NLL`, Table 9 in the paper).

| Checkpoint | VAE bundle | Supervision | Resolution | rFID ↓ | PSNR ↑ | GenEval ↑ | DPG-Bench ↑ | Linear Probe Top-1 (%) ↑ |
|---|---|---|---|---|---|---|---|---|
| [`shared-latents/siglip/eit-qwenvit-siglip-t2i-512.pt`](https://huggingface.co/ByteDance-Seed/GenFirst/blob/main/shared-latents/siglip/eit-qwenvit-siglip-t2i-512.pt) | [`vae/qwenvit-siglip-b256-s2/`](https://huggingface.co/ByteDance-Seed/GenFirst/tree/main/vae/qwenvit-siglip-b256-s2) | E2E + SigLIP | 512x512 | **0.39** | **27.64** | **0.90** | **83.76** | 79.25 |
| [`shared-latents/vlm-nll/eit-qwenvit-vlm-nll-t2i-512.pt`](https://huggingface.co/ByteDance-Seed/GenFirst/blob/main/shared-latents/vlm-nll/eit-qwenvit-vlm-nll-t2i-512.pt) | [`vae/qwenvit-nlltrunk-s2/`](https://huggingface.co/ByteDance-Seed/GenFirst/tree/main/vae/qwenvit-nlltrunk-s2) | E2E + VLM-NLL | 512x512 | 0.46 | 27.19 | 0.89 | 83.49 | **81.69** |

### Unified text-image modeling (`JointMMDiT-XL/2`)

Bidirectional block-causal `JointMMDiT-XL/2` trained on the jointly E2E-shaped text VAE (Cola) and image VAE (FLUX.1) (`vae/e2e-joint-gan-l2-330k/`, Table 10 in the paper). The phase-3 SFT checkpoint (`-512-sft.pt`) is used for text-to-image (GenEval / DPG-Bench), and the phase-2 pre-SFT checkpoint (`-512.pt`) is used for image-to-text COCO captioning.

| Checkpoints | Setting | GenEval ↑ | DPG-Bench ↑ | CIDEr ↑ | BLEU-4 ↑ | METEOR ↑ | ROUGE-L ↑ | SPICE ↑ | CLIPScore ↑ |
|---|---|---|---|---|---|---|---|---|---|
| [`eit-joint-e2e-text-e2e-img-512-sft.pt`](https://huggingface.co/ByteDance-Seed/GenFirst/blob/main/unified-text-image/e2e-joint/eit-joint-e2e-text-e2e-img-512-sft.pt) (T2I) & [`eit-joint-e2e-text-e2e-img-512.pt`](https://huggingface.co/ByteDance-Seed/GenFirst/blob/main/unified-text-image/e2e-joint/eit-joint-e2e-text-e2e-img-512.pt) (I2T) | E2E-joint text + image VAEs ([`vae/e2e-joint-gan-l2-330k/`](https://huggingface.co/ByteDance-Seed/GenFirst/tree/main/vae/e2e-joint-gan-l2-330k)) | **86.57** | **72.68** | **33.28** | **7.76** | **17.74** | **34.16** | **11.81** | **57.97** |

### End-to-end VAEs

Reconstruction quality of the posterior mean, measured at the resolution the
VAE is used at.

| VAE | Downsample / channels | Initialized from | rFID | PSNR | SSIM | LPIPS | Download |
|---|---|---|---|---|---|---|---|
| `f8d4` | 8x, 4ch | SD-VAE (`stabilityai/sd-vae-ft-ema`) | **0.57** | 25.13 | 0.71 | 0.14 | [Hugging Face](https://huggingface.co/ByteDance-Seed/GenFirst/tree/main/vae/e2e-f8d4) |
| `f16d32` (decoder-ft, @256) | 16x, 32ch | VA-VAE (LightningDiT) | 0.36 | 25.82 | 0.74 | 0.12 | [Hugging Face](https://huggingface.co/ByteDance-Seed/GenFirst/tree/main/vae/e2e-f16d32) |
| `f16d32` (no dec-ft, @512) | 16x, 32ch | VA-VAE (LightningDiT) | – | – | – | – | [Hugging Face](https://huggingface.co/ByteDance-Seed/GenFirst/tree/main/vae/e2e-f16d32-nodecft) |
| `f8d16` variant A | 8x, 16ch | FLUX.1 VAE | 0.23 | 29.43 | 0.85 | 0.07 | [Hugging Face](https://huggingface.co/ByteDance-Seed/GenFirst/tree/main/vae/e2e-flux-vae-sit-imagenet-s2) |
| `f8d16` variant B | 8x, 16ch | FLUX.1 VAE | 0.21 | 29.37 | 0.85 | 0.07 | [Hugging Face](https://huggingface.co/ByteDance-Seed/GenFirst/tree/main/vae/e2e-flux-vae-mmdit-l2-t2i-s2) |
| `qwenvit-siglip-b256-s2` | 8x, 16ch | Qwen3-VL ViT + ViT decoder (SigLIP) | 0.39 | 27.64 | – | – | [Hugging Face](https://huggingface.co/ByteDance-Seed/GenFirst/tree/main/vae/qwenvit-siglip-b256-s2) |
| `qwenvit-nlltrunk-s2` | 8x, 16ch | Qwen3-VL ViT + ViT decoder (VLM-NLL) | 0.46 | 27.19 | – | – | [Hugging Face](https://huggingface.co/ByteDance-Seed/GenFirst/tree/main/vae/qwenvit-nlltrunk-s2) |
| `e2e-joint-gan-l2-330k` | 8x, 16ch (image) + 16d (text) | FLUX.1 VAE + Cola Text VAE | – | – | – | – | [Hugging Face](https://huggingface.co/ByteDance-Seed/GenFirst/tree/main/vae/e2e-joint-gan-l2-330k) |
| `f8d32` (`e2e-vae-flux2-sitxl2-auxkl2p5e8-s3-20k`, @512) | 8x, 32ch | FLUX.2 VAE | **0.041** | **36.46** | **0.953** | **0.026** | [Hugging Face](https://huggingface.co/ByteDance-Seed/GenFirst/tree/main/vae/e2e-vae-flux2-sitxl2-auxkl2p5e8-s3-20k) |

**A VAE and its latent statistics must be kept together.** The prior was
trained on latents normalized by those exact statistics; pairing a VAE with the
wrong statistics file produces noise, not a degraded image. Each VAE directory
is laid out as:

```
<name>/
  <name>.pt                     VAE weights
  <name>.safetensors            VAE weights in diffusers safetensors format (for FLUX.2 release)
  <name>-latents-stats.pt       latent normalization (scale + bias)
  <name>-discriminator-ckpt.pt  discriminator, only needed to resume end-to-end training
  config.json / vae_params.json diffusers config and explicit scaling_factor / shift_factor (for FLUX.2 release)
```

For `vae/e2e-vae-flux2-sitxl2-auxkl2p5e8-s3-20k`, the per-channel `scaling_factor` and `shift_factor` parameters are:

```bash
--vae_pth=vae/e2e-vae-flux2-sitxl2-auxkl2p5e8-s3-20k/e2e-vae-flux2-sitxl2-auxkl2p5e8-s3-20k.safetensors
--model.vae.scaling_factor=[0.887221,0.886926,0.887096,0.881914,0.886901,0.887307,0.886685,0.886833,0.886540,0.886646,0.887031,0.887073,0.887381,0.886768,0.887299,0.885674,0.886466,0.886795,0.886979,0.886519,0.886949,0.886949,0.887036,0.886828,0.887344,0.887004,0.886816,0.886799,0.887354,0.887523,0.887012,0.886643]
--model.vae.shift_factor=[-0.000623,0.000143,0.000282,0.001599,-0.001093,-0.000477,0.000855,-0.000830,-0.000883,0.001076,-0.000417,0.000200,0.000626,-0.000649,-0.000137,0.001453,-0.000816,-0.000307,-0.000290,0.000662,0.000484,0.000993,0.000574,-0.000139,-0.000914,0.000578,-0.000691,-0.000323,0.000619,-0.000606,-0.000372,0.000437]
```

## Usage

These checkpoints need the training repository; there is no `diffusers`
integration and no `from_pretrained` entry point.

```bash
git clone https://github.com/ByteDance-Seed/GenFirst
cd GenFirst
pip install -r requirements.txt          # install torch first
```

Class-conditional sampling, writing a small batch of images rather than a full
evaluation run:

```bash
cd Prior_only
python generate.py \
    --model SiT-XL/1 --resolution 256 \
    --ckpt /path/to/prior.pt \
    --vae-arch f16d32 \
    --vae-ckpt /path/to/e2e-f16d32.pt \
    --latents-stats /path/to/e2e-f16d32-latents-stats.pt \
    --num-fid-samples 64 --label-sampling random \
    --cfg-scale 1.5 --sample-dir samples
```

`--model` must match the VAE: `SiT-XL/2` for `f8d4`, `SiT-XL/1` for `f16d32`.
Text-to-image sampling, the guidance settings that reproduce the numbers above,
and the full evaluation harness are documented in the repository under `docs/`.

## Evaluation

Reported metrics and their protocols:

- **gFID / IS** — 50,000 generated samples scored with
  [torch-fidelity](https://github.com/LTH14/torch-fidelity) against the
  reference statistics used by [MAR](https://github.com/LTH14/mar), at the
  stated resolution. Sampling is SDE with 250 steps. Guided rows use
  [autoguidance](https://arxiv.org/abs/2406.02507) (pointing away from an
  early checkpoint of the same run) with CFG on top, except the 512 rows, which
  use plain CFG.
- **rFID / PSNR / SSIM / LPIPS** — posterior-mean reconstructions over a
  50,000-image validation split, following
  [latent-diffusion](https://github.com/CompVis/latent-diffusion). rFID here is
  measured against those validation images, not a training-set reference, so it
  is not comparable to numbers computed the other way.
- **GenEval / DPG-Bench** — overall scores, Euler sampling with 50 steps and
  CFG 7.0 on the interval `[0.35, 1.0]`, no prompt rewriting.

Numbers were produced with the versions pinned in `environment.yml`. FID is
sensitive to reference statistics and to the resizing path, so comparisons
against other work are only meaningful under the same protocol.

## Training

Training proceeds in two stages. The end-to-end stage trains the VAE jointly
with a prior under the two-phase GenFirst schedule (global batch 256; 400k
steps at the high weight, then 200k at the low one). The prior-only stage then
freezes the VAE, caches its latents, and trains from scratch the prior that is
actually evaluated (4M steps at 256, 2M at 512). An optional 10k-step
decoder-only finetune between the stages improves unguided gFID by roughly 0.2.
Text-to-image follows the same structure, with an `MMDiT-L/2` prior shaping the
latent during the end-to-end stage and the released `MMDiT-XL/2` trained on the
frozen result.

Training data consists of publicly available datasets. Individual corpora are
not enumerated here; each is subject to its own licence and terms of use, which
govern independently of the licence on these weights.

## Intended use

Intended for research: reproducing the paper, studying end-to-end VAE and prior
training, latent-space analysis, and use as a baseline or initialization for
further non-commercial research.

**Out of scope.** Any commercial use. Generating content about real,
identifiable people. Generating sexual, violent, hateful, or otherwise harmful
imagery. Any use presenting outputs as real photographs or as factual records.
Any safety-critical or decision-making pipeline. These models produce synthetic
images with no factual grounding and must not be treated as a source of truth.

No safety filter, watermarking, or content classifier is included. Anyone
deploying these weights in a setting where outputs reach other people is
responsible for adding those controls.

## Limitations and biases

- The class-conditional models generate from a fixed 1000-class label set and
  cannot follow text. Their outputs reflect the visual distribution and the
  category taxonomy they were fit to, including the geographic and cultural
  skew documented for that style of curated image collection.
- The text-to-image models inherit the biases of both their training data and
  the frozen CLIP-L text encoder. Expect uneven representation across
  demographics, professions, and cultures, and expect prompt-following to
  degrade on long, compositional, or counting-heavy prompts. GenEval and
  DPG-Bench scores measure prompt alignment on those benchmarks only and do not
  certify safety or fairness.
- Text rendering, hands, and fine structure remain unreliable, as with
  comparable models at this scale.
- Metrics reported here are distribution-level. A good FID says nothing about
  the quality or appropriateness of any single sample.
- The 512 class-conditional models are reported with guidance only; their
  unguided behaviour is untuned.
- A VAE paired with mismatched latent statistics fails silently in the sense
  that it produces output rather than an error. Check the pairing.

## Licence and third-party terms

**Weights: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) —
non-commercial research use only, with attribution.** This is stricter than the
MIT licence on the code and is the binding constraint on these artifacts.

The code in the GitHub repository is MIT, but the repository as a whole is not
usable commercially as it stands: the data preprocessing derives from
[NVIDIA EDM2](https://github.com/NVlabs/edm2) under CC BY-NC-SA 4.0, and
several vendored encoders and samplers trace back to Meta projects under
CC BY-NC 4.0. The repository's `NOTICE` lists this component by component.

These weights are also derivative of upstream models, each carrying its own
terms that continue to apply and that you should verify for your use:

| Component | Upstream |
|---|---|
| `f8d4` VAE initialization | [stabilityai/sd-vae-ft-ema](https://huggingface.co/stabilityai/sd-vae-ft-ema) |
| `f16d32` VAE initialization | VA-VAE, from [LightningDiT](https://github.com/hustvl/LightningDiT) |
| `f8d16` VAE initialization | FLUX.1 VAE, [black-forest-labs](https://huggingface.co/black-forest-labs) |
| Text encoder | [OpenAI CLIP](https://github.com/openai/CLIP) |
| Prior architecture lineage | [REPA](https://github.com/sihyun-yu/REPA), [REPA-E](https://github.com/End2End-Diffusion/REPA-E), and DiT/SiT before them |

Training datasets carry their own terms independently of this licence, and
those terms may further restrict what you may do with a model trained on them.

## Security

A `.pt` checkpoint is a pickle: loading one executes arbitrary Python. These
checkpoints store an `args` namespace alongside the tensors, so they cannot be
loaded with `weights_only=True` and require `weights_only=False` on torch 2.6
and newer. Load checkpoints only from sources you trust, and verify you are
downloading from the official repository.

To report a security vulnerability, follow `SECURITY.md` in the code
repository: contact the ByteDance security team rather than opening a public
issue.

## Citation

```bibtex
@article{zheng2026genfirst,
  title={GenFirst: Generation Before Reconstruction for Stable End-to-End Latent Generative Modeling},
  author={Zheng, Guangting and Zhang, Yiyuan and Yang, Tao and Chen, Yunpeng and Zhu, Rui and Deng, Jiajun and Zhang, Yanyong},
  journal={arXiv preprint arXiv:2608.29335},
  year={2026}
}
```

## Contact

Questions through [GitHub Issues](https://github.com/ByteDance-Seed/GenFirst/issues) or
mail to [zgt@mail.ustc.edu.cn](mailto:zgt@mail.ustc.edu.cn). Security reports
go to the address in `SECURITY.md`, not to Issues.
