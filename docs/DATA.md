# Data preparation

The two stages want the same images in different forms, because they have
different access patterns.


| Stage       | Format                                   | Why                                                     |
| ----------- | ---------------------------------------- | ------------------------------------------------------- |
| End-to-end  | one HDF5 of raw images                   | the VAE is being trained, so it needs pixels every step |
| Prior-only  | image dir + cached `.npy` latent moments | the VAE is frozen, so its output is precomputed once    |


The prior-only stage is the long run (4M steps), which is exactly why its latents
are cached: re-encoding on the fly would put a frozen VAE forward pass in the
critical path for 800 epochs.

## ImageNet

You need the ILSVRC-2012 train split (~1.28M images) and, optionally, 5k
validation images for in-training reconstruction metrics. Download from
[image-net.org](https://www.image-net.org/download.php); it is not
redistributable here.

### End-to-end stage: HDF5

```bash
python tools/build_image_h5.py --source /path/to/imagenet/train \
    --dest data --resolution 256
```

For 512, pass `--resolution 512` and use a separate output directory
(`data512`), then point the training script at it with `DATA_DIR=data512`.

### Prior-only stage: images, then cached latents

This is two steps. First convert ImageNet into a directory of cropped images:

```bash
python tools/build_image_dir.py \
    --source=/path/to/imagenet/train \
    --dest=data/imagenet256/images \
    --resolution=256x256 --transform=center-crop-dhariwal
```

Then cache the latents of your trained VAE. `run_prior_only_pipeline.sh encode` does
this for you and, importantly, names the cache directory after both the VAE and
the resolution:

```bash
cd Prior_only
E2E_CKPT=../exps/<e2e-run>/checkpoints/0600000.pt \
  bash scripts/run_prior_only_pipeline.sh extract
bash scripts/run_prior_only_pipeline.sh encode
```

A latent cache is valid for exactly one VAE at one resolution. Reusing a stale
cache does not raise an error — training proceeds and produces a model that
generates noise — so let the script own the directory naming rather than setting
`FEATURE_SUBDIR` by hand.

`--transform=center-crop-dhariwal` is the ADM/Dhariwal crop. It matters: the
official FID reference batches were built with it, and using a different crop
shifts FID by a non-trivial margin while looking perfectly fine to the eye.

## Initial VAE weights

The end-to-end stage starts from an existing VAE. Which one is the largest single
factor in the final number, worth about 0.8 gFID between the two supported
options.


| Config   | Source                                                                                   | Files expected under `pretrained/`                                  |
| -------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `f8d4`   | SD-VAE ([stabilityai/sd-vae-ft-ema](https://huggingface.co/stabilityai/sd-vae-ft-ema)) | `sdvae/sdvae-f8d4.pt`, `sdvae/sdvae-f8d4-discriminator-ckpt.pt`     |
| `f16d32` | VA-VAE ([LightningDiT](https://github.com/hustvl/LightningDiT))                          | `vavae/vavae-f16d32.pt`, `vavae/vavae-f16d32-discriminator-ckpt.pt` |


Training from scratch also works — drop `--vae-ckpt` and
`--disc-pretrained-ckpt` — and costs roughly 0.8 gFID against the `f16d32`
initialization.

The discriminator checkpoint is not optional in practice. Starting the end-to-end
stage with a freshly initialized discriminator against an already-good VAE gives
the generator a meaningless adversarial signal for the first several thousand
steps.

## Alignment encoder

REPA alignment needs DINOv2 ViT-B. Fetch it once, before any distributed launch:

```bash
mkdir -p pretrained
python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')"
```

`utils.py` builds the architecture through `timm` and loads weights from
`pretrained/dinov2_vitb14_pretrain.pth` (override with `DINOV2_WEIGHTS`), rather
than calling `torch.hub.load()` at training time. Hub loading fetches repository
source code over the network on every run, which fails in offline environments
and rate-limits when many ranks start at once.

Other encoders (`mocov3`, `mae`, `jepa`, `clip`) are kept for parity with REPA
and read from `ckpts/`, overridable with `ENCODER_CKPT_DIR`. They are not used by
any recipe here. `dinov1` is referenced by REPA's code but its backbone is not
part of this repository, so requesting it raises `NotImplementedError`.

## Disk budget

Rough numbers at 256, for planning:


| What                              | Size                     |
| --------------------------------- | ------------------------ |
| ImageNet train, original JPEGs    | ~145 GB                  |
| End-to-end HDF5                   | ~140 GB                  |
| Prior-only image directory        | ~70 GB                   |
| Prior-only latent cache, `f8d4`   | ~35 GB                   |
| Prior-only latent cache, `f16d32` | ~70 GB                   |
| FID reference statistics          | 32 MB each, already in `fid_stats/` |


At 512 the image directory and caches grow roughly 4x. The `f16d32` cache is
larger than `f8d4` despite the coarser grid because 32 channels at 16x16 is twice
the data of 4 channels at 32x32.

Both copies of the dataset are needed only if you are training both stages; the
HDF5 can be deleted once the end-to-end stage finishes.