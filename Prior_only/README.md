# `Prior_only/` — prior-only training and sampling

This directory holds the **prior-only** half of the pipeline: training a diffusion
transformer on a *frozen* VAE, plus the sampling and evaluation entrypoints.

It began as a vendored copy of [REPA](https://github.com/sihyun-yu/REPA) and
keeps that repository's model, loss and preprocessing code, so the attribution
and citation at the bottom of this file still apply. The training scripts
themselves have been rewritten for GenFirst.

**Do not follow upstream REPA's instructions for this directory** — the data
layout, the checkpoint format and the VAE handling all differ. The commands you
want are in the top-level docs:

| What | Where |
| --- | --- |
| prior-only training (class-conditional) | [`../README.md`](../README.md#usage) |
| latent caching and data layout | [`../docs/DATA.md`](../docs/DATA.md) |
| FID / sampling | [`../docs/EVAL.md`](../docs/EVAL.md) |
| text-to-image training | [`../docs/T2I.md`](../docs/T2I.md) |
| GenEval / DPG-Bench | [`../docs/T2I_EVAL.md`](../docs/T2I_EVAL.md) |

## Entrypoints

| File | Purpose |
| --- | --- |
| `train_prior_only.py` | prior-only: SiT on a frozen VAE's cached latents |
| `train_t2i_frozen.py` | prior-only: text-conditional MMDiT on a frozen FLUX.1 VAE |
| `generate.py` | class-conditional sampling, with CFG and autoguidance |
| `eval_fid.py` | FID / IS for a folder of samples, against `fid_stats/` |
| `gen_for_bench.py` | sharded text-to-image generation for GenEval / DPG prompt sets |
| `infer_t2i.py` | text-to-image inference for qualitative inspection |

Dataset conversion and VAE latent caching live at the top level, in
`../tools/build_image_dir.py` and `../tools/encode_latents.py`.

Prefer the wrappers in `scripts/`, which set the flags these entrypoints expect:
`run_prior_only_pipeline.sh`, `eval_fid.sh`, `train_t2i_prior_only.sh`.

## Environment

There is a single environment for the whole repository, defined at the top level
(`environment.yml` / `requirements.txt`). Do not create a separate one here.
The text-to-image benchmarks are the one exception — they need their own
environment, see [`../docs/T2I_EVAL.md`](../docs/T2I_EVAL.md).

## Alignment encoders

The class-conditional prior-only trainer (`train_prior_only.py`) always uses a
representation-alignment loss: `--enc-type` picks the frozen encoder (default
`dinov2-vit-b`) and
`--proj-coeff` weights the term. It cannot be turned off. The text-to-image
trainer (`train_t2i_frozen.py`) makes it optional instead, via `--use-repa` /
`--no-use-repa`.

DINOv2 and CLIP download themselves. The other encoders need weights placed in
`ckpts/` (override the directory with `ENCODER_CKPT_DIR`):

- `mocov3`: ViT-B/16 or ViT-L/16 from [RCG](https://github.com/LTH14/rcg) → `ckpts/mocov3_vitb.pth` / `ckpts/mocov3_vitl.pth`
- `jepa`: ViT-H/14 (ImageNet-1K) from [I-JEPA](https://github.com/facebookresearch/ijepa) → `ckpts/ijepa_vith.pth`
- `mae`: ViT-L from [MAE](https://github.com/facebookresearch/mae) → `ckpts/mae_vitl.pth`

`dinov1` is referenced by upstream's encoder list but its model file is not
included here; selecting it raises `NotImplementedError`.

## Acknowledgement

This directory is built on [REPA](https://github.com/sihyun-yu/REPA), which in
turn builds on [DiT](https://github.com/facebookresearch/DiT),
[SiT](https://github.com/willisma/SiT), [edm2](https://github.com/NVlabs/edm2)
and [RCG](https://github.com/LTH14/rcg).

```bibtex
@inproceedings{yu2025repa,
  title={Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think},
  author={Sihyun Yu and Sangkyung Kwak and Huiwon Jang and Jongheon Jeong and Jonathan Huang and Jinwoo Shin and Saining Xie},
  year={2025},
  booktitle={International Conference on Learning Representations},
}
```
