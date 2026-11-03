# Text-to-image evaluation

Two compositional benchmarks, both driven from `eval_bench/`. Neither is a distribution distance like FID — both check whether the image actually contains what the prompt asked for.


| Benchmark | What it measures                                     | How                                           | Paper result |
| --------- | ---------------------------------------------------- | --------------------------------------------- | ------------ |
| GenEval   | object presence, counts, colours, relative position  | Mask2Former detector + CLIP colour classifier | **0.90**     |
| DPG-Bench | dense prompt following on long, multi-clause prompts | mPLUG VQA over 4 samples per prompt           | 82.60        |


## Setup

Run once, before either benchmark:

```bash
cd eval_bench
TORCH_CUDA_ARCH_LIST=8.0 bash setup_eval_env.sh
```

Set `TORCH_CUDA_ARCH_LIST` to your GPU (V100 `7.0`, A100 `8.0`, A10 `8.6`, H100 `9.0`) — `mmcv` compiles CUDA ops and needs it.

**Install this into a separate environment from training.** Both benchmark stacks are version-sensitive and conflict with the training environment. The setup script deliberately does not touch `torch`/`torchvision`: `mmcv` is compiled against whatever torch is already present.

Useful flags: `SKIP_MMCV_BUILD=1` and `SKIP_DOWNLOADS=1` to re-run parts of it.

## GenEval

Set the released prior, its matching released VAE bundle, and the frozen CLIP-L
encoder. The old training checkpoint records machine-local asset paths, so the
three overrides below make it portable; no checkpoint conversion is required.

```bash
cd eval_bench
export CKPT=/path/to/eit-t2i-variant-a.pt
export VAE_CKPT=/path/to/e2e-flux-vae-sit-imagenet-s2.pt
export LATENTS_STATS=/path/to/e2e-flux-vae-sit-imagenet-s2-latents-stats.pt
export TEXT_ENCODER_NAME=openai/clip-vit-large-patch14
bash scripts/eval_geneval.sh
```

For variant B, keep the command unchanged and replace the prior plus the two VAE
paths with the `e2e-flux-vae-mmdit-l2-t2i-s2` release.

Generation is sharded one process per GPU (`NGPU`, default 8) and skips images that already exist, so an interrupted run can simply be restarted. Scoring then runs on a single GPU.

Overridable: `VAE_CKPT`, `LATENTS_STATS`, `TEXT_ENCODER_NAME`, `NGPU`, `NPP`
(samples per prompt, default 4), `STEPS`, `CFG`, `BS`, `OUT`, `META`,
`MODEL_CONFIG`, `DETECTOR_CKPT`, `OBJECT_NAMES`, `CLIP_PATH`.

## DPG-Bench

```bash
cd eval_bench
# Reuse the four exported model paths from the GenEval example.
bash scripts/eval_dpg.sh
```

## Qualitative inspection

To generate images from prompts without scoring anything:

```bash
cd Prior_only
python infer_t2i.py \
    --ckpt /path/to/eit-t2i-variant-a.pt \
    --vae-ckpt /path/to/e2e-flux-vae-sit-imagenet-s2.pt \
    --latents-stats /path/to/e2e-flux-vae-sit-imagenet-s2-latents-stats.pt \
    --text-encoder-name openai/clip-vit-large-patch14 \
    --out exps/<run>/infer \
    --prompts "a photograph of an astronaut riding a horse||an oil painting of a lighthouse during a storm" \
    --cfg-scale 4.0 --num-steps 50 --seed 0
```

Prompts are `||`-separated, or use `--prompt-file` with one per line. Model
architecture is restored from `ckpt["args"]`; the explicit asset overrides
replace paths from the original training machine. Writes individual PNGs plus a
contact-sheet grid.

