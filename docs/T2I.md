# Text-to-image

The same GenFirst recipe transfers to text conditioning, with the FLUX.1-dev VAE in place of the class-conditional VAEs and MMDiT in place of SiT.

## The two stages

The end-to-end stage jointly trains the FLUX.1-dev VAE with a generative prior.
There are two variants, and they differ only in what supplies the prior gradient:


| Variant | Prior     | Data                   | Notes                                                |
| ------- | --------- | ---------------------- | ---------------------------------------------------- |
| A       | SiT-XL/2  | ImageNet               | trains the prior on the ImageNet                     |
| B       | MMDiT-L/2 | text-to-image datasets | trains the prior on the target distribution directly |


The prior-only stage freezes whichever VAE came out of the end-to-end stage and trains the target MMDiT-XL/2 on the full text-to-image datasets. 

## Data

The paper trains on publicly available datasets.

### Caption source

Several of these datasets ship with weak or absent captions, and the paper's runs use captions regenerated with a vision-language model (Qwen2.5-VL) rather than the originals. 

### Expected format

Training reads parquet shards. Build them from a directory of images plus a captions file:

```bash
python tools/build_t2i_parquet.py \
    --images-dir /path/to/images \
    --captions captions.jsonl \
    --out-dir /path/to/parquet_shards \
    --caption-column caption \
    --rows-per-file 128
```

Schema:


| Column            | Type     | Notes                                                                             |
| ----------------- | -------- | --------------------------------------------------------------------------------- |
| `image`           | `binary` | JPEG/PNG bytes (one image per row)                                                |
| caption column(s) | `string` | e.g. `caption`, `prompt`, `text` — names are configurable via `--caption-columns` |


Caption column names are configurable rather than hardcoded, so you can point the
dataset at whichever column your shards actually carry.

## Training

```bash
# end-to-end stage: joint FLUX VAE + MMDiT-L/2 or SiT-XL/2 prior (2-phase GenFirst schedule, 256)
PHASE=1 DATA=/path/to/parquet bash scripts/train_t2i_e2e.sh
PHASE=2 bash scripts/train_t2i_e2e.sh

# extract VAE from the final end-to-end checkpoint
python save_vae_weights.py \
    --e2e-ckpt exps/t2i-e2e-flux-vae-mmdit-l2-phase2-256-c0.01/checkpoints/0200000.pt \
    --vae-name e2e-flux-vae \
    --save-dir pretrained/extracted_vae

# prior-only stage: freeze the VAE, train the target MMDiT (3 phases)
cd Prior_only
RUN=e2e E2E_VAE_CKPT=../pretrained/extracted_vae/e2e-flux-vae/e2e-flux-vae.pt \
  DATA=/path/to/parquet PHASE=1 bash scripts/train_t2i_prior_only.sh
RUN=e2e DATA=/path/to/parquet PHASE=2 bash scripts/train_t2i_prior_only.sh
RUN=e2e DATA_FT=/path/to/sft-parquet PHASE=3 bash scripts/train_t2i_prior_only.sh
```

The prior-only phases are what the released checkpoints were trained with:


| Phase | Resolution | Steps | Global batch | Data                    |
| ----- | ---------- | ----- | ------------ | ----------------------- |
| 1     | 256x256    | 200K  | 1536         | text-to-image mixture   |
| 2     | 512x512    | 80K   | 512          | text-to-image mixture   |
| 3     | 512x512    | 40K   | 512          | high-quality SFT subset |


Phase 1's batch is large; `BATCH_SIZE` overrides all phases and `S1_BATCH` / `S2_BATCH` / `S3_BATCH` override one. Phase 3 is a finetune on a smaller, higher-quality caption set (`DATA_FT`), and defaults to `DATA` if unset.

To use the 32-channel `f8d32` FLUX.2 VAE instead of FLUX.1, pass `VAE_ARCH=flux2` to `scripts/train_t2i_e2e.sh` or `Prior_only/scripts/train_t2i_prior_only.sh`, or train the three-stage aspect-ratio-bucketed FLUX.2 E2E VAE via `scripts/train_e2e_vae.sh` (`train_e2e_vae.py`).

### Text encoder

CLIP-L/14 is the default. 

## Evaluation

Two compositional benchmarks, both in `eval_bench/`. **GenEval** measures object presence, counts, colours and relative position with an object detector; the paper reports **0.90**. **DPG-Bench** measures dense prompt following with a VQA model over long, multi-clause prompts.

```bash
cd eval_bench
TORCH_CUDA_ARCH_LIST=8.0 bash setup_eval_env.sh   # one-time

export CKPT=/path/to/eit-t2i-variant-a.pt
export VAE_CKPT=/path/to/e2e-flux-vae-sit-imagenet-s2.pt
export LATENTS_STATS=/path/to/e2e-flux-vae-sit-imagenet-s2-latents-stats.pt
export TEXT_ENCODER_NAME=openai/clip-vit-large-patch14
bash scripts/eval_geneval.sh
bash scripts/eval_dpg.sh
```

Both stacks are heavy and version-sensitive, and should go in a separate environment from training. See [T2I_EVAL.md](T2I_EVAL.md) for the full setup, external assets, output formats, runtimes and eval-only dependencies.

## Inference

To generate images from prompts with a trained model, without scoring:

```bash
cd Prior_only
python infer_t2i.py \
    --ckpt /path/to/eit-t2i-variant-a.pt \
    --vae-ckpt /path/to/e2e-flux-vae-sit-imagenet-s2.pt \
    --latents-stats /path/to/e2e-flux-vae-sit-imagenet-s2-latents-stats.pt \
    --text-encoder-name openai/clip-vit-large-patch14 \
    --out exps/<run>/infer \
    --prompts "a photograph of an astronaut riding a horse||an oil painting of a lighthouse during a storm"
```

