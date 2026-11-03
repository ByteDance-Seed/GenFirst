# Joint image-text modeling

One block-causal MMDiT that models images and text in the same latent sequence, in both directions. The image branch reuses this repo's VAE; text gets its own VAE, so a caption becomes a sequence of continuous latents rather than tokens, and both modalities can be denoised by the same flow-matching objective.

A direction is drawn per *sample*, so one batch carries both:

- `t2i` — text is the clean condition, the image latent is denoised.
- `i2t` — the image latent is the clean condition, text is denoised.

The text stream is block-causal (bidirectional inside a block, causal across blocks) in both directions, so the text representation is computed the same way whichever way the model is used. Attention never crosses samples, and the conditioning modality never attends to the generated one.

Like the image-only recipe, this runs in two stages: an **end-to-end** stage that exists to reshape the text VAE and throws its DiT away, then a **prior-only** stage that freezes both VAEs and trains a new model.

## Prerequisites

**Text VAE.** The text branch uses the Text VAE from [Cola DLM](https://github.com/ByteDance-Seed/Cola-DLM) (Apache-2.0). Its modeling code is vendored under `models/cola_text_vae/`, so only the weights need downloading. The 1.8B DiT prior in that release is *not* used, so skip it:

```bash
pip install -U huggingface_hub
huggingface-cli download ByteDance-Seed/Cola-DLM \
    --include 'cola_dlm/cola_vae/*' 'tokenizer.json' \
    --local-dir pretrained/cola-dlm
```

That gives the layout the scripts expect (`COLA_ROOT=pretrained/cola-dlm`):

```
pretrained/cola-dlm/cola_dlm/cola_vae/{config.json,model.safetensors}
pretrained/cola-dlm/tokenizer.json
```

The tokenizer is OLMo 2's, with a 100,278-entry vocabulary and end-of-text at id 100257. `CaptionCollator` pads each caption's last block with end-of-text rather than `<|pad|>`: Cola's no-padding layout means its VAE never saw `<|pad|>` as a reconstruction target and decodes it to ordinary words.

**Image VAE.** Either a FLUX.1 VAE (as in [T2I.md](T2I.md)) or one reshaped by the T2I end-to-end stage. 

**Data.** The same parquet caption data as the T2I recipe; see [T2I.md](T2I.md).

## Stage 1 — end-to-end

Trains the text VAE jointly with the joint MMDiT, so the text latent space is shaped by the diffusion prior instead of only by reconstruction:

Two phases, as in the image-only and T2I recipes: the prior weight is 0.1 for the first 200K steps and 0.01 for another 100K.

```bash
PHASE=1 DATA=/path/to/parquet_roots bash scripts/train_joint_e2e.sh
PHASE=2 DATA=/path/to/parquet_roots bash scripts/train_joint_e2e.sh
```

The image branch is frozen by default (`IMAGE_VAE_MODE=frozen`), which only runs its encoder. `e2e-recon` / `e2e-gan` train the image VAE too, which roughly doubles step time and peak memory.

Then extract the reshaped VAEs:

```bash
python save_joint_vae_weights.py \
    --joint-ckpt exps/joint-e2e-256-JointMMDiT-L-2/checkpoints/0300000.pt \
    --name e2e-joint-l2 --save-dir extracted_vae
```



## Stage 2 — prior-only

Both VAEs frozen, a fresh joint MMDiT trained on them, in three phases warm-started by weights only:

the same schedule the image-only prior-only stage uses:


| Phase | Resolution | Steps | LR   | Data         |
| ----- | ---------- | ----- | ---- | ------------ |
| 1     | 256        | 200K  | 1e-4 | captions     |
| 2     | 512        | 80K   | 1e-4 | captions     |
| 3     | 512        | 40K   | 5e-5 | SFT captions |


`RUN` selects where the frozen text VAE comes from, which is the whole experiment:

```bash
# baseline: Cola's pretrained text VAE, untouched
RUN=cola PHASE=1 DATA=/path/to/parquet_roots \
    bash Prior_only/scripts/train_joint_prior_only.sh

# the text VAE the end-to-end stage reshaped
RUN=e2e TEXT_VAE_CKPT=extracted_vae/e2e-joint-l2/e2e-joint-l2-text-vae.pt \
    PHASE=1 DATA=/path/to/parquet_roots \
    bash Prior_only/scripts/train_joint_prior_only.sh
```

`IMAGE_VAE` (`flux` or `e2e`) is independent of `RUN`, so the text VAE can be isolated on its own by comparing `RUN=e2e IMAGE_VAE=flux` against`RUN=cola IMAGE_VAE=flux`.

## Sampling

```bash
CKPT=Prior_only/exps/<run>/checkpoints/0040000.pt bash scripts/sample_joint.sh
```

This runs a text VAE round-trip first, which fails loudly if the checkpoint's text VAE weights did not load, then `t2i` (captions in, image grid out) and `i2t` (images in, captions printed against the ground truth). 

`i2t` solves one ODE per text block, in order, because block `b` conditions on a clean history; `t2i` produces the whole image latent in one ODE.

## Evaluation: text-to-image

GenEval and DPG-Bench, scored by the same harnesses as the image-only T2I runs (see [T2I_EVAL.md](T2I_EVAL.md)). Generation is sharded over ranks by prompt and skips prompts already on disk, so an interrupted run can be relaunched as-is:

```bash
CKPT=Prior_only/exps/<run>/checkpoints/0200000.pt BENCH=geneval \
    GENEVAL_METADATA=<bench>/geneval/prompts/evaluation_metadata.jsonl \
    bash scripts/generate_joint_bench.sh
```

Then score with the T2I harnesses in score-only mode. `SKIP_GEN=1` is what keeps them from regenerating with the image-only loader, which cannot read a joint checkpoint; DPG also needs `SKIP_STITCH=1`, because `generate_joint_bench.py` writes the 2x2 grids itself:

```bash
cd eval_bench
SKIP_GEN=1 OUT=<gen out-dir>/geneval bash scripts/eval_geneval.sh
SKIP_GEN=1 SKIP_STITCH=1 GRID_DIR=<gen out-dir>/dpg RESOLUTION=512 \
    OUT=outputs/dpg_joint bash scripts/eval_dpg.sh
```

`RESOLUTION` is the size of one tile, not of the grid.

## Evaluation: image-to-text

The other direction, captioning COCO: CIDEr, BLEU-4, METEOR, ROUGE-L and SPICE against the human annotations, plus reference-free CLIPScore.

```bash
CKPT=Prior_only/exps/<run>/checkpoints/0080000.pt ARM=e2etext \
    COCO_ROOT=/data/coco KARPATHY_JSON=/data/coco_karpathy/dataset_coco.json \
    OUT=outputs/i2t bash scripts/eval_caption.sh
```

One sampling pass covers the union of COCO val2017 and the Karpathy test split (9407 unique images) and is then scored on both, because the two answer different questions:

- **val2017** is disjoint from `train2017` and therefore from the training mix. This is the clean number.
- **karpathy-test** is what the captioning literature reports, but 4407 of its 5000 images live in `train2017`, so the model has seen them during training (with synthetic captions, not the human ones). It exists for comparability, not as evidence.

Scoring needs `pycocoevalcap` and a JVM on `PATH` — PTBTokenizer, METEOR and SPICE all shell out to bundled jars:

```bash
pip install pycocoevalcap
```

Two things about that JVM, both of which fail late, after generation:

- **SPICE needs Java 8.** SPICE 1.0 drives the Nashorn JavaScript engine, which was removed in Java 15, so on a modern JVM it dies with `"scriptEngine" is null` while the other five metrics compute normally. Any Java 8 runtime works and it does not have to be the system one — prepend it to `PATH` for the scoring step. `SKIP_SPICE=1` drops the metric instead.
- **The jars need a writable directory.** They write scratch files next to themselves, which fails when the package sits in a root-owned `site-packages`; copy the package somewhere writable and point `PYCOCOEVALCAP_DIR` at it.

`SKIP_CLIPSCORE=1` drops the one metric that needs CLIP weights (`CLIP_PATH`, a local directory or a hub id).

Re-scoring an existing pass costs nothing extra:

```bash
SKIP_GEN=1 ARM=e2etext OUT=outputs/i2t COCO_ROOT=/data/coco \
    KARPATHY_JSON=... bash scripts/eval_caption.sh
```

Decoding matters when comparing arms and is held fixed by the script: EMA weights, cfg 1.5, 20 Euler steps per block, 4 blocks (64 tokens) with an end-of-text early stop.

One caveat about which checkpoint to caption with: the phase-3 SFT captions are long and descriptive, so a phase-3 checkpoint scores far worse on COCO's short human captions than the phase-2 one it came from, while being the better text-to-image model. The two directions are best read off their own phase.

## Released checkpoints (`unified-text-image/e2e-joint/`)

The best setting from Table 10 (**E2E-joint-trained text VAE and image VAE**: `86.57` GenEval, `72.68` DPG-Bench, `33.28` CIDEr) is published under `unified-text-image/e2e-joint/` together with its extracted joint VAE bundle `vae/e2e-joint-gan-l2-330k/`:

- `unified-text-image/e2e-joint/eit-joint-e2e-text-e2e-img-512-sft.pt`: Phase-3 512x512 SFT checkpoint (use for text-to-image GenEval / DPG-Bench).
- `unified-text-image/e2e-joint/eit-joint-e2e-text-e2e-img-512.pt`: Phase-2 512x512 checkpoint before SFT (use for image-to-text COCO captioning).
- `vae/e2e-joint-gan-l2-330k/e2e-joint-gan-l2-330k.pt`: E2E-reshaped image VAE (`--image-vae-ckpt`).
- `vae/e2e-joint-gan-l2-330k/e2e-joint-gan-l2-330k-latents-stats.pt`: Image latent statistics.
- `vae/e2e-joint-gan-l2-330k/e2e-joint-gan-l2-330k-text-vae.pt`: Standalone E2E-reshaped text VAE (note: the joint prior checkpoints already embed `ckpt["text_vae"]`, so only `--text-vae-path pretrained/cola-dlm/cola_dlm/cola_vae` for the architecture config and `--tokenizer-path pretrained/cola-dlm/tokenizer.json` are needed at evaluation time).

Example evaluation commands with the released checkpoints:

```bash
# Text-to-image (GenEval / DPG-Bench)
CKPT=/path/to/unified-text-image/e2e-joint/eit-joint-e2e-text-e2e-img-512-sft.pt \
IMAGE_VAE_CKPT=/path/to/vae/e2e-joint-gan-l2-330k/e2e-joint-gan-l2-330k.pt \
TEXT_VAE_PATH=pretrained/cola-dlm/cola_dlm/cola_vae \
TOKENIZER_PATH=pretrained/cola-dlm/tokenizer.json \
BENCH=geneval GENEVAL_METADATA=eval_bench/geneval/prompts/evaluation_metadata.jsonl \
  bash scripts/generate_joint_bench.sh

# Image-to-text (COCO captioning)
CKPT=/path/to/unified-text-image/e2e-joint/eit-joint-e2e-text-e2e-img-512.pt \
IMAGE_VAE_CKPT=/path/to/vae/e2e-joint-gan-l2-330k/e2e-joint-gan-l2-330k.pt \
TEXT_VAE_PATH=pretrained/cola-dlm/cola_dlm/cola_vae \
TOKENIZER_PATH=pretrained/cola-dlm/tokenizer.json \
ARM=e2e_joint COCO_ROOT=/data/coco KARPATHY_JSON=/data/coco_karpathy/dataset_coco.json \
  bash scripts/eval_caption.sh
```

