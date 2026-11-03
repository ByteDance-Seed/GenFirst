# Security notes

How to report a vulnerability is in [SECURITY.md](../SECURITY.md). This file is
the engineering counterpart: what the threat model here actually is, why the
dependency pins are what they are, and how the advisories that composition
scanners raise against this repository were assessed.

## What this code is

GenFirst is research training and evaluation code. It is not a service: it has
no network listener, no authentication surface and no multi-tenant boundary. It
runs on data and checkpoints that the operator supplies. That shapes which
dependency advisories actually matter here, and the notes below say so per
component rather than repeating a scanner's severity label.

## Loading checkpoints is equivalent to running code

Every training and sampling entry point loads checkpoints with `torch.load`,
which unpickles arbitrary Python. The checkpoints in this repository store an
`args` namespace alongside the tensors, so they cannot be read with
`weights_only=True`, and on torch 2.6 and newer, which flipped that default,
they need `weights_only=False` to load at all.

Treat a `.pt` file the way you would treat a script: only load checkpoints you
produced or obtained from a source you trust. The same applies to
`from_pretrained` on a Hugging Face repository id, which can execute code from
the remote repository.

## Dependency pins

`requirements.txt` carries security floors and is what you should install.
`environment.yml` is the archival lock: the exact versions the published
numbers were produced with, kept so results can be reproduced bit-for-bit. The
two files therefore disagree on purpose, and `environment.yml` installs
packages with known CVEs.

The following pins were raised above the versions used for the paper. None of
the upgrades touches an API this repository calls.

| Package    | Was     | Now     | Why |
|------------|---------|---------|-----|
| transformers | 4.46.3 | 5.9.0  | Clears every advisory the scanners report against the old pin. See below. |
| diffusers  | 0.31.0  | 0.38.0  | `trust_remote_code` bypasses in `from_pretrained` (CVE-2026-44513, CVE-2026-44827, CVE-2026-45804). This repository loads VAEs from Hub repository ids, so the path is reachable. |
| pillow     | 11.1.0  | 12.3.0  | Decoder and parser memory-safety bugs. Reachable in principle wherever user-supplied images are decoded. |
| requests   | 2.32.3  | 2.33.0  | CVE-2026-25645 and CVE-2024-47081. |
| pyarrow    | 17.0.0  | 23.0.1  | CVE-2026-25087. |
| click      | 8.1.8   | 8.3.3   | `click.edit` command injection. |

`huggingface-hub` (0.29.3 to 1.5.0), `tokenizers` (0.20.3 to 0.22.1), `regex`
(2024.11.6 to 2025.10.22) and `safetensors` (0.5.3 to 0.8.0) moved because
transformers 5.9.0 requires them, not as independent choices. The safetensors
bump also repairs the pin set: `diffusers` 0.38.0 requires `safetensors>=0.8.0`,
so the previous combination could not be installed at all.

## The transformers upgrade

Going from 4.46.3 to 5.9.0 crosses a major version, so it was verified rather
than assumed. `numpy` stays at 1.26.4: transformers 5.x declares only
`numpy>=1.17`, and nothing in the resolved set pulls numpy 2.x.

No source change was needed. The two `from_pretrained` calls that pass
`torch_dtype=` still work, because 5.x accepts it alongside the new `dtype`
spelling. The CLIP and T5 text encoders produce bit-identical outputs before and
after — context mean -0.108690, std 1.312527, pooled mean -0.113043,
`CLIPTextModelWithProjection` embeddings mean -0.000465, and `CLIPModel`
`logits_per_image` 19.108131 on a fixed input, on both versions. `diffusers`,
`timm`, `accelerate`, `torchmetrics` and `wandb` were also exercised against
`huggingface-hub` 1.x, including loading a local Flux VAE and building
`vit_base_patch14_dinov2`.

One visible difference: `PretrainedConfig.to_dict()` returns 47 keys instead of
91, because 5.x dropped generation defaults (`num_beams`, `temperature`,
`do_sample` and similar), TensorFlow legacy flags, and renamed `torch_dtype` to
`dtype`. None of the 45 removed keys is read by this repository; the vendored
Cola text VAE config was checked attribute by attribute and instantiates to the
same 195.29M parameters on both versions.

The upgrade also removes a caveat. Qwen3-VL, needed by the understanding stage,
required `transformers >= 4.57` and used to be an opt-in install on top of the
pin. It is now available on the pin itself.

### A note on CVE-2026-4372

This is the advisory that forced the upgrade, and the one whose scanner
metadata is most misleading. NVD marks every version below 5.3.0 as affected,
which is why it was reported against 4.46.3. The vulnerable code — the
kernel-dispatch path that turns `_attn_implementation_internal` in a
`config.json` into an `importlib.import_module()` of arbitrary Hub-hosted Python
— was introduced in 4.56.0 and removed in 5.3.0, and reaching it also requires
the optional `kernels` package. The old 4.46.3 pin therefore predated the
vulnerable code and never had this exposure, and 4.57.x, the obvious
intermediate target, sits inside the affected range. That is why the upgrade
went to 5.9.0 rather than to a 4.5x release.

Similarly, CVE-2025-14930 (GLM4 weight parsing) has 4.57.1 as its only affected
version, so it never applied to the old pin either.

## Outstanding

Nothing. Every advisory reported against the dependency set has either been
cleared by a pin or assessed as not applicable below.

### Advisories that were reported against the old transformers pin

All of these are fixed in 5.9.0, so this section is history rather than an open
risk. It is kept because it answers the follow-up question a reviewer asks next:
whether the old pin was actually exposed. Mostly it was not, because this
repository imports a narrow slice of the library. It uses `AutoConfig`,
`AutoModel`, `AutoTokenizer`, `AutoImageProcessor`, `PreTrainedModel`,
`PretrainedConfig`, the CLIP and T5 classes, and
`Qwen3VLForConditionalGeneration`. It does not use `Trainer`, the `transformers`
CLI, or `trust_remote_code`. Against that surface:

| Advisory | Affected component | Was it reachable on 4.46.3 |
|----------|--------------------|----------------------------|
| CVE-2026-1839 | `Trainer._load_rng_state()` calling `torch.load` | No. Training runs on a local accelerate loop; `Trainer` is never imported. |
| CVE-2025-6921 | `AdamWeightDecay._do_use_weight_decay()` | No. That is the TensorFlow optimizer; training uses `torch.optim.AdamW`. |
| CVE-2025-2099 | `testing_utils.preprocess_string()` | No. Test-only helper. |
| CVE-2025-3262 | `commands/chat.py` | No. CLI chat command. |
| CVE-2025-3264 | `dynamic_module_utils.get_imports()` | No. Only runs when loading remote code. |
| CVE-2025-3777 | `image_utils.py` URL validation | No. Images are opened locally with PIL, never fetched by URL through transformers. |
| CVE-2025-6638 | `MarianTokenizer.remove_language_code()` | No. Marian is not used. |
| CVE-2024-12720 | `tokenization_nougat_fast.py` | No. Nougat is not used. |
| CVE-2025-6051 | `EnglishNormalizer.normalize_numbers()` | No. Not used. |
| CVE-2025-14930 | GLM4 weight parsing | No. GLM4 is not used, and 4.57.1 was the only affected version. |
| CVE-2025-3263 | `configuration_utils.get_configuration_file()` | Marginal. Runs during `from_pretrained`, but needs adversarial filenames inside the model repository. |

The class of advisory that a narrow import surface does *not* protect against is
the one that turns `from_pretrained` on a Hub repository id into code execution
regardless of `trust_remote_code`, since loading pretrained CLIP, T5 and
Qwen3-VL weights is exactly what this repository does. CVE-2026-4372 is that
case, and it is why the pin moved. Independently of any version, load models
only from Hub repositories you trust.

## Reported but not applicable

| Reported | Assessment |
|----------|------------|
| pyarrow CVE-2026-25087 | The upstream advisory states the affected pre-buffering API is not exposed in the Python binding. Not reachable from Python at all; the pin was raised anyway. |
| requests CVE-2026-25645 | Affects callers of `requests.utils.extract_zipped_paths()`. This repository never calls it. |
| click `click.edit` | No code in this repository imports `click`. Its only direct user was `Prior_only/preprocessing/dataset_tools.py`, which has been removed; the pin remains because `wandb` depends on it. `click.edit` is never called. |
| InvokeAI | Not a dependency and the reported code is absent; see below. |
| xAR | Not a dependency. Earlier revisions of `Prior_only/loss.py` were a close copy of REPA's `SILoss`, which xAR also vendors, so a fingerprint scan attributed the file to xAR and then applied xAR's "nolicense" status. The file has been rewritten (same algorithm and trainer API; numerically checked against the previous implementation) and is MIT via `Prior_only/LICENSE`. |

### InvokeAI

InvokeAI is not a dependency of this project, direct or transitive. It appears
only as a source-fingerprint match on `models/autoencoder.py`, which derives
from CompVis latent-diffusion, as does InvokeAI's own copy. Shared ancestry,
not a shared supply chain.

Every InvokeAI advisory reported so far is a defect in its HTTP API server, and
this repository ships no server: there is no FastAPI, Flask, uvicorn or aiohttp
dependency, no route is defined anywhere, and none of the affected features
exist here.

| CVE | Where the defect lives | Why it cannot apply |
|-----|------------------------|---------------------|
| CVE-2025-6237 | Path traversal in `invokeai/app/services/bulk_download/bulk_download_default.py`, reached through `GET /api/v1/images/download/{bulk_download_item_name}` | No HTTP route, no bulk-download service |
| CVE-2024-10821 | Infinite loop parsing multipart boundaries, reached through `POST /api/v1/images/upload` | No HTTP route, no upload endpoint, no multipart parsing |
| CVE-2024-11043 | Resource exhaustion from an oversized `board_name`, reached through `PATCH /api/v1/boards/{board_id}` | No HTTP route, and no concept of boards, which is an InvokeAI gallery feature |

Each requires an unauthenticated attacker to reach a listening InvokeAI web
server over the network. Nothing in this repository listens on a socket.

The same scans also report license findings against imm, InvokeAI and tqdm.
Those are assessed in [NOTICE](../NOTICE), under "Notes on source-fingerprint
scan results"; one of the three is a real obligation and the other two are not.
