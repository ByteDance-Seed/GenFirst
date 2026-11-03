# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

"""Caption-NLL "understanding prior" on top of continuous VAE latents.

The VAE latent is patchified, projected to the LLM's hidden width, and injected
at the image-placeholder positions of a Qwen3-VL prompt. A frozen Qwen3-VL then
scores the caption; its token-level NLL is the loss. Gradients reach the
projector and (when the VAE is unfrozen) the VAE encoder, so the latent space is
shaped by "can a frozen VLM still read this image out of the latent".

Two reference numbers make the NLL interpretable:
  * ``vit`` : the same LLM reading the real Qwen3-VL ViT features -> the ceiling.
  * ``null``: the same prompt with no visual information at all -> the floor.
The gap ``null - latent`` is the pointwise mutual information (PMI) that the
latent actually carries about the caption; ``latent`` on its own is dominated by
the language entropy and is nearly useless as a progress signal.

Note that subtracting a *detached* null NLL does not change gradients at all, so
PMI is a diagnostic by default. ``pmi_mode="token_weight"`` is the variant that
does change the optimisation: it reweights the per-token CE by each token's PMI,
concentrating the gradient on the tokens that depend on the image.

Qwen3-VL requires ``transformers >= 4.57``. Only this module and
models/qwen_vit_ae.py need it, so it is an optional dependency: the SigLIP prior
(models/siglip_prior.py) and the rest of the repo run on any supported version.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_QWEN3VL = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_SYSTEM = "You are a helpful assistant."
DEFAULT_INSTRUCTION = "Describe this image in detail."


# ---------------------------------------------------------------------------
# projector: latent patch -> LLM hidden
# ---------------------------------------------------------------------------
class _SelfAttnBlock(nn.Module):
    def __init__(self, width, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        hidden = int(width * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class LatentProjector(nn.Module):
    """Map patchified latent tokens to LLM input embeddings.

    ``mode`` controls capacity, which is the central experimental knob:
      * ``linear`` / ``mlp``: deliberately weak, so a falling NLL means the
        *latent* improved rather than the projector compensating for it.
      * ``transformer``: deliberately strong, used with a frozen VAE to measure
        how much caption information the latent contains at all (the ceiling of
        the whole approach) and to provide the shortcut-diagnostic baseline.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        mode="linear",
        width=1024,
        depth=4,
        num_heads=8,
        num_tokens=1024,
        zero_init_out=False,
        out_rms=None,
    ):
        super().__init__()
        self.mode = mode
        self.out_dim = out_dim

        if mode == "linear":
            self.body = nn.Identity()
            self.head = nn.Linear(in_dim, out_dim)
        elif mode == "mlp":
            self.body = nn.Sequential(nn.Linear(in_dim, width), nn.GELU())
            self.head = nn.Linear(width, out_dim)
        elif mode == "transformer":
            self.stem = nn.Linear(in_dim, width)
            self.pos = nn.Parameter(torch.zeros(1, num_tokens, width))
            nn.init.trunc_normal_(self.pos, std=0.02)
            self.body = nn.ModuleList([_SelfAttnBlock(width, num_heads) for _ in range(depth)])
            self.final_norm = nn.LayerNorm(width)
            self.head = nn.Linear(width, out_dim)
        else:
            raise ValueError(f"unknown projector mode: {mode}")

        # Match the scale the frozen LLM expects at its input. Without this the
        # projected tokens sit at a wildly different RMS than real embeddings and
        # the first thousand steps are spent just fixing the norm.
        self.out_norm = nn.LayerNorm(out_dim, elementwise_affine=False)
        self.out_scale = nn.Parameter(torch.tensor(float(out_rms) if out_rms else 1.0))

        if zero_init_out:
            nn.init.zeros_(self.head.weight)
            if self.head.bias is not None:
                nn.init.zeros_(self.head.bias)

    def forward(self, x):
        """x: (B, N, in_dim) -> (B, N, out_dim)."""
        if self.mode == "transformer":
            h = self.stem(x)
            n = h.shape[1]
            if n > self.pos.shape[1]:
                raise ValueError(f"projector pos table too small: {n} > {self.pos.shape[1]}")
            h = h + self.pos[:, :n]
            for blk in self.body:
                h = blk(h)
            h = self.final_norm(h)
        else:
            h = self.body(x)
        h = self.head(h)
        return self.out_norm(h) * self.out_scale


# ---------------------------------------------------------------------------
# the prior itself
# ---------------------------------------------------------------------------
class UnderstandingPrior(nn.Module):
    def __init__(
        self,
        model_path=DEFAULT_QWEN3VL,
        latent_channels=16,
        latent_patch=2,
        projector_mode="linear",
        projector_width=1024,
        projector_depth=4,
        projector_heads=8,
        zero_init_out=False,
        load_vision=True,
        via_merger=False,
        tap="latent",
        use_deepstack=True,
        dtype=torch.bfloat16,
        device="cuda",
        max_caption_tokens=64,
        system_prompt=DEFAULT_SYSTEM,
        instruction=DEFAULT_INSTRUCTION,
        pmi_mode="monitor",
        null_mode="shuffle",
        grad_checkpointing=True,
    ):
        super().__init__()
        try:
            from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
        except ImportError as e:
            import transformers

            raise ImportError(
                "the caption-NLL understanding prior needs Qwen3-VL, which requires "
                f"transformers>=4.57 (found {transformers.__version__}). Install it "
                "with: pip install -U 'transformers>=4.57'"
            ) from e

        self.model_path = model_path
        self.latent_channels = latent_channels
        self.latent_patch = latent_patch
        self.max_caption_tokens = max_caption_tokens
        self.system_prompt = system_prompt
        self.instruction = instruction
        self.pmi_mode = pmi_mode
        self.null_mode = null_mode
        self.dtype = dtype
        if tap not in ("latent", "trunk"):
            raise ValueError(f"tap must be 'latent' or 'trunk', got {tap!r}")
        self.tap = tap
        self.use_deepstack = bool(use_deepstack)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        try:
            vlm = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, dtype=dtype, attn_implementation="sdpa",
            )
        except TypeError:  # older kwarg name
            vlm = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=dtype, attn_implementation="sdpa",
            )

        cfg = vlm.config
        self.image_token_id = int(cfg.image_token_id)
        self.merge_size = int(cfg.vision_config.spatial_merge_size)
        self.vit_patch_size = int(cfg.vision_config.patch_size)
        self.hidden_size = int(cfg.text_config.hidden_size)
        self.vit_dim = int(cfg.vision_config.hidden_size)
        self.via_merger = bool(via_merger)

        # `get_rope_index` lives on the Qwen3VLModel wrapper; keep it for correct
        # 3D (t,h,w) mrope ids on both the latent and the ViT path.
        self._vlm = vlm.model
        self.llm = vlm.model.language_model
        self.lm_head = vlm.lm_head
        # `via_merger` routes the latent back through the tower's own patch merger
        # instead of letting the projector map straight into LLM embedding space.
        # It is the whole point of pairing this loss with the Qwen3-VL encoder: the
        # merger is the tower's trained interface to the language model, so with it
        # fixed the only way to lower the NLL is to make the *latent* readable,
        # whereas a free projector can lower it by inventing an embedding space of
        # its own. It also matches how Qwen really sees an image -- 256 patches
        # merged into 64 LLM tokens at 256px, rather than 256 tokens.
        self.merger = vlm.model.visual.merger if via_merger else None
        self.visual = vlm.model.visual if load_vision else None
        if not load_vision:
            self._vlm.visual = None

        for p in self.llm.parameters():
            p.requires_grad_(False)
        for p in self.lm_head.parameters():
            p.requires_grad_(False)
        if self.visual is not None:
            for p in self.visual.parameters():
                p.requires_grad_(False)
        if self.merger is not None:
            for p in self.merger.parameters():
                p.requires_grad_(False)
            self.merger.eval()
        self.llm.eval()
        self.lm_head.eval()
        if self.visual is not None:
            self.visual.eval()

        if grad_checkpointing:
            self.llm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            # HF gates checkpointing on `self.training`, so the trunk has to be in
            # train mode for it to engage. Safe here: Qwen3-VL's text stack has
            # attention_dropout=0.0 and only RMSNorm, so train and eval modes are
            # numerically identical.
            self.llm.train()

        if tap == "trunk":
            # The trunk tap is fed the vision tower's own output, so there is
            # nothing to project and this module ends up with *zero* trainable
            # parameters. That is the point of it: with no projector, no pooling
            # head and no temperature anywhere on the path, the objective has no
            # free variable it can move instead of the encoder, so the only way to
            # lower the NLL is to make the trunk itself more informative.
            self.projector = None
        else:
            in_dim = latent_channels * latent_patch * latent_patch
            # Straight into embedding space, the projector has to match the
            # embedding table's RMS. Through the merger it does not: the merger
            # opens with a LayerNorm, which sets the scale for us.
            out_dim = self.vit_dim if via_merger else self.hidden_size
            emb_rms = None if via_merger else \
                self.llm.embed_tokens.weight.detach().float().pow(2).mean().sqrt().item()
            self.projector = LatentProjector(
                in_dim=in_dim,
                out_dim=out_dim,
                mode=projector_mode,
                width=projector_width,
                depth=projector_depth,
                num_heads=projector_heads,
                zero_init_out=zero_init_out,
                out_rms=emb_rms,
            )

        self.to(device)
        if self.projector is not None:
            self.projector.float()
        self._prompt_cache = {}

    # -- device / dtype -----------------------------------------------------
    @property
    def device(self):
        return self.llm.embed_tokens.weight.device

    def trainable_parameters(self):
        # Empty on the trunk tap, which reads the tower's own output and so has no
        # projector: the caller has to cope with there being no und optimizer.
        if self.projector is None:
            return []
        return [p for p in self.projector.parameters() if p.requires_grad]

    # -- sequence construction ---------------------------------------------
    def _ids(self, text):
        if text not in self._prompt_cache:
            self._prompt_cache[text] = self.tokenizer(text, add_special_tokens=False).input_ids
        return self._prompt_cache[text]

    def _build_batch(self, captions, num_visual_tokens):
        """Assemble padded input_ids / labels / masks for a caption batch.

        Layout (labels are active on the caption tokens only, so the reported NLL
        is per caption token and directly comparable across settings):
            <|im_start|>system\\n{sys}<|im_end|>
            <|im_start|>user\\n<|vision_start|>{IMG * N}<|vision_end|>{instr}<|im_end|>
            <|im_start|>assistant\\n{caption}<|im_end|>
        """
        pre = self._ids(
            f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|>"
        )
        mid = self._ids(f"<|vision_end|>{self.instruction}<|im_end|>\n<|im_start|>assistant\n")
        post = self._ids("<|im_end|>")

        img = [self.image_token_id] * num_visual_tokens
        seqs, labs = [], []
        for cap in captions:
            cap_ids = self.tokenizer(cap or "", add_special_tokens=False).input_ids
            cap_ids = cap_ids[: self.max_caption_tokens]
            if len(cap_ids) == 0:
                cap_ids = self._ids("image")
            ids = pre + img + mid + cap_ids + post
            lab = (
                [-100] * (len(pre) + len(img) + len(mid))
                + cap_ids
                + [-100] * len(post)
            )
            seqs.append(ids)
            labs.append(lab)

        max_len = max(len(s) for s in seqs)
        pad_id = self.tokenizer.pad_token_id or 0
        B = len(seqs)
        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
        labels = torch.full((B, max_len), -100, dtype=torch.long)
        attn = torch.zeros((B, max_len), dtype=torch.long)
        for i, (s, l) in enumerate(zip(seqs, labs)):
            input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            labels[i, : len(l)] = torch.tensor(l, dtype=torch.long)
            attn[i, : len(s)] = 1
        dev = self.device
        return input_ids.to(dev), labels.to(dev), attn.to(dev)

    def _rope_ids(self, input_ids, grid_thw, attention_mask):
        pos, _ = self._vlm.get_rope_index(
            input_ids=input_ids, image_grid_thw=grid_thw, attention_mask=attention_mask
        )
        return pos

    # -- latent -> visual embeddings ---------------------------------------
    def latent_to_embeds(self, z):
        """(B, C, H, W) latent -> (B, N, hidden) plus the LLM visual token grid.

        The returned grid counts *LLM* tokens, which is what ``forward`` needs to
        build the prompt and the mrope ids. With ``via_merger`` that is a factor of
        ``merge_size`` smaller in each dimension than the latent's token grid,
        because the merger folds each 2x2 block into one token.
        """
        B, C, H, W = z.shape
        p = self.latent_patch
        assert C == self.latent_channels, f"latent has {C} channels, expected {self.latent_channels}"
        assert H % p == 0 and W % p == 0, f"latent {H}x{W} not divisible by patch {p}"
        hl, wl = H // p, W // p
        x = z.reshape(B, C, hl, p, wl, p).permute(0, 2, 4, 1, 3, 5).reshape(B, hl * wl, C * p * p)
        h = self.projector(x.float())
        if self.merger is None:
            return h, (hl, wl)

        m = self.merge_size
        assert hl % m == 0 and wl % m == 0, f"token grid {hl}x{wl} not divisible by merge {m}"
        # The merger does `view(-1, vit_dim * m^2)` on *consecutive* tokens, so the
        # sequence has to be in merge-block-major order, not raster order.
        h = h.reshape(B, hl // m, m, wl // m, m, self.vit_dim)
        h = h.permute(0, 1, 3, 2, 4, 5).reshape(B * hl * wl, self.vit_dim)
        merged = self.merger(h.to(self.merger.linear_fc1.weight.dtype))
        return merged.reshape(B, -1, self.hidden_size), (hl // m, wl // m)

    # -- the frozen LLM ----------------------------------------------------
    def _llm_nll(self, inputs_embeds, position_ids, attention_mask, labels,
                 token_weights=None, visual_pos_masks=None, deepstack=None):
        # deepstack_visual_embeds are added to the hidden states at the visual
        # positions in the first len(deepstack) layers, which is how Qwen3-VL
        # actually consumes an image. Each entry is indexed as
        # hidden_states[visual_pos_masks], i.e. flattened row-major over the batch,
        # so the (B, n_tok, dim) tensors have to arrive folded the same way.
        extra = {}
        if deepstack:
            extra["visual_pos_masks"] = visual_pos_masks
            extra["deepstack_visual_embeds"] = [
                d.reshape(-1, d.shape[-1]).to(self.dtype) for d in deepstack
            ]
        out = self.llm(
            inputs_embeds=inputs_embeds.to(self.dtype),
            attention_mask=attention_mask,
            position_ids=position_ids,
            **extra,
        )
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

        # Next-token prediction: hidden[:, i] predicts labels[:, i+1].
        shift_hidden = hidden[:, :-1]
        shift_labels = labels[:, 1:]
        mask = shift_labels != -100
        if mask.sum() == 0:
            zero = inputs_embeds.sum() * 0.0
            return zero, zero.detach(), None

        # Only run the 152k-way head on scored positions; doing it on the full
        # sequence would dominate activation memory.
        sel_hidden = shift_hidden[mask]
        sel_labels = shift_labels[mask]
        logits = self.lm_head(sel_hidden).float()
        tok_nll = F.cross_entropy(logits, sel_labels, reduction="none")

        batch_idx = mask.nonzero(as_tuple=True)[0]
        B = labels.shape[0]
        counts = torch.zeros(B, device=tok_nll.device).index_add_(0, batch_idx, torch.ones_like(tok_nll))
        sums = torch.zeros(B, device=tok_nll.device).index_add_(0, batch_idx, tok_nll.detach())
        per_sample = sums / counts.clamp(min=1)

        if token_weights is not None:
            w = token_weights.detach()
            loss = (tok_nll * w).sum() / w.sum().clamp(min=1e-6)
        else:
            loss = tok_nll.mean()
        return loss, per_sample, (tok_nll, batch_idx, mask)

    def _scatter(self, text_embeds, img_mask, visual_embeds):
        embeds = text_embeds.clone()
        flat = visual_embeds.reshape(-1, visual_embeds.shape[-1]).to(text_embeds.dtype)
        embeds[img_mask] = flat
        return embeds

    # -- public API --------------------------------------------------------
    def forward(self, z, captions, compute_null=None):
        """Caption NLL of ``captions`` given latent ``z``.

        Returns a dict with ``loss`` (the differentiable NLL), ``nll``,
        and, when the null pass runs, ``nll_null`` / ``pmi``.
        """
        if self.tap != "latent":
            raise RuntimeError("this prior was built with tap='trunk'; use forward_vlm")
        embeds, grid = self.latent_to_embeds(z)
        return self._score(embeds, None, grid, captions, compute_null)

    def forward_vlm(self, vlm, captions, compute_null=None):
        """Caption NLL given the vision tower's own output (``tap='trunk'``).

        ``vlm`` is a ``VlmTokens`` from ``QwenViTAutoencoder.vit_forward``: the
        merged image embeddings and deepstack features, unmodified. Nothing on this
        path is trainable and nothing is reparameterised, so the forward is the
        stock Qwen3-VL one and the loss is a question purely about the trunk. With
        a frozen trunk it therefore reproduces the base model's caption NLL
        exactly, which doubles as the wiring check.
        """
        deepstack = list(vlm.deepstack) if self.use_deepstack else None
        return self._score(vlm.embeds, deepstack, vlm.grid, captions, compute_null)

    def _score(self, embeds, deepstack, grid, captions, compute_null):
        n_h, n_w = grid
        n_tok = n_h * n_w
        batch = embeds.shape[0]
        input_ids, labels, attn = self._build_batch(captions, n_tok)
        # get_rope_index derives the token count as (h//merge)*(w//merge), so the
        # grid it wants is the LLM token grid scaled up by merge_size. That holds
        # for every route: they just arrive with a grid already merge_size smaller.
        grid_thw = torch.tensor(
            [[1, n_h * self.merge_size, n_w * self.merge_size]] * batch,
            dtype=torch.long, device=self.device,
        )
        position_ids = self._rope_ids(input_ids, grid_thw, attn)

        with torch.no_grad():
            text_embeds = self.llm.embed_tokens(input_ids)
        img_mask = input_ids == self.image_token_id
        inputs_embeds = self._scatter(text_embeds, img_mask, embeds)

        need_null = compute_null if compute_null is not None else (self.pmi_mode != "off")
        null_per_sample = None
        token_weights = None
        if need_null:
            with torch.no_grad():
                null_embeds = self._null_embeds(text_embeds, img_mask, embeds)
                _, null_per_sample, null_parts = self._llm_nll(
                    null_embeds, position_ids, attn, labels,
                    visual_pos_masks=img_mask,
                    deepstack=self._null_deepstack(deepstack, batch),
                )
            if self.pmi_mode == "token_weight" and null_parts is not None:
                # Surprisal without the (matching) image is a cheap proxy for
                # "this token could depend on the image"; upweight those.
                null_tok = null_parts[0]
                token_weights = (null_tok - null_tok.mean()).clamp(min=0.0) + 1e-3

        loss, per_sample, _ = self._llm_nll(
            inputs_embeds, position_ids, attn, labels, token_weights=token_weights,
            visual_pos_masks=img_mask, deepstack=deepstack,
        )

        res = {
            "loss": loss,
            "nll": per_sample.mean(),
            "nll_per_sample": per_sample,
            "num_visual_tokens": n_tok,
        }
        if null_per_sample is not None:
            res["nll_null"] = null_per_sample.mean()
            res["pmi"] = (null_per_sample - per_sample).mean()
        return res

    def _null_deepstack(self, deepstack, batch):
        """``_null_embeds``' counterpart for the injected features.

        The control has to be shifted on *every* visual input at once. Rolling the
        merged tokens while leaving deepstack pointing at the true image would let
        the real image back in through the injected layers, and PMI would collapse
        towards zero for reasons that have nothing to do with the encoder.
        """
        if not deepstack or self.null_mode == "pad" or batch < 2:
            return None
        return [torch.roll(d, 1, dims=0) for d in deepstack]

    def _null_embeds(self, text_embeds, img_mask, visual_embeds):
        """The control condition that ``pmi`` is measured against.

        ``shuffle`` (default) rolls the batch so every caption is scored against
        a *different* image's visual tokens. The token statistics, count and
        positions are then identical to the real pass, so the gap isolates
        information about *this* image.

        ``pad`` instead leaves the placeholders at their own text embedding. That
        control is far more degenerate than any real visual input, so it inflates
        PMI (a randomly initialised projector already "wins" against it) and is
        kept only for reference.
        """
        if self.null_mode == "pad" or visual_embeds.shape[0] < 2:
            return text_embeds
        return self._scatter(text_embeds, img_mask, torch.roll(visual_embeds, 1, dims=0))

    @torch.no_grad()
    def vit_reference_nll(self, pixel_values, image_grid_thw, captions):
        """Ceiling: the same frozen LLM reading the real ViT features.

        DeepStack is intentionally *not* applied, matching the latent path, so
        the two numbers differ only in where the visual tokens came from.
        """
        if self.visual is None:
            raise RuntimeError("UnderstandingPrior was built with load_vision=False")
        pixel_values = pixel_values.to(self.device, self.dtype)
        image_grid_thw = image_grid_thw.to(self.device)
        vis_out = self.visual(pixel_values, grid_thw=image_grid_thw)
        vis = vis_out[0] if isinstance(vis_out, (tuple, list)) else vis_out

        merge_sq = self.merge_size ** 2
        per_image = (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]) // merge_sq
        n_tok = int(per_image[0].item())
        assert bool((per_image == n_tok).all()), "vit_reference_nll expects a uniform grid"

        input_ids, labels, attn = self._build_batch(captions, n_tok)
        position_ids = self._rope_ids(input_ids, image_grid_thw, attn)
        text_embeds = self.llm.embed_tokens(input_ids)
        img_mask = input_ids == self.image_token_id
        vis = vis.reshape(len(captions), n_tok, -1)

        _, per_sample, _ = self._llm_nll(
            self._scatter(text_embeds, img_mask, vis), position_ids, attn, labels
        )
        _, null_per_sample, _ = self._llm_nll(
            self._null_embeds(text_embeds, img_mask, vis), position_ids, attn, labels
        )
        return per_sample.mean(), null_per_sample.mean()
