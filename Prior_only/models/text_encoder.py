# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

"""Frozen text encoders for text-to-image conditioning.

Provides a small wrapper that turns a list of caption strings into:
  * ``context``: per-token sequence embeddings ``(B, L, context_dim)`` used as
    the text stream in the MMDiT joint attention;
  * ``pooled``: a single pooled vector ``(B, pooled_dim)`` added to the timestep
    embedding for adaLN modulation;
  * ``mask``: ``(B, L)`` attention mask (1 = keep).

The encoder is always frozen (eval + requires_grad=False). CLIP-L/14 is the
default (context/pooled dim = 768), which keeps the model light and matches the
context embedder dimension used in the reference MMDiT. T5 can be selected for
stronger text understanding at a higher memory cost.
"""

import torch
import torch.nn as nn


class FrozenTextEncoder(nn.Module):
    def __init__(self, encoder_type="clip", model_name=None, max_length=77, device="cpu"):
        super().__init__()
        self.encoder_type = encoder_type
        self.max_length = max_length

        if encoder_type == "clip":
            from transformers import CLIPTextModel, CLIPTokenizer

            model_name = model_name or "openai/clip-vit-large-patch14"
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
            self.model = CLIPTextModel.from_pretrained(model_name)
            self.context_dim = self.model.config.hidden_size
            self.pooled_dim = self.model.config.hidden_size
        elif encoder_type == "t5":
            from transformers import T5EncoderModel, T5TokenizerFast

            model_name = model_name or "google/t5-v1_1-xxl"
            self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
            self.model = T5EncoderModel.from_pretrained(model_name)
            self.context_dim = self.model.config.d_model
            # T5 has no pooled output; mean-pool the sequence for the adaLN vector.
            self.pooled_dim = self.model.config.d_model
        else:
            raise NotImplementedError(f"Unknown text encoder type: {encoder_type}")

        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(device)

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def encode(self, captions):
        """captions: list[str] -> (context, pooled, mask)."""
        tok = self.tokenizer(
            captions,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tok.input_ids.to(self.device)
        attention_mask = tok.attention_mask.to(self.device)

        if self.encoder_type == "clip":
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            context = out.last_hidden_state  # (B, L, D)
            pooled = out.pooler_output  # (B, D)
        else:  # t5
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            context = out.last_hidden_state  # (B, L, D)
            m = attention_mask.unsqueeze(-1).to(context.dtype)
            pooled = (context * m).sum(1) / m.sum(1).clamp(min=1.0)  # masked mean

        return context, pooled, attention_mask

    def forward(self, captions):
        return self.encode(captions)
