"""
Transformer backbones for the decision transformer.
- gpt2: custom GPT-2 (random init, config-driven). Replaces HF ``wte`` / ``wpe`` with parameter-free
  modules that return zeros (same ``forward`` signature as ``nn.Embedding``). Stock ``GPT2Model.forward``
  is unmodified; adding zero keeps ``inputs_embeds`` unchanged. Old checkpoints with ``wte``/``wpe``
  weights need ``strict=False`` or key stripping.
- llama2: HuggingFace LLaMA 2 pretrained; input/output projection to match hidden_size.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model


class _ZeroEmbedLike(nn.Module):
    """Drop-in for ``nn.Embedding``: long indices in, float tensor ``(*, dim)`` out, all zeros."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            *input_ids.shape,
            self.embedding_dim,
            device=input_ids.device,
            dtype=torch.float32,
        )


def build_gpt2_backbone(
    hidden_size: int,
    n_layer: int = 3,
    n_head: int = 1,
    n_inner: Optional[int] = None,
    n_positions: int = 1024,
    activation_function: str = "relu",
    resid_pdrop: float = 0.1,
    attn_pdrop: float = 0.1,
    **kwargs: Any,
) -> nn.Module:
    """Build custom GPT-2 backbone (no pretrained weights)."""
    n_inner = n_inner or 4 * hidden_size
    config = GPT2Config(
        vocab_size=1,
        n_embd=hidden_size,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=n_inner,
        activation_function=activation_function,
        resid_pdrop=resid_pdrop,
        attn_pdrop=attn_pdrop,
        n_positions=n_positions,
        **kwargs,
    )
    model = GPT2Model(config)
    d = int(config.n_embd)
    model.wte = _ZeroEmbedLike(d)
    model.wpe = _ZeroEmbedLike(d)
    return model


class Llama2BackboneWrapper(nn.Module):
    """
    Wraps HuggingFace LLaMA 2 so it matches the DT interface: inputs_embeds (B, seq, hidden_size)
    and attention_mask -> last_hidden_state (B, seq, hidden_size). Uses input/output projection
    so the rest of the model keeps a single hidden_size.
    """

    def __init__(
        self,
        hidden_size: int,
        llama_model_name: str = "meta-llama/Llama-2-7b-hf",
        n_positions: int = 1024,
        **kwargs: Any,
    ):
        super().__init__()
        from transformers import LlamaModel, LlamaConfig

        self.hidden_size = hidden_size
        # Load in float32 so inputs from DINOv2/rest of model (float32) match; avoids float vs half mismatch
        kwargs.setdefault("torch_dtype", torch.float32)
        self._llama = LlamaModel.from_pretrained(llama_model_name, **kwargs)
        llama_hidden = self._llama.config.hidden_size
        self.input_proj = nn.Linear(hidden_size, llama_hidden)
        self.output_proj = nn.Linear(llama_hidden, hidden_size)
        self.n_positions = n_positions

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # inputs_embeds: (B, seq, hidden_size)
        x = self.input_proj(inputs_embeds)
        # Match Llama dtype so input to q_proj etc. matches (avoids RuntimeError: float != c10::Half)
        llama_dtype = next(self._llama.parameters()).dtype
        x = x.to(llama_dtype)
        # LLaMA: attention_mask (B, seq) with 1 = attend, 0 = mask (same as our convention)
        out = self._llama(inputs_embeds=x, attention_mask=attention_mask)
        return self.output_proj(out.last_hidden_state.float())


def build_transformer_backbone(
    backbone_type: str = "gpt2",
    hidden_size: int = 128,
    n_layer: int = 3,
    n_head: int = 1,
    n_inner: Optional[int] = None,
    n_positions: int = 1024,
    activation_function: str = "relu",
    resid_pdrop: float = 0.1,
    attn_pdrop: float = 0.1,
    llama_model_name: Optional[str] = None,
    **kwargs: Any,
) -> nn.Module:
    """
    Factory: returns a backbone that implements forward(inputs_embeds, attention_mask) -> last_hidden_state
    with last_hidden_state shape (B, seq, hidden_size).
    """
    if backbone_type == "gpt2":
        return build_gpt2_backbone(
            hidden_size=hidden_size,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_inner,
            n_positions=n_positions,
            activation_function=activation_function,
            resid_pdrop=resid_pdrop,
            attn_pdrop=attn_pdrop,
            **kwargs,
        )
    if backbone_type == "llama2":
        name = llama_model_name or "meta-llama/Llama-2-7b-hf"
        return Llama2BackboneWrapper(
            hidden_size=hidden_size,
            llama_model_name=name,
            n_positions=n_positions,
            **kwargs,
        )
    raise ValueError(f"Unknown transformer_backbone: {backbone_type}. Use 'gpt2' or 'llama2'.")
