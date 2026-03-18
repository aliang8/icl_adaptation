"""
VLA (Vision-Language-Action) Decision Transformer: language + multi-view camera images.

Extends MetaDecisionTransformer with optional:
- Modular vision encoder (patch / crossmae / dinov2 / dinov3 / paligemma) -> fused with state.
- Language instruction embedding (one per task/sample) -> fused with state.

Reference: https://github.com/Max-Fu/icrt (ICRA 2025)
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

from src.models.meta_dt import MetaDecisionTransformer
from src.models.types import DTBatch, DTOutput
from src.models.vision_encoders import build_vision_encoder


class LanguageEmbedder(nn.Module):
    """Embed language instructions (e.g. one per task) to hidden_size."""

    def __init__(
        self,
        vocab_size: int = 512,
        hidden_size: int = 128,
        max_instruction_length: int = 64,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.max_len = max_instruction_length

    def forward(self, instruction_ids: Tensor) -> Tensor:
        x = self.embed(instruction_ids)
        return x.mean(dim=1)


class VLADecisionTransformer(MetaDecisionTransformer):
    """
    Meta-DT with optional vision (multi-view images) and language (instruction) conditioning.
    Same forward(batch) -> DTOutput API; encode_state fuses image and language when present.
    """

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden_size: int,
        context_dim: int = 16,
        max_length: int = 20,
        max_ep_len: int = 200,
        action_tanh: bool = True,
        n_layer: int = 3,
        n_head: int = 1,
        n_inner: Optional[int] = None,
        activation_function: str = "relu",
        resid_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        n_positions: int = 1024,
        use_vision: bool = False,
        use_language: bool = False,
        num_instructions: int = 0,
        num_views: int = 2,
        image_embed_dim: int = 256,
        vision_fusion: str = "concat",
        transformer_backbone: str = "gpt2",
        llama_model_name: Optional[str] = None,
        vision_encoder_type: str = "patch",
        vision_encoder_pool: bool = True,
        vision_encoder_attention_pool: bool = False,
        freeze_vision_encoder: bool = False,
        vision_encoder_chunk_size: Optional[int] = None,
        use_precomputed_embeddings: bool = False,
        precomputed_vision_embed_dim: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(
            state_dim=state_dim,
            act_dim=act_dim,
            hidden_size=hidden_size,
            context_dim=context_dim,
            max_length=max_length,
            max_ep_len=max_ep_len,
            action_tanh=action_tanh,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_inner,
            activation_function=activation_function,
            resid_pdrop=resid_pdrop,
            attn_pdrop=attn_pdrop,
            n_positions=n_positions,
            transformer_backbone=transformer_backbone,
            llama_model_name=llama_model_name,
            **kwargs,
        )

        self.use_vision = use_vision
        self.use_language = use_language
        self.vision_fusion = vision_fusion

        if use_vision:
            if use_precomputed_embeddings and precomputed_vision_embed_dim is not None:
                # Training with precomputed embeddings: skip loading vision encoder (saves memory). vision_proj maps embeddings to hidden_size. For inference, load a checkpoint that includes the encoder or set use_precomputed_embeddings=False.
                self.vision_encoder = None
                self.vision_proj = nn.Linear(precomputed_vision_embed_dim, hidden_size)
            else:
                self.vision_encoder = build_vision_encoder(
                    encoder_type=vision_encoder_type,
                    num_views=num_views,
                    embed_dim=image_embed_dim,
                    img_size=(224, 224),
                    pool=vision_encoder_pool,
                    attention_pool=vision_encoder_attention_pool,
                    chunk_size=vision_encoder_chunk_size,
                )
                if freeze_vision_encoder:
                    for p in self.vision_encoder.parameters():
                        p.requires_grad = False
                out_dim = self.vision_encoder.output_dim
                self.vision_proj = nn.Linear(out_dim, hidden_size)
        else:
            self.vision_encoder = None
            self.vision_proj = None

        if use_language:
            self.language_proj = nn.Linear(hidden_size, hidden_size)
            n_instr = max(1, num_instructions)
            self.instruction_embed = nn.Embedding(n_instr, hidden_size)
        else:
            self.language_proj = None
            self.instruction_embed = None

    def encode_state(self, batch: DTBatch) -> Tensor:
        """State + context + timestep; then fuse image and language when present."""
        state_emb = super().encode_state(batch)

        if batch.image_embeddings is not None and self.vision_proj is not None:
            v = self.vision_proj(batch.image_embeddings)
            state_emb = state_emb + v

        if batch.instruction_indices is not None and self.instruction_embed is not None:
            lang = self.instruction_embed(batch.instruction_indices)
            if self.language_proj is not None:
                lang = self.language_proj(lang).unsqueeze(1)
            state_emb = state_emb + lang

        return state_emb
