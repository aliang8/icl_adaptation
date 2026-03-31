"""
Typed batch and output containers for policy / DT models.
Semantic names, easy to extend, readable training code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from torch import Tensor


@dataclass
class DTBatch:
    """Input batch for Decision Transformer–style models."""

    states: Tensor  # [B, T, state_dim]
    contexts: Tensor  # [B, T, context_dim]
    actions: Tensor  # [B, T, act_dim]
    returns_to_go: Tensor  # [B, T, 1]
    timesteps: Tensor  # [B, T]
    attention_mask: Optional[Tensor] = None  # [B, T]
    # Trial id per timestep: **0** = padding; prompt uses **1..**; query = max(valid prompt trial) + 1.
    trial_indices: Optional[Tensor] = None  # [B, T], long
    # In-context prompt: (prompt_states, prompt_actions, prompt_rewards, prompt_rtg, prompt_ts, prompt_mask [, prompt_trial_idx])
    prompt: Optional[Tuple[Tensor, ...]] = None
    # VLA: optional modalities
    image_embeddings: Optional[Tensor] = None  # [B, T, D_vision]
    instruction_indices: Optional[Tensor] = None  # [B]


@dataclass
class DTOutput:
    """Output of a Decision Transformer–style forward pass."""

    pred_actions: Tensor  # [B, T, act_dim]
    pred_states: Optional[Tensor] = None  # [B, T, state_dim]
    pred_returns: Optional[Tensor] = None  # [B, T, 1]
    hidden_states: Optional[Tensor] = None
    loss: Optional[Tensor] = None  # set when batch.actions available and training
