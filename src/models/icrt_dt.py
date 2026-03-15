"""
ICRT-style Decision Transformer: language + multi-view camera images.

Extends MetaDecisionTransformer with optional:
- Multi-view vision encoder (exterior + wrist images) -> image embeddings fused with state.
- Language instruction embedding (one per task/sample) -> added to state or prepended as token.

Reference: https://github.com/Max-Fu/icrt (ICRA 2025)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.models.meta_dt import MetaDecisionTransformer
from src.models.vision import MultiViewVisionEncoder


class LanguageEmbedder(nn.Module):
    """Embed language instructions (e.g. one per task) to hidden_size. Simple bag-of-words or lookup."""

    def __init__(
        self,
        vocab_size: int = 512,
        hidden_size: int = 128,
        max_instruction_length: int = 64,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.max_len = max_instruction_length

    def forward(self, instruction_ids: torch.Tensor) -> torch.Tensor:
        # instruction_ids: (B, L) long -> (B, hidden_size) pooled
        x = self.embed(instruction_ids)
        return x.mean(dim=1)


class ICRTDecisionTransformer(MetaDecisionTransformer):
    """
    Meta-DT with optional vision (multi-view images) and language (instruction) conditioning.

    - vision_encoder: MultiViewVisionEncoder that returns (B, T, num_tokens, D) or (B, T, D_pooled).
    - language_embedder: maps instruction indices or embeddings to hidden_size; optional.
    - When image_embeddings and/or language_embeddings are passed in forward, they are fused
      with state embeddings before the transformer.
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
        n_inner: int = None,
        activation_function: str = "relu",
        resid_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        n_positions: int = 1024,
        # ICRT-style
        use_vision: bool = False,
        use_language: bool = False,
        num_views: int = 2,
        image_embed_dim: int = 256,
        vision_fusion: str = "concat",  # "concat" (project to hidden) or "add" (after project)
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
            **kwargs,
        )

        self.use_vision = use_vision
        self.use_language = use_language
        self.vision_fusion = vision_fusion

        if use_vision:
            self.vision_encoder = MultiViewVisionEncoder(
                num_views=num_views,
                embed_dim=image_embed_dim,
                img_size=(224, 224),
            )
            if vision_fusion == "concat":
                self.vision_proj = nn.Linear(image_embed_dim * num_views, hidden_size)
            else:
                self.vision_proj = nn.Linear(image_embed_dim * num_views, hidden_size)
        else:
            self.vision_encoder = None
            self.vision_proj = None

        if use_language:
            self.language_proj = nn.Linear(hidden_size, hidden_size)
        else:
            self.language_proj = None

    def forward(
        self,
        states,
        contexts,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        attention_mask=None,
        prompt=None,
        image_embeddings: Optional[torch.Tensor] = None,
        language_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        image_embeddings: (B, T, D_vision) from vision encoder (pooled per step).
        language_embeddings: (B, D_lang) one vector per sample.
        """
        batch_size, seq_length = states.shape[0], states.shape[1]
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=states.device)

        timesteps = timesteps.long()

        state_encoding = self.state_encoder(states)
        state_embeddings = self.embed_state(torch.cat((state_encoding, contexts), dim=-1))

        if image_embeddings is not None and self.vision_proj is not None:
            # image_embeddings: (B, T, D_vision)
            v = self.vision_proj(image_embeddings)
            state_embeddings = state_embeddings + v

        if language_embeddings is not None and self.language_proj is not None:
            # (B, D_lang) -> (B, 1, D) broadcast to (B, T, D)
            lang = self.language_proj(language_embeddings).unsqueeze(1)
            state_embeddings = state_embeddings + lang

        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3 * seq_length)

        if prompt is not None:
            prompt_states, prompt_actions, prompt_rewards, prompt_returns_to_go, prompt_timesteps, prompt_attention_mask = prompt
            prompt_timesteps = prompt_timesteps.long()
            prompt_seq_length = prompt_states.shape[1]
            if prompt_returns_to_go.shape[1] % 10 == 1:
                prompt_returns_embeddings = self.prompt_embed_return(prompt_returns_to_go[:, :-1])
            else:
                prompt_returns_embeddings = self.prompt_embed_return(prompt_returns_to_go)
            prompt_state_embeddings = self.prompt_embed_state(prompt_states)
            prompt_action_embeddings = self.prompt_embed_action(prompt_actions)
            prompt_time_embeddings = self.prompt_embed_timestep(prompt_timesteps)
            prompt_state_embeddings = prompt_state_embeddings + prompt_time_embeddings
            prompt_action_embeddings = prompt_action_embeddings + prompt_time_embeddings
            prompt_returns_embeddings = prompt_returns_embeddings + prompt_time_embeddings
            prompt_stacked_inputs = torch.stack(
                (prompt_returns_embeddings, prompt_state_embeddings, prompt_action_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(prompt_states.shape[0], 3 * prompt_seq_length, self.hidden_size)
            prompt_stacked_attention_mask = torch.stack(
                (prompt_attention_mask, prompt_attention_mask, prompt_attention_mask), dim=1
            ).permute(0, 2, 1).reshape(prompt_states.shape[0], 3 * prompt_seq_length)
            if prompt_stacked_inputs.shape[1] == 3 * seq_length:
                prompt_stacked_inputs = prompt_stacked_inputs.reshape(1, -1, self.hidden_size)
                prompt_stacked_attention_mask = prompt_stacked_attention_mask.reshape(1, -1)
                stacked_inputs = torch.cat((prompt_stacked_inputs.repeat(batch_size, 1, 1), stacked_inputs), dim=1)
                stacked_attention_mask = torch.cat((prompt_stacked_attention_mask.repeat(batch_size, 1), stacked_attention_mask), dim=1)
            else:
                stacked_inputs = torch.cat((prompt_stacked_inputs, stacked_inputs), dim=1)
                stacked_attention_mask = torch.cat((prompt_stacked_attention_mask, stacked_attention_mask), dim=1)

        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs.last_hidden_state
        if prompt is None:
            x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
        else:
            x = x.reshape(batch_size, -1, 3, self.hidden_size).permute(0, 2, 1, 3)

        return_preds = self.predict_return(x[:, 2])[:, -seq_length:, :]
        state_preds = self.predict_state(x[:, 2])[:, -seq_length:, :]
        action_preds = self.predict_action(x[:, 1])[:, -seq_length:, :]
        return state_preds, action_preds, return_preds
