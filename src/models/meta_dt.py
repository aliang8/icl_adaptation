"""
Meta Decision Transformer: in-context learning for robot trajectories.
Context trajectories (same task) sorted by returns during training;
at inference, zero-shot adaptation with previous rollouts sorted ascending.
Supports transformer_backbone: gpt2 (custom) or llama2 (HuggingFace pretrained).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.models.backbones import build_transformer_backbone
from src.models.types import DTBatch, DTOutput


class MetaDecisionTransformer(nn.Module):
    """
    Models (Return_t, state_t, action_t, ...) with optional prompt (context) sequences.
    State is encoded with context then fed as state_dim*2 -> hidden.
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
        transformer_backbone: str = "gpt2",
        llama_model_name: Optional[str] = None,
        query_loss_only: bool = True,
        predict_returns: bool = False,
        predict_state: bool = False,
        condition_rtg: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.query_loss_only = query_loss_only
        # Keep flags under different names so they aren't overwritten by the layer attributes below
        self._predict_returns = predict_returns
        self._predict_state = predict_state
        self._condition_rtg = condition_rtg
        self.act_dim = act_dim
        self.context_dim = context_dim
        self.max_length = max_length
        self.hidden_size = hidden_size

        self.transformer = build_transformer_backbone(
            backbone_type=transformer_backbone,
            hidden_size=hidden_size,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_inner,
            n_positions=n_positions,
            activation_function=activation_function,
            resid_pdrop=resid_pdrop,
            attn_pdrop=attn_pdrop,
            llama_model_name=llama_model_name,
            **kwargs,
        )

        self.max_ep_len = max_ep_len
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_action = nn.Linear(act_dim, hidden_size)
        self.state_encoder = nn.Linear(state_dim, context_dim)
        self.embed_state = nn.Linear(context_dim * 2, hidden_size)
        self.prompt_embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.prompt_embed_return = nn.Linear(1, hidden_size)
        self.prompt_embed_state = nn.Linear(state_dim, hidden_size)
        self.prompt_embed_action = nn.Linear(act_dim, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)

        self.predict_state = nn.Linear(hidden_size, state_dim)
        self.predict_action = nn.Sequential(
            nn.Linear(hidden_size, act_dim),
            (nn.Tanh() if action_tanh else nn.Identity()),
        )
        self.predict_return = nn.Linear(hidden_size, 1)
        # Freeze return/state heads when disabled so they don't train and table shows "no"
        if not self._predict_returns:
            for p in self.predict_return.parameters():
                p.requires_grad = False
        if not self._predict_state:
            for p in self.predict_state.parameters():
                p.requires_grad = False
        self.use_language = False
        self.vision_encoder = None

    def forward(self, batch: DTBatch) -> DTOutput:
        B, T = batch.states.shape[0], batch.states.shape[1]
        mask = batch.attention_mask
        if mask is None:
            mask = torch.ones((B, T), dtype=torch.long, device=batch.states.device)

        state_emb = self.encode_state(batch)
        return_emb = None
        if self._condition_rtg:
            return_emb = self.embed_return(batch.returns_to_go) + self.embed_timestep(
                batch.timesteps.long()
            )
        action_emb = self.embed_action(batch.actions) + self.embed_timestep(batch.timesteps.long())

        stacked, stacked_mask, tokens_per_step = self._stack_with_prompt(
            return_emb, state_emb, action_emb, mask, batch.prompt, T
        )
        hidden = self._run_backbone(stacked, stacked_mask, tokens_per_step)
        action_index = tokens_per_step - 1  # 2 when (r,s,a), 1 when (s,a)
        pred_returns = None
        pred_states = None
        if tokens_per_step == 3:
            pred_returns = (
                self.predict_return(hidden[:, 2])[:, -T:, :] if self._predict_returns else None
            )
            pred_states = (
                self.predict_state(hidden[:, 2])[:, -T:, :] if self._predict_state else None
            )
        pred_actions_full = self.predict_action(hidden[:, action_index])
        pred_actions = pred_actions_full[:, -T:, :]

        if self.query_loss_only:
            loss = self.compute_loss(pred_actions, batch.actions, mask)
        else:
            if batch.prompt is not None:
                (
                    _,
                    prompt_actions,
                    _,
                    _,
                    _,
                    prompt_attention_mask,
                ) = batch.prompt
                full_actions = torch.cat(
                    [prompt_actions.to(pred_actions_full.device), batch.actions], dim=1
                )
                full_mask = torch.cat([prompt_attention_mask.to(mask.device), mask], dim=1)
                loss = self.compute_loss(pred_actions_full, full_actions, full_mask)
            else:
                loss = self.compute_loss(pred_actions, batch.actions, mask)
        return DTOutput(
            loss=loss,
            pred_actions=pred_actions,
            pred_states=pred_states,
            pred_returns=pred_returns,
            hidden_states=hidden,
        )

    def encode_state(self, batch: DTBatch) -> Tensor:
        """Fuse state + context and add timestep embedding. [B, T, D] -> [B, T, H]."""
        ts = batch.timesteps.long()
        enc = self.state_encoder(batch.states)
        state_emb = self.embed_state(torch.cat((enc, batch.contexts), dim=-1))
        return state_emb + self.embed_timestep(ts)

    def _stack_with_prompt(
        self,
        return_emb: Optional[Tensor],
        state_emb: Tensor,
        action_emb: Tensor,
        mask: Tensor,
        prompt: Optional[Tuple[Tensor, ...]],
        seq_length: int,
    ) -> Tuple[Tensor, Tensor, int]:
        B = state_emb.shape[0]
        if return_emb is not None:
            tokens_per_step = 3
            stacked = (
                torch.stack((return_emb, state_emb, action_emb), dim=1)
                .permute(0, 2, 1, 3)
                .reshape(B, 3 * seq_length, self.hidden_size)
            )
            stacked_mask = (
                torch.stack((mask, mask, mask), dim=1).permute(0, 2, 1).reshape(B, 3 * seq_length)
            )
        else:
            tokens_per_step = 2
            stacked = (
                torch.stack((state_emb, action_emb), dim=1)
                .permute(0, 2, 1, 3)
                .reshape(B, 2 * seq_length, self.hidden_size)
            )
            stacked_mask = (
                torch.stack((mask, mask), dim=1).permute(0, 2, 1).reshape(B, 2 * seq_length)
            )
        stacked = self.embed_ln(stacked)

        if prompt is not None:
            (
                prompt_states,
                prompt_actions,
                _,
                prompt_returns_to_go,
                prompt_timesteps,
                prompt_attention_mask,
            ) = prompt
            # If RTG has one extra element (e.g. data quirk), drop last step from all prompt tensors so lengths align
            if prompt_returns_to_go.shape[1] % 10 == 1 and prompt_returns_to_go.shape[1] > 1:
                prompt_states = prompt_states[:, :-1]
                prompt_actions = prompt_actions[:, :-1]
                prompt_returns_to_go = prompt_returns_to_go[:, :-1]
                prompt_timesteps = prompt_timesteps[:, :-1]
                prompt_attention_mask = prompt_attention_mask[:, :-1]
            prompt_ts = prompt_timesteps.long()
            prompt_T = prompt_states.shape[1]
            ps_emb = self.prompt_embed_state(prompt_states) + self.prompt_embed_timestep(prompt_ts)
            pa_emb = self.prompt_embed_action(prompt_actions) + self.prompt_embed_timestep(
                prompt_ts
            )
            if return_emb is not None:
                prtg = self.prompt_embed_return(prompt_returns_to_go)
                prtg_emb = prtg + self.prompt_embed_timestep(prompt_ts)
                prompt_stacked = (
                    torch.stack((prtg_emb, ps_emb, pa_emb), dim=1)
                    .permute(0, 2, 1, 3)
                    .reshape(prompt_states.shape[0], 3 * prompt_T, self.hidden_size)
                )
                prompt_stacked_mask = (
                    torch.stack(
                        (prompt_attention_mask, prompt_attention_mask, prompt_attention_mask),
                        dim=1,
                    )
                    .permute(0, 2, 1)
                    .reshape(prompt_states.shape[0], 3 * prompt_T)
                )
            else:
                prompt_stacked = (
                    torch.stack((ps_emb, pa_emb), dim=1)
                    .permute(0, 2, 1, 3)
                    .reshape(prompt_states.shape[0], 2 * prompt_T, self.hidden_size)
                )
                prompt_stacked_mask = (
                    torch.stack((prompt_attention_mask, prompt_attention_mask), dim=1)
                    .permute(0, 2, 1)
                    .reshape(prompt_states.shape[0], 2 * prompt_T)
                )
            query_len = tokens_per_step * seq_length
            if prompt_stacked.shape[1] == query_len:
                prompt_stacked = prompt_stacked.reshape(1, -1, self.hidden_size)
                prompt_stacked_mask = prompt_stacked_mask.reshape(1, -1)
                stacked = torch.cat((prompt_stacked.repeat(B, 1, 1), stacked), dim=1)
                stacked_mask = torch.cat((prompt_stacked_mask.repeat(B, 1), stacked_mask), dim=1)
            else:
                stacked = torch.cat((prompt_stacked, stacked), dim=1)
                stacked_mask = torch.cat((prompt_stacked_mask, stacked_mask), dim=1)

        return stacked, stacked_mask, tokens_per_step

    def _run_backbone(self, stacked: Tensor, mask: Tensor, tokens_per_step: int = 3) -> Tensor:
        """Transformer forward; return last_hidden_state reshaped to [B, tokens_per_step, seq, H]."""
        B, seq_len, _ = stacked.shape
        if mask.shape[0] != B or mask.shape[1] != seq_len:
            mask = (
                mask[:, :seq_len]
                if mask.shape[1] >= seq_len
                else torch.cat(
                    [
                        mask,
                        torch.zeros(
                            B, seq_len - mask.shape[1], device=mask.device, dtype=mask.dtype
                        ),
                    ],
                    dim=1,
                )
            )
        out = self.transformer(inputs_embeds=stacked, attention_mask=mask)
        x = out.last_hidden_state
        B = stacked.shape[0]
        seq_total = x.shape[1]
        x = x.reshape(B, seq_total // tokens_per_step, tokens_per_step, self.hidden_size).permute(
            0, 2, 1, 3
        )
        return x

    def compute_loss(self, pred_actions: Tensor, actions: Tensor, mask: Tensor) -> Optional[Tensor]:
        """MSE on actions at valid (masked) positions."""
        if pred_actions is None or actions is None:
            return None
        act_dim = pred_actions.shape[-1]
        pred_flat = pred_actions.reshape(-1, act_dim)[mask.reshape(-1) > 0]
        target_flat = actions.reshape(-1, act_dim)[mask.reshape(-1) > 0]
        if pred_flat.numel() == 0:
            return None
        return torch.nn.functional.mse_loss(pred_flat, target_flat)

    def get_action(
        self,
        states: Tensor,
        contexts: Tensor,
        actions: Tensor,
        rewards: Tensor,
        returns_to_go: Tensor,
        timesteps: Tensor,
        prompt: Optional[Tuple[Tensor, ...]],
        warm_train_steps: int,
        current_step: int,
        **kwargs,
    ) -> Tensor:
        """Single-step action for inference; pads to max_length and calls forward."""
        states = states.reshape(1, -1, self.state_dim)
        contexts = contexts.reshape(1, -1, self.context_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:, -self.max_length :]
            contexts = contexts[:, -self.max_length :]
            actions = actions[:, -self.max_length :]
            returns_to_go = returns_to_go[:, -self.max_length :]
            timesteps = timesteps[:, -self.max_length :]

        device = states.device
        L_s, L_c, L_a, L_r, L_t = (
            states.shape[1],
            contexts.shape[1],
            actions.shape[1],
            returns_to_go.shape[1],
            timesteps.shape[1],
        )
        pad_s = self.max_length - L_s
        pad_c = self.max_length - L_c
        pad_a = self.max_length - L_a
        pad_r = self.max_length - L_r
        pad_t = self.max_length - L_t
        states = torch.cat(
            [torch.zeros(1, pad_s, self.state_dim, device=device), states], dim=1
        ).float()
        contexts = torch.cat(
            [torch.zeros(1, pad_c, self.context_dim, device=device), contexts], dim=1
        ).float()
        actions = torch.cat(
            [torch.ones(1, pad_a, self.act_dim, device=device) * -10.0, actions], dim=1
        ).float()
        returns_to_go = torch.cat(
            [torch.zeros(1, pad_r, 1, device=device), returns_to_go], dim=1
        ).float()
        timesteps = torch.cat(
            [torch.zeros(1, pad_t, device=device, dtype=torch.long), timesteps], dim=1
        )
        attention_mask = torch.cat(
            [
                torch.zeros(pad_s, device=device, dtype=torch.long),
                torch.ones(L_s, device=device, dtype=torch.long),
            ]
        ).reshape(1, -1)

        image_embeddings = kwargs.get("image_embeddings")
        instruction_indices = kwargs.get("instruction_indices")
        if image_embeddings is not None:
            image_embeddings = image_embeddings.to(device)
            L_img = image_embeddings.shape[1]
            D_img = image_embeddings.shape[2]
            if L_img >= self.max_length:
                image_embeddings = image_embeddings[:, -self.max_length :, :]
            else:
                pad_img = self.max_length - L_img
                image_embeddings = torch.cat(
                    [
                        torch.zeros(1, pad_img, D_img, device=device, dtype=image_embeddings.dtype),
                        image_embeddings,
                    ],
                    dim=1,
                )

        use_prompt = prompt is not None and current_step > warm_train_steps
        batch = DTBatch(
            states=states,
            contexts=contexts,
            actions=actions,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
            prompt=prompt if use_prompt else None,
            image_embeddings=image_embeddings,
            instruction_indices=instruction_indices,
        )
        out = self.forward(batch)
        return out.pred_actions[0, -1]
