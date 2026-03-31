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
    State path: when ``num_context_trajectories > 0`` and ``context_dim > 0``, concat encoded proprio
    with ``batch.contexts``; otherwise ``Linear(state_dim -> hidden)`` only (typical no-ICL / zeros context).
    """

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden_size: int,
        context_dim: int = 16,
        num_context_trajectories: int = 1,
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
        use_trial_index_embedding: bool = True,
        max_trial_embeddings: int = 64,
        **kwargs,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.query_loss_only = query_loss_only
        # Keep flags under different names so they aren't overwritten by the layer attributes below
        self._predict_returns = predict_returns
        self._predict_state = predict_state
        self._condition_rtg = condition_rtg
        self.use_trial_index_embedding = use_trial_index_embedding
        self.max_trial_embeddings = max(1, int(max_trial_embeddings))
        self.act_dim = act_dim
        self.context_dim = context_dim
        self.num_context_trajectories = int(num_context_trajectories)
        # Per-timestep batch.contexts are only fused when ICL is enabled (N>0); for N<=0 data uses zeros.
        self._fuse_context_in_state = context_dim > 0 and self.num_context_trajectories > 0
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
        if self._fuse_context_in_state:
            self.state_encoder = nn.Linear(state_dim, context_dim)
            self.embed_state = nn.Linear(context_dim * 2, hidden_size)
        else:
            self.state_encoder = None
            self.embed_state = nn.Linear(state_dim, hidden_size)
        self.prompt_embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.prompt_embed_return = nn.Linear(1, hidden_size)
        self.prompt_embed_state = nn.Linear(state_dim, hidden_size)
        self.prompt_embed_action = nn.Linear(act_dim, hidden_size)
        # Discrete trial id: embedding row **0** = padding; rows **1+** = context / query (1-based trial ids).
        self.embed_trial_idx = (
            nn.Embedding(self.max_trial_embeddings, hidden_size)
            if use_trial_index_embedding
            else None
        )
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

    def _embed_trial_idx(self, trial_indices: Tensor) -> Tensor:
        """Map integer trial indices to hidden_dim vectors (nn.Embedding), clamped to table size. **0** = padding."""
        if self.embed_trial_idx is None:
            raise RuntimeError("_embed_trial_idx called while use_trial_index_embedding is off")
        idx = trial_indices.long().clamp(0, self.embed_trial_idx.num_embeddings - 1)
        return self.embed_trial_idx(idx)

    def forward(self, batch: DTBatch) -> DTOutput:
        B, T = batch.states.shape[0], batch.states.shape[1]
        mask = batch.attention_mask
        if mask is None:
            mask = torch.ones((B, T), dtype=torch.long, device=batch.states.device)

        state_emb = self.encode_state(batch)
        B, T = batch.states.shape[0], batch.states.shape[1]
        trial_e: Optional[Tensor] = None
        if self.embed_trial_idx is not None:
            ti = batch.trial_indices
            if ti is None:
                ti = torch.zeros((B, T), dtype=torch.long, device=batch.states.device)
            trial_e = self._embed_trial_idx(ti)
        return_emb = None

        if self._condition_rtg:
            return_emb = self.embed_return(batch.returns_to_go) + self.embed_timestep(
                batch.timesteps.long()
            )
            if trial_e is not None:
                return_emb = return_emb + trial_e
        action_emb = self.embed_action(batch.actions) + self.embed_timestep(batch.timesteps.long())
        if trial_e is not None:
            action_emb = action_emb + trial_e

        stacked, stacked_mask, tokens_per_step = self._stack_with_prompt(
            return_emb, state_emb, action_emb, mask, batch.prompt, T
        )
        hidden = self._run_backbone(stacked, stacked_mask, tokens_per_step)
        # Match kzl Decision Transformer head indexing after
        # x.reshape(B, T, tokens_per_step, H).permute(0, 2, 1, 3):
        # https://github.com/kzl/decision-transformer/blob/master/gym/decision_transformer/models/decision_transformer.py
        # return/state preds: hidden at action token (index 2 for r,s,a); action pred: state token (index 1).
        action_hidden_idx = tokens_per_step - 2  # 1 for (r,s,a), 0 for (s,a) only
        pred_returns = None
        pred_states = None
        if tokens_per_step == 3:
            pred_returns = (
                self.predict_return(hidden[:, 2])[:, -T:, :] if self._predict_returns else None
            )
            pred_states = (
                self.predict_state(hidden[:, 2])[:, -T:, :] if self._predict_state else None
            )
        pred_actions_full = self.predict_action(hidden[:, action_hidden_idx])
        pred_actions = pred_actions_full[:, -T:, :]

        # Action MSE uses attention_mask / prompt_attention_mask: 0 = padded (no loss), 1 = valid.
        if self.query_loss_only:
            loss = self.compute_loss(pred_actions, batch.actions, mask)
        else:
            if batch.prompt is not None:
                p = batch.prompt
                prompt_actions = p[1]
                prompt_attention_mask = p[5]
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
        """Proprio -> hidden (fused with batch.contexts iff ``_fuse_context_in_state``); + timestep (+ trial)."""
        ts = batch.timesteps.long()
        if self._fuse_context_in_state:
            enc = self.state_encoder(batch.states)  # type: ignore[union-attr]
            state_emb = self.embed_state(torch.cat((enc, batch.contexts), dim=-1))
        else:
            state_emb = self.embed_state(batch.states)

        out = state_emb + self.embed_timestep(ts)
        if self.embed_trial_idx is not None:
            ti = batch.trial_indices
            if ti is None:
                ti = torch.zeros_like(ts)
            out = out + self._embed_trial_idx(ti)
        return out

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
            prompt_trial = None
            if len(prompt) >= 7:
                (
                    prompt_states,
                    prompt_actions,
                    _,
                    prompt_returns_to_go,
                    prompt_timesteps,
                    prompt_attention_mask,
                    prompt_trial,
                ) = prompt[:7]
            else:
                (
                    prompt_states,
                    prompt_actions,
                    _,
                    prompt_returns_to_go,
                    prompt_timesteps,
                    prompt_attention_mask,
                ) = prompt[:6]
            prompt_ts = prompt_timesteps.long()
            prompt_T = prompt_states.shape[1]
            if prompt_trial is None:
                prompt_trial = torch.zeros(
                    (prompt_states.shape[0], prompt_T),
                    dtype=torch.long,
                    device=prompt_states.device,
                )
            else:
                prompt_trial = prompt_trial.long()
            ps_emb = self.prompt_embed_state(prompt_states) + self.prompt_embed_timestep(prompt_ts)
            pa_emb = self.prompt_embed_action(prompt_actions) + self.prompt_embed_timestep(
                prompt_ts
            )
            if self.embed_trial_idx is not None:
                pt_e = self._embed_trial_idx(prompt_trial)
                ps_emb = ps_emb + pt_e
                pa_emb = pa_emb + pt_e
            if return_emb is not None:
                prtg = self.prompt_embed_return(prompt_returns_to_go)
                prtg_emb = prtg + self.prompt_embed_timestep(prompt_ts)
                if self.embed_trial_idx is not None:
                    prtg_emb = prtg_emb + self._embed_trial_idx(prompt_trial)
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
        """MSE on actions; only timesteps with mask > 0 (valid data) contribute to the mean.

        Padded query prefix and padded prompt slots use mask 0 in the dataset/collate, so they
        are excluded here. Padded values must not appear in the reduced loss.
        """
        if pred_actions is None or actions is None:
            return None
        if pred_actions.shape[:2] != actions.shape[:2] or pred_actions.shape[:2] != mask.shape[:2]:
            raise ValueError(
                "compute_loss: pred_actions, actions, and mask must align on [batch, time]; "
                f"got pred {tuple(pred_actions.shape)}, actions {tuple(actions.shape)}, "
                f"mask {tuple(mask.shape)}"
            )
        act_dim = pred_actions.shape[-1]
        valid = (mask > 0).reshape(-1)
        pred_flat = pred_actions.reshape(-1, act_dim)[valid]
        target_flat = actions.reshape(-1, act_dim)[valid]
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
        **kwargs,
    ) -> Tensor:
        """Single-step inference forward.

        **Action length:** Same as training ``_build_main_segment``, there is one action vector per
        state timestep. Rollouts may pass only executed actions (length ``T-1``); we append a **zero**
        row for the current decision step, then read ``pred_actions[..., -1, :]``.

        Query segment matches training (``query_window`` / ``max_length``): left-pad with mask 0 when
        shorter than K; once history reaches K, use the last K timesteps only.
        """
        states = states.reshape(1, -1, self.state_dim)
        contexts = contexts.reshape(1, -1, self.context_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        L_s0 = states.shape[1]
        # Match training windows: one action slot per state; rollout historically passed one fewer
        # action (past acts only). Append a dummy action for the current decision timestep.
        if contexts.shape[1] == 1 and L_s0 > 1:
            contexts = contexts.expand(1, L_s0, -1)
        L_a0 = actions.shape[1]
        if L_a0 == L_s0 - 1:
            z = torch.zeros(1, 1, self.act_dim, device=actions.device, dtype=actions.dtype)
            actions = torch.cat([actions, z], dim=1)
        elif L_a0 != L_s0:
            raise ValueError(
                f"get_action: need len(actions)==len(states) or len(states)-1; "
                f"got states T={L_s0}, actions T={L_a0}"
            )

        trial_indices_kw = kwargs.get("trial_indices")
        if prompt is not None and len(prompt) >= 7:
            pm, pt = prompt[5], prompt[6]
            valid = pm > 0
            if valid.any():
                query_trial_index = int(pt[valid].max().item()) + 1
            else:
                query_trial_index = 1
        elif "query_trial_index" in kwargs:
            query_trial_index = int(kwargs["query_trial_index"])
        else:
            query_trial_index = 1

        qw = kwargs.get("query_window")
        if qw is not None:
            query_cap = int(qw)
        elif self.max_length is not None:
            query_cap = int(self.max_length)
        else:
            query_cap = None

        if query_cap is not None:
            states = states[:, -query_cap:]
            contexts = contexts[:, -query_cap:]
            actions = actions[:, -query_cap:]
            returns_to_go = returns_to_go[:, -query_cap:]
            timesteps = timesteps[:, -query_cap:]
            if trial_indices_kw is not None:
                trial_indices_kw = trial_indices_kw[:, -query_cap:]

        device = states.device
        if trial_indices_kw is not None:
            trial_indices = trial_indices_kw.reshape(1, -1).long().to(device)
        else:
            trial_indices = torch.full(
                (1, timesteps.shape[1]),
                query_trial_index,
                device=device,
                dtype=torch.long,
            )

        L_s, L_c, L_a, L_r, L_t = (
            states.shape[1],
            contexts.shape[1],
            actions.shape[1],
            returns_to_go.shape[1],
            timesteps.shape[1],
        )
        L_tr = trial_indices.shape[1]
        if query_cap is not None:
            pad_s = query_cap - L_s
            pad_c = query_cap - L_c
            pad_a = query_cap - L_a
            pad_r = query_cap - L_r
            pad_t = query_cap - L_t
            pad_tr = query_cap - L_tr
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
            trial_indices = torch.cat(
                [torch.zeros(1, pad_tr, device=device, dtype=torch.long), trial_indices], dim=1
            )
            attention_mask = torch.cat(
                [
                    torch.zeros(pad_s, device=device, dtype=torch.long),
                    torch.ones(L_s, device=device, dtype=torch.long),
                ]
            ).reshape(1, -1)
        else:
            attention_mask = torch.ones(1, L_s, device=device, dtype=torch.long)

        image_embeddings = kwargs.get("image_embeddings")
        instruction_indices = kwargs.get("instruction_indices")
        if image_embeddings is not None:
            image_embeddings = image_embeddings.to(device)
            L_img = image_embeddings.shape[1]
            D_img = image_embeddings.shape[2]
            if query_cap is not None:
                if L_img >= query_cap:
                    image_embeddings = image_embeddings[:, -query_cap:, :]
                else:
                    pad_img = query_cap - L_img
                    image_embeddings = torch.cat(
                        [
                            torch.zeros(
                                1, pad_img, D_img, device=device, dtype=image_embeddings.dtype
                            ),
                            image_embeddings,
                        ],
                        dim=1,
                    )
            else:
                Tq = states.shape[1]
                if L_img >= Tq:
                    image_embeddings = image_embeddings[:, -Tq:, :]
                elif L_img < Tq:
                    pad_img = Tq - L_img
                    image_embeddings = torch.cat(
                        [
                            torch.zeros(
                                1, pad_img, D_img, device=device, dtype=image_embeddings.dtype
                            ),
                            image_embeddings,
                        ],
                        dim=1,
                    )

        batch = DTBatch(
            states=states,
            contexts=contexts,
            actions=actions,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
            trial_indices=trial_indices,
            prompt=prompt,
            image_embeddings=image_embeddings,
            instruction_indices=instruction_indices,
        )
        out = self.forward(batch)
        return out.pred_actions[0, -1]
