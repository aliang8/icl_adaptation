"""Meta Decision Transformer: DT-style sequence model with optional in-context prompt (GPT2 or Llama backbone)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.models.backbones import build_transformer_backbone
from src.models.types import DTBatch, DTOutput


class MetaDecisionTransformer(nn.Module):
    """(R,)S,A token layout + transformer; optional prompt prepended on the time axis in ``forward``."""

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
        sequence_token_layout: Optional[str] = None,
        use_trial_index_embedding: bool = True,
        max_trial_embeddings: int = 64,
        **kwargs,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.query_loss_only = query_loss_only
        self._predict_returns = predict_returns
        self._predict_state = predict_state
        raw_layout = (
            str(sequence_token_layout).strip().lower().replace("-", "_")
            if sequence_token_layout is not None and str(sequence_token_layout).strip()
            else ""
        )
        if raw_layout in ("none", "null"):
            raw_layout = ""
        if raw_layout:
            allowed = {"rtg_state_action", "state_action", "state_action_reward"}
            if raw_layout not in allowed:
                raise ValueError(
                    f"sequence_token_layout must be one of {sorted(allowed)}; got {raw_layout!r}"
                )
            self._sequence_token_layout = raw_layout
        else:
            self._sequence_token_layout = "rtg_state_action" if condition_rtg else "state_action"
        self._condition_rtg = self._sequence_token_layout == "rtg_state_action"
        self.use_trial_index_embedding = use_trial_index_embedding
        self.max_trial_embeddings = max(1, int(max_trial_embeddings))
        self.act_dim = act_dim
        self.context_dim = context_dim
        self.num_context_trajectories = int(num_context_trajectories)
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
        if not self._predict_returns:
            for p in self.predict_return.parameters():
                p.requires_grad = False
        if not self._predict_state:
            for p in self.predict_state.parameters():
                p.requires_grad = False
        self.use_language = False
        self.vision_encoder = None

    def _embed_trial_idx(self, trial_indices: Tensor) -> Tensor:
        if self.embed_trial_idx is None:
            raise RuntimeError("_embed_trial_idx called while use_trial_index_embedding is off")
        idx = trial_indices.long().clamp(0, self.embed_trial_idx.num_embeddings - 1)
        return self.embed_trial_idx(idx)

    def _merge_prompt_query_sequence(self, batch: DTBatch) -> Optional[Dict[str, Any]]:
        prompt = batch.prompt
        if prompt is None or len(prompt) < 6:
            return None
        (
            prompt_states,
            prompt_actions,
            prompt_rewards,
            prompt_returns_to_go,
            prompt_timesteps,
            prompt_attention_mask,
        ) = prompt[:6]
        prompt_trial = prompt[6] if len(prompt) >= 7 else None
        if prompt_states is None or prompt_states.dim() < 2 or int(prompt_states.shape[1]) == 0:
            return None

        B = int(batch.states.shape[0])
        layout = self._sequence_token_layout

        def _eb(x: Tensor) -> Tensor:
            if x.shape[0] == 1 and B > 1:
                return x.expand(B, *x.shape[1:])
            if int(x.shape[0]) != B:
                raise ValueError(f"prompt batch {x.shape[0]} vs query batch {B}")
            return x

        ps = _eb(prompt_states)
        pa = _eb(prompt_actions)
        prw = _eb(prompt_rewards)
        prtg = _eb(prompt_returns_to_go)
        pts = _eb(prompt_timesteps.long())
        pm = _eb(prompt_attention_mask.long())
        if prompt_trial is not None:
            ptrial = _eb(prompt_trial.long())
        else:
            ptrial = torch.zeros((ps.shape[0], ps.shape[1]), dtype=torch.long, device=ps.device)

        Tq = int(batch.states.shape[1])
        states = torch.cat([ps.float(), batch.states.float()], dim=1)
        actions = torch.cat([pa.float(), batch.actions.float()], dim=1)
        returns_to_go = torch.cat([prtg.float(), batch.returns_to_go.float()], dim=1)

        q_mask = batch.attention_mask
        if q_mask is None:
            q_mask = torch.ones((B, Tq), dtype=torch.long, device=batch.states.device)
        else:
            q_mask = q_mask.long()

        attention_mask = torch.cat([pm, q_mask], dim=1)
        rewards_cat: Optional[Tensor] = None
        if layout == "state_action_reward":
            mv = attention_mask.long().clamp(0, 1)
            timesteps = ((mv.cumsum(dim=1) - 1).clamp(min=0) * mv).long()
            if batch.rewards is None:
                raise ValueError(
                    "DTBatch.rewards is required when sequence_token_layout=state_action_reward"
                )
            rw_p = prw.float()
            if rw_p.dim() == 2:
                rw_p = rw_p.unsqueeze(-1)
            rw_q = batch.rewards.float()
            if rw_q.dim() == 2:
                rw_q = rw_q.unsqueeze(-1)
            rewards_cat = torch.cat([rw_p, rw_q], dim=1)
        else:
            timesteps = torch.cat([pts, batch.timesteps.long()], dim=1)

        ti_q = batch.trial_indices
        if ti_q is None:
            ti_q = torch.zeros((B, Tq), dtype=torch.long, device=batch.states.device)
        else:
            ti_q = ti_q.long()
        trial_indices = torch.cat([ptrial, ti_q], dim=1)

        merged: Dict[str, Any] = {
            "states": states,
            "actions": actions,
            "returns_to_go": returns_to_go,
            "timesteps": timesteps,
            "trial_indices": trial_indices,
            "attention_mask": attention_mask,
            "query_len": Tq,
        }

        if self._fuse_context_in_state:
            pt = int(ps.shape[1])
            zc = torch.zeros(
                (B, pt, self.context_dim),
                device=batch.contexts.device,
                dtype=batch.contexts.dtype,
            )
            merged["contexts"] = torch.cat([zc, batch.contexts.float()], dim=1)
        else:
            merged["contexts"] = None

        merged["rewards"] = rewards_cat

        if batch.image_embeddings is not None:
            ie = batch.image_embeddings
            pt = int(ps.shape[1])
            D = int(ie.shape[-1])
            zimg = torch.zeros((B, pt, D), device=ie.device, dtype=ie.dtype)
            merged["image_embeddings"] = torch.cat([zimg, ie], dim=1)
        else:
            merged["image_embeddings"] = None

        return merged

    def forward(self, batch: DTBatch) -> DTOutput:
        B, T_query = batch.states.shape[0], batch.states.shape[1]
        loss_mask = batch.attention_mask
        if loss_mask is None:
            loss_mask = torch.ones((B, T_query), dtype=torch.long, device=batch.states.device)

        merged = self._merge_prompt_query_sequence(batch)
        if merged is not None:
            states = merged["states"]
            actions = merged["actions"]
            returns_to_go = merged["returns_to_go"]
            timesteps = merged["timesteps"]
            ti = merged["trial_indices"]
            mask = merged["attention_mask"]
            contexts = merged["contexts"]
            rewards = merged["rewards"]
            image_embeddings = merged["image_embeddings"]
            T = T_query
            seq_length = int(states.shape[1])
        else:
            states = batch.states
            actions = batch.actions
            returns_to_go = batch.returns_to_go
            timesteps = batch.timesteps
            ti = batch.trial_indices    3
            if ti is None:
                ti = torch.zeros((B, T_query), dtype=torch.long, device=batch.states.device)
            contexts = batch.contexts if self._fuse_context_in_state else None
            rewards = batch.rewards
            image_embeddings = batch.image_embeddings
            T = T_query
            mask = loss_mask
            seq_length = T_query

        if self._sequence_token_layout == "state_action_reward" and merged is None:
            mv = loss_mask.long().clamp(0, 1)
            timesteps = ((mv.cumsum(dim=1) - 1).clamp(min=0) * mv).long()

        state_emb = self.encode_state(
            states,
            timesteps,
            ti,
            contexts=contexts if self._fuse_context_in_state else None,
            image_embeddings=image_embeddings,
            instruction_indices=batch.instruction_indices,
        )
        trial_e: Optional[Tensor] = None
        if self.embed_trial_idx is not None:
            trial_e = self._embed_trial_idx(ti)

        layout = self._sequence_token_layout

        action_emb = self.embed_action(actions) + self.embed_timestep(timesteps.long())
        if trial_e is not None:
            action_emb = action_emb + trial_e

        query_pair: Optional[Tuple[Tensor, Tensor]] = None
        query_triple: Optional[Tuple[Tensor, Tensor, Tensor]] = None

        if layout == "rtg_state_action":
            rtemb = self.embed_return(returns_to_go) + self.embed_timestep(timesteps.long())
            if trial_e is not None:
                rtemb = rtemb + trial_e
            query_triple = (rtemb, state_emb, action_emb)
        elif layout == "state_action_reward":
            if rewards is None:
                raise ValueError(
                    "DTBatch.rewards is required when sequence_token_layout=state_action_reward"
                )
            rw = rewards
            if rw.dim() == 2:
                rw = rw.unsqueeze(-1)
            rwemb = self.embed_return(rw.float()) + self.embed_timestep(timesteps.long())
            if trial_e is not None:
                rwemb = rwemb + trial_e
            query_triple = (state_emb, action_emb, rwemb)
        else:
            query_pair = (state_emb, action_emb)

        stacked, stacked_mask, tokens_per_step = self._stack_sequence(
            query_pair, query_triple, mask, seq_length
        )
        hidden = self._run_backbone(stacked, stacked_mask, tokens_per_step)
        if tokens_per_step == 3 and layout == "state_action_reward":
            action_hidden_idx = 0
        else:
            action_hidden_idx = 1
        pred_returns = None
        pred_states = None
        if tokens_per_step == 3:
            if layout == "state_action_reward":
                h_aux = hidden[:, 0]
            else:
                h_aux = hidden[:, 2]
            pred_returns = self.predict_return(h_aux)[:, -T:, :] if self._predict_returns else None
            pred_states = self.predict_state(h_aux)[:, -T:, :] if self._predict_state else None
        pred_actions_full = self.predict_action(hidden[:, action_hidden_idx])
        pred_actions = pred_actions_full[:, -T:, :]

        if self.query_loss_only:
            loss = self.compute_loss(pred_actions, batch.actions, loss_mask)
        else:
            if batch.prompt is not None:
                p = batch.prompt
                prompt_actions = p[1]
                prompt_attention_mask = p[5]
                pa = prompt_actions.to(pred_actions_full.device)
                if pa.shape[0] == 1 and B > 1:
                    pa = pa.expand(B, -1, -1)
                full_actions = torch.cat([pa, batch.actions], dim=1)
                if merged is not None:
                    full_mask = mask
                else:
                    pm = prompt_attention_mask.to(loss_mask.device)
                    if pm.shape[0] == 1 and B > 1:
                        pm = pm.expand(B, -1)
                    full_mask = torch.cat([pm, loss_mask], dim=1)
                loss = self.compute_loss(pred_actions_full, full_actions, full_mask)
            else:
                loss = self.compute_loss(pred_actions, batch.actions, loss_mask)

        return DTOutput(
            loss=loss,
            pred_actions=pred_actions,
            pred_states=pred_states,
            pred_returns=pred_returns,
            hidden_states=hidden,
        )

    def encode_state(
        self,
        states: Tensor,
        timesteps: Tensor,
        trial_indices: Optional[Tensor] = None,
        *,
        contexts: Optional[Tensor] = None,
        image_embeddings: Optional[Tensor] = None,
        instruction_indices: Optional[Tensor] = None,
    ) -> Tensor:
        _ = image_embeddings, instruction_indices
        ts = timesteps.long()
        st = states.float()
        if self._fuse_context_in_state:
            enc = self.state_encoder(st)  # type: ignore[union-attr]
            if contexts is None:
                ctx = torch.zeros(
                    enc.shape[0],
                    enc.shape[1],
                    self.context_dim,
                    device=enc.device,
                    dtype=enc.dtype,
                )
            else:
                ctx = contexts.float()
            state_emb = self.embed_state(torch.cat((enc, ctx), dim=-1))
        else:
            state_emb = self.embed_state(st)
        out = state_emb + self.embed_timestep(ts)
        if self.embed_trial_idx is not None:
            ti = trial_indices if trial_indices is not None else torch.zeros_like(ts)
            out = out + self._embed_trial_idx(ti.long())
        return out

    def _stack_sequence(
        self,
        query_pair: Optional[Tuple[Tensor, Tensor]],
        query_triple: Optional[Tuple[Tensor, Tensor, Tensor]],
        mask: Tensor,
        seq_length: int,
    ) -> Tuple[Tensor, Tensor, int]:
        if (query_pair is None) == (query_triple is None):
            raise ValueError("Provide exactly one of query_pair or query_triple")
        B = mask.shape[0]
        if query_triple is not None:
            t0, t1, t2 = query_triple
            tokens_per_step = 3
            stacked = (
                torch.stack((t0, t1, t2), dim=1)
                .permute(0, 2, 1, 3)
                .reshape(B, 3 * seq_length, self.hidden_size)
            )
            stacked_mask = (
                torch.stack((mask, mask, mask), dim=1).permute(0, 2, 1).reshape(B, 3 * seq_length)
            )
        else:
            s0, s1 = query_pair  # type: ignore[misc]
            tokens_per_step = 2
            stacked = (
                torch.stack((s0, s1), dim=1)
                .permute(0, 2, 1, 3)
                .reshape(B, 2 * seq_length, self.hidden_size)
            )
            stacked_mask = (
                torch.stack((mask, mask), dim=1).permute(0, 2, 1).reshape(B, 2 * seq_length)
            )
        stacked = self.embed_ln(stacked)
        return stacked, stacked_mask, tokens_per_step

    def _run_backbone(self, stacked: Tensor, mask: Tensor, tokens_per_step: int = 3) -> Tensor:
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
        """Masked MSE on actions (mask==0 ignored)."""
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
        """Rollout step: builds a batch and returns the last predicted action."""
        states = states.reshape(1, -1, self.state_dim)
        contexts = contexts.reshape(1, -1, self.context_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        L_s0 = states.shape[1]
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

        L_s_m = states.shape[1]
        rvec = rewards.reshape(-1).float().to(states.device)
        L_rew = int(rvec.numel())
        if L_rew == L_s_m:
            rewards_ta = rvec.reshape(1, L_s_m, 1)
        elif L_rew == L_s_m - 1:
            zr = torch.zeros(1, device=rvec.device, dtype=rvec.dtype)
            rewards_ta = torch.cat([rvec, zr], dim=0).reshape(1, L_s_m, 1)
        elif L_rew == 0 and L_s_m == 1:
            rewards_ta = torch.zeros(1, 1, 1, device=states.device, dtype=torch.float32)
        else:
            raise ValueError(
                f"get_action: need len(rewards)==len(states) or len(states)-1; "
                f"got states T={L_s_m}, rewards n={L_rew}"
            )

        trial_indices_kw = kwargs.get("trial_indices")
        prompt_nonempty = (
            prompt is not None
            and len(prompt) > 0
            and prompt[0] is not None
            and prompt[0].dim() >= 2
            and int(prompt[0].shape[1]) > 0
        )
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
            rewards_ta = rewards_ta[:, -query_cap:]
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

        L_s, L_c, L_a, L_r, L_rw, L_t = (
            states.shape[1],
            contexts.shape[1],
            actions.shape[1],
            returns_to_go.shape[1],
            rewards_ta.shape[1],
            timesteps.shape[1],
        )
        L_tr = trial_indices.shape[1]
        if query_cap is not None and not prompt_nonempty:
            pad_s = query_cap - L_s
            pad_c = query_cap - L_c
            pad_a = query_cap - L_a
            pad_r = query_cap - L_r
            pad_rw = query_cap - L_rw
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
            rewards_ta = torch.cat(
                [torch.zeros(1, pad_rw, 1, device=device), rewards_ta], dim=1
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

        ts_cap = max(0, int(self.embed_timestep.num_embeddings) - 1)
        timesteps = timesteps.clamp(0, ts_cap)

        image_embeddings = kwargs.get("image_embeddings")
        instruction_indices = kwargs.get("instruction_indices")
        if image_embeddings is not None:
            image_embeddings = image_embeddings.to(device)
            L_img = image_embeddings.shape[1]
            D_img = image_embeddings.shape[2]
            Tq = int(states.shape[1])
            if L_img >= Tq:
                image_embeddings = image_embeddings[:, -Tq:, :]
            else:
                pad_img = Tq - L_img
                image_embeddings = torch.cat(
                    [
                        torch.zeros(1, pad_img, D_img, device=device, dtype=image_embeddings.dtype),
                        image_embeddings,
                    ],
                    dim=1,
                )

        batch = DTBatch(
            states=states,
            contexts=contexts,
            actions=actions,
            returns_to_go=returns_to_go,
            rewards=rewards_ta if self._sequence_token_layout == "state_action_reward" else None,
            timesteps=timesteps,
            attention_mask=attention_mask,
            trial_indices=trial_indices,
            prompt=prompt,
            image_embeddings=image_embeddings,
            instruction_indices=instruction_indices,
        )
        out = self.forward(batch)
        return out.pred_actions[0, -1]
