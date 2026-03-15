"""
Meta Decision Transformer: in-context learning for robot trajectories.
Context trajectories (same task) sorted by returns during training;
at inference, zero-shot adaptation with previous rollouts sorted ascending.
Based on Meta-DT (NJU-RL/Meta-DT); uses transformers.GPT2Model with inputs_embeds.
"""
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model


class MetaDecisionTransformer(nn.Module):
    """
    Models (Return_t, state_t, action_t, ...) with optional prompt (context) sequences.
    State is encoded with context (e.g. from RNN context encoder) then fed as state_dim*2 -> hidden.
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
        **kwargs,
    ):
        super().__init__()
        n_inner = n_inner or 4 * hidden_size
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.context_dim = context_dim
        self.max_length = max_length
        self.hidden_size = hidden_size

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
        self.transformer = GPT2Model(config)

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
    ):
        batch_size, seq_length = states.shape[0], states.shape[1]
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=states.device)

        state_encoding = self.state_encoder(states)
        state_embeddings = self.embed_state(torch.cat((state_encoding, contexts), dim=-1))
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

    def get_action(
        self,
        states,
        contexts,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        prompt,
        warm_train_steps: int,
        current_step: int,
        **kwargs,
    ):
        states = states.reshape(1, -1, self.state_dim)
        contexts = contexts.reshape(1, -1, self.context_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:, -self.max_length:]
            contexts = contexts[:, -self.max_length:]
            actions = actions[:, -self.max_length:]
            returns_to_go = returns_to_go[:, -self.max_length:]
            timesteps = timesteps[:, -self.max_length:]

        attention_mask = torch.cat([
            torch.zeros(self.max_length - states.shape[1], device=states.device, dtype=torch.long),
            torch.ones(states.shape[1], device=states.device, dtype=torch.long),
        ]).reshape(1, -1)
        states = torch.cat([
            torch.zeros((1, self.max_length - states.shape[1], self.state_dim), device=states.device),
            states,
        ], dim=1).float()
        contexts = torch.cat([
            torch.zeros((1, self.max_length - contexts.shape[1], self.context_dim), device=contexts.device),
            contexts,
        ], dim=1).float()
        actions = torch.cat([
            torch.ones((1, self.max_length - actions.shape[1], self.act_dim), device=actions.device) * -10.0,
            actions,
        ], dim=1).float()
        returns_to_go = torch.cat([
            torch.zeros((1, self.max_length - returns_to_go.shape[1], 1), device=returns_to_go.device),
            returns_to_go,
        ], dim=1).float()
        timesteps = torch.cat([
            torch.zeros((1, self.max_length - timesteps.shape[1]), device=timesteps.device, dtype=torch.long),
            timesteps,
        ], dim=1)

        use_prompt = prompt is not None and current_step > warm_train_steps
        _, action_preds, _ = self.forward(
            states, contexts, actions, None, returns_to_go, timesteps,
            attention_mask=attention_mask, prompt=prompt if use_prompt else None,
        )
        return action_preds[0, -1]
