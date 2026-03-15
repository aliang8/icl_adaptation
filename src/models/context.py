"""Context encoder and decoders (world model components) for in-context adaptation."""
import torch
import torch.nn as nn


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class RNNContextEncoder(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, context_dim: int, context_hidden_dim: int):
        super().__init__()
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, context_dim), nn.ReLU())
        self.action_encoder = nn.Sequential(nn.Linear(action_dim, context_dim), nn.ReLU())
        self.reward_encoder = nn.Sequential(nn.Linear(1, context_dim), nn.ReLU())
        self.gru = nn.GRU(input_size=3 * context_dim, hidden_size=context_hidden_dim, num_layers=1)
        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)
        self.context_output = nn.Linear(context_hidden_dim, context_dim)
        self.apply(weights_init_)

    def forward(self, states, actions, rewards):
        """states, actions, rewards: [seq_len, batch, dim]."""
        states = states.reshape((-1, *states.shape[-2:]))
        actions = actions.reshape((-1, *actions.shape[-2:]))
        rewards = rewards.reshape((-1, *rewards.shape[-2:]))
        hs = self.state_encoder(states)
        ha = self.action_encoder(actions)
        hr = self.reward_encoder(rewards)
        h = torch.cat((ha, hs, hr), dim=-1)
        gru_output, _ = self.gru(h)
        contexts = self.context_output(gru_output[-1])
        return contexts


class RewardDecoder(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, context_dim: int, context_hidden_dim: int):
        super().__init__()
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, context_dim), nn.ReLU())
        self.action_encoder = nn.Sequential(nn.Linear(action_dim, context_dim), nn.ReLU())
        self.linear1 = nn.Linear(context_dim * 4, context_hidden_dim)
        self.linear2 = nn.Linear(context_hidden_dim, context_hidden_dim)
        self.linear3 = nn.Linear(context_hidden_dim, 1)
        self.apply(weights_init_)

    def forward(self, state, action, next_state, context):
        hs = self.state_encoder(state)
        ha = self.action_encoder(action)
        hs_next = self.state_encoder(next_state)
        h = torch.cat((hs, ha, hs_next, context), dim=-1)
        h = torch.relu(self.linear1(h))
        h = torch.relu(self.linear2(h))
        return self.linear3(h)


class StateDecoder(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, context_dim: int, context_hidden_dim: int):
        super().__init__()
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, context_dim), nn.ReLU())
        self.action_encoder = nn.Sequential(nn.Linear(action_dim, context_dim), nn.ReLU())
        self.reward_encoder = nn.Sequential(nn.Linear(1, context_dim), nn.ReLU())
        self.linear1 = nn.Linear(context_dim * 3, context_hidden_dim)
        self.linear2 = nn.Linear(context_hidden_dim, context_hidden_dim)
        self.linear3 = nn.Linear(context_hidden_dim, state_dim)
        self.apply(weights_init_)

    def forward(self, state, action, reward, next_state, context):
        hs = self.state_encoder(state)
        ha = self.action_encoder(action)
        hr = self.reward_encoder(reward)
        h = torch.cat((hs, ha, context), dim=-1)
        h = torch.relu(self.linear1(h))
        h = torch.relu(self.linear2(h))
        return self.linear3(h)
