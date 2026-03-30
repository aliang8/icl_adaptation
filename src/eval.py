"""
Evaluation: load checkpoint or inference artifact, run env episodes, report metrics.
Use for eval-only runs and for validating exported models.
"""

import os
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Add project root
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import MetaDecisionTransformer, RNNContextEncoder
from src.data.trajectories import sort_trajectories_by_return, discount_cumsum
from src.data.rtg import initial_rtg_token


def load_model_for_eval(
    checkpoint_path: str, device: torch.device, inference_artifact: bool = False
):
    """Load model from training checkpoint or inference artifact."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=inference_artifact)
    state_dim = ckpt.get("config", {}).get("model", {}).get("state_dim", 27)
    act_dim = ckpt.get("config", {}).get("model", {}).get("act_dim", 8)
    if inference_artifact:
        cfg = ckpt.get("config", {})
        m = cfg.get("model", cfg)
    else:
        m = ckpt.get("config", {}).get("model", {})
    model = MetaDecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_size=m.get("hidden_size", 128),
        context_dim=m.get("context_dim", 16),
        max_length=m.get("max_length", 20),
        max_ep_len=m.get("max_ep_len", 200),
        n_layer=m.get("n_layer", 3),
        n_head=m.get("n_head", 1),
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    state_mean = ckpt.get("state_mean")
    state_std = ckpt.get("state_std")
    return model, state_mean, state_std


def build_context_prompt_from_rollouts(
    rollouts: list,
    prompt_length: int,
    state_dim: int,
    action_dim: int,
    scale: float,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    device: torch.device,
    ascending: bool = True,
):
    """
    Build prompt from previous rollouts for zero-shot adaptation.
    Sorted ascending (worst to best) as per project spec.
    """
    sorted_rollouts = sort_trajectories_by_return(rollouts, ascending=ascending)
    if not sorted_rollouts:
        return None
    traj = sorted_rollouts[0]
    L = min(prompt_length, len(traj["rewards"]))
    start = max(0, len(traj["rewards"]) - L)
    s = (traj["observations"][start : start + L] - state_mean) / state_std
    a = traj["actions"][start : start + L]
    r = traj["rewards"][start : start + L].reshape(-1, 1)
    rtg = discount_cumsum(traj["rewards"][start:], gamma=1.0)[: L + 1].reshape(-1, 1)
    if rtg.shape[0] <= L:
        rtg = np.concatenate([rtg, np.zeros((1, 1))], axis=0)
    rtg = rtg / scale
    ts = np.arange(start, start + L)
    pad = prompt_length - L
    s = np.concatenate([np.zeros((pad, s.shape[1])), s], axis=0)
    a = np.concatenate([np.ones((pad, a.shape[1])) * -10.0, a], axis=0)
    r = np.concatenate([np.zeros((pad, 1)), r], axis=0)
    rtg = np.concatenate([np.zeros((pad, 1)), rtg], axis=0)
    ts = np.concatenate([np.zeros(pad), ts], axis=0)
    mask = np.concatenate([np.zeros(pad), np.ones(L)], axis=0)
    return (
        torch.from_numpy(s).float().unsqueeze(0).to(device),
        torch.from_numpy(a).float().unsqueeze(0).to(device),
        torch.from_numpy(r).float().unsqueeze(0).to(device),
        torch.from_numpy(rtg[:, :-1]).float().unsqueeze(0).to(device),
        torch.from_numpy(ts).long().unsqueeze(0).to(device),
        torch.from_numpy(mask).float().unsqueeze(0).to(device),
    )


def run_eval_episodes(
    model,
    env,
    context_encoder,
    state_mean,
    state_std,
    device,
    num_episodes: int = 5,
    max_episode_steps: int = 200,
    rtg_scale: float = 500.0,
    warm_train_steps: int = 0,
    current_step: int = 0,
    prompt=None,
    eval_target_return: Optional[float] = None,
):
    """Run evaluation episodes and return mean return and mean length.

    ``rtg_scale`` must match ``data.rtg_scale``. Per-step env rewards are raw;
    RTG tokens match ``eval_viz._run_one_rollout`` (``initial_rtg_token`` + ``r_env/rtg_scale``).
    """
    model.eval()
    if context_encoder is not None:
        context_encoder.eval()
    state_mean_t = torch.from_numpy(state_mean).to(device) if state_mean is not None else 0.0
    state_std_t = torch.from_numpy(state_std).to(device) if state_std is not None else 1.0

    rs = float(rtg_scale)
    rtg0 = initial_rtg_token(rs, eval_target_return=eval_target_return)
    returns = []
    lengths = []
    for _ in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        context_dim = model.context_dim
        states = torch.from_numpy(state).float().reshape(1, -1).to(device)
        contexts = torch.zeros(1, context_dim, device=device)
        actions = torch.zeros(0, model.act_dim, device=device)
        rewards = torch.zeros(0, device=device)
        returns_to_go = torch.tensor([[rtg0]], device=device, dtype=torch.float32)
        timesteps = torch.zeros(1, 1, dtype=torch.long, device=device)
        ep_return = 0.0
        for t in range(max_episode_steps):
            action = model.get_action(
                (states - state_mean_t) / state_std_t,
                contexts,
                actions,
                rewards,
                returns_to_go,
                timesteps,
                prompt=prompt,
                warm_train_steps=warm_train_steps,
                current_step=current_step,
            )
            action_np = action.detach().cpu().numpy()
            next_state, reward, done, _ = env.step(action_np)
            r_env = float(reward)
            ep_return += r_env
            actions = torch.cat([actions, action.unsqueeze(0)], dim=0)
            rewards = torch.cat([rewards, torch.tensor([r_env], device=device)])
            states = torch.cat(
                [states, torch.from_numpy(next_state).float().reshape(1, -1).to(device)], dim=0
            )
            if context_encoder is not None:
                # Update context from recent (s,a,r) segment
                pass  # Stub: compute context from last context_horizon steps
            returns_to_go = torch.cat(
                [returns_to_go, (returns_to_go[0, -1] - r_env / rs).reshape(1, 1)], dim=1
            )
            timesteps = torch.cat([timesteps, torch.tensor([[t + 1]], device=device)], dim=1)
            if done:
                break
        returns.append(ep_return)
        lengths.append(t + 1)
    return float(np.mean(returns)), float(np.mean(lengths))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--inference-artifact", action="store_true")
    parser.add_argument("--output-dir", type=str, default="outputs/eval")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, state_mean, state_std = load_model_for_eval(
        args.checkpoint, device, inference_artifact=args.inference_artifact
    )
    os.makedirs(args.output_dir, exist_ok=True)
    # Without a real env we only load and optionally save metrics placeholder
    print("Model loaded. Run with a real env (e.g. AntDir) to compute returns.")
    print("state_mean:", state_mean is not None)
    print("state_std:", state_std is not None)


if __name__ == "__main__":
    main()
