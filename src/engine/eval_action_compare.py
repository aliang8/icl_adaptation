"""
Action-comparison evaluation: run teacher-forced (or autoregressive) prediction on held-out
trajectories and plot predicted vs GT actions per dimension (ICRT eval_plot-style).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.data.trajectories import discount_cumsum


def run_action_compare_eval(
    model: Any,
    trajectories: List[Dict[str, np.ndarray]],
    state_mean: np.ndarray,
    state_std: np.ndarray,
    device: Any,
    run_dir: Path,
    step: int,
    num_demos: int = 3,
    max_episode_steps: int = 200,
    scale: float = 5000.0,
    use_gt_action: bool = True,
    warm_train_steps: int = 0,
) -> Dict[str, float]:
    """
    For each of the first `num_demos` trajectories, run step-by-step action prediction
    (teacher-forced if use_gt_action else autoregressive), collect pred vs GT, then plot
    per action dimension and save to run_dir/viz/action_compare/step_XXXXX/demo_*/.

    Returns metrics: eval/action_mse_mean, eval/action_mse_std (over demos).
    """
    if not trajectories:
        return {"eval/action_mse_mean": 0.0, "eval/action_mse_std": 0.0}

    state_mean_t = state_mean
    state_std_t = state_std
    if hasattr(state_mean_t, "numpy"):
        state_mean_t = state_mean_t.numpy()
    if hasattr(state_std_t, "numpy"):
        state_std_t = state_std_t.numpy()
    state_mean_t = np.asarray(state_mean_t, dtype=np.float32)
    state_std_t = np.asarray(state_std_t, dtype=np.float32)
    if state_mean_t.ndim == 0:
        state_mean_t = np.broadcast_to(state_mean_t, (trajectories[0]["observations"].shape[1],))
    if state_std_t.ndim == 0:
        state_std_t = np.broadcast_to(state_std_t, (trajectories[0]["observations"].shape[1],))

    num_demos = min(num_demos, len(trajectories))
    context_dim = model.context_dim
    act_dim = model.act_dim
    state_dim = model.state_dim

    all_mse: List[float] = []
    all_pred: List[np.ndarray] = []
    all_gt: List[np.ndarray] = []

    for demo_idx in range(num_demos):
        traj = trajectories[demo_idx]
        obs = np.asarray(traj["observations"], dtype=np.float32)
        gt_actions = np.asarray(traj["actions"], dtype=np.float32)
        rewards = np.asarray(traj["rewards"], dtype=np.float32)
        T = min(obs.shape[0], gt_actions.shape[0], max_episode_steps)
        if T < 2:
            continue
        obs = obs[:T]
        gt_actions = gt_actions[:T]
        rewards = rewards[:T]

        rtg = discount_cumsum(rewards, gamma=1.0).reshape(-1, 1) / scale

        pred_list: List[np.ndarray] = []
        actions_so_far: List[torch.Tensor] = []

        for t in range(T):
            L = t + 1
            states_np = (obs[:L] - state_mean_t) / state_std_t
            rtg_seg = rtg[:L].reshape(1, L, 1)
            timesteps_np = np.arange(L, dtype=np.float32).reshape(1, L)

            states = torch.from_numpy(states_np).float().reshape(1, L, state_dim).to(device)
            contexts = torch.zeros(1, L, context_dim, device=device)
            if t == 0:
                actions_input = torch.ones(1, 1, act_dim, device=device) * -10.0
            else:
                if use_gt_action:
                    prev_actions = torch.from_numpy(gt_actions[:t].astype(np.float32)).to(device)
                else:
                    prev_actions = torch.from_numpy(np.stack(pred_list).astype(np.float32)).to(
                        device
                    )
                prev_actions = prev_actions.reshape(1, t, act_dim)
                pad_last = torch.ones(1, 1, act_dim, device=device) * -10.0
                actions_input = torch.cat([prev_actions, pad_last], dim=1)
            returns_to_go = torch.from_numpy(rtg_seg).float().to(device)
            timesteps = torch.from_numpy(timesteps_np).long().to(device)

            with torch.no_grad():
                pred_a = model.get_action(
                    states,
                    contexts,
                    actions_input,
                    returns_to_go=returns_to_go,
                    timesteps=timesteps,
                    prompt=None,
                    warm_train_steps=warm_train_steps,
                    current_step=step,
                )
            pred_np = pred_a.cpu().numpy().flatten()
            pred_list.append(pred_np)

        pred_arr = np.stack(pred_list)
        mse = float(np.mean((pred_arr - gt_actions) ** 2))
        all_mse.append(mse)
        all_pred.append(pred_arr)
        all_gt.append(gt_actions)

    if not all_mse:
        return {"eval/action_mse_mean": 0.0, "eval/action_mse_std": 0.0}

    out_dir = run_dir / "viz" / "action_compare" / f"step_{step:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for demo_idx in range(len(all_pred)):
        pred_arr = all_pred[demo_idx]
        gt_arr = all_gt[demo_idx]
        demo_dir = out_dir / f"demo_{demo_idx}"
        demo_dir.mkdir(parents=True, exist_ok=True)
        action_dim = pred_arr.shape[1]
        for j in range(action_dim):
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(gt_arr[:, j], label="GT", color="C0")
            ax.plot(pred_arr[:, j], label="pred", color="C1", alpha=0.8)
            ax.set_xlabel("step")
            ax.set_ylabel(f"action_{j}")
            ax.legend()
            ax.set_title(f"Demo {demo_idx} action dim {j}")
            fig.tight_layout()
            fig.savefig(demo_dir / f"action_dim_{j}.png", dpi=100)
            plt.close(fig)

    return {
        "eval/action_mse_mean": float(np.mean(all_mse)),
        "eval/action_mse_std": float(np.std(all_mse)) if len(all_mse) > 1 else 0.0,
    }
