"""
Eval rollout visualization: run env rollouts and save state/action curves and returns to viz/samples/step_XXXXX/.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _try_make_env(env_name: str):
    """Create env: LIBERO suite (libero_10, libero_spatial, ...) via make_libero_env, else Gymnasium/gym."""
    from src.envs.libero_env import LIBERO_SUITES, make_libero_env

    if env_name in LIBERO_SUITES:
        return make_libero_env(suite_name=env_name, task_id=0, state_dim=9, action_dim=7)
    try:
        import gymnasium as gym

        return gym.make(env_name)
    except Exception:
        try:
            import gym

            return gym.make(env_name)
        except Exception:
            return None


def _wrap_record_video(env: Any, video_folder: Path):
    """Wrap env with RecordVideo if available (Gymnasium or gym). Returns wrapped env or original."""
    try:
        import gymnasium as gym

        if hasattr(gym, "wrappers") and hasattr(gym.wrappers, "RecordVideo"):
            return gym.wrappers.RecordVideo(
                env,
                str(video_folder),
                episode_trigger=lambda ep: True,
            )
    except Exception:
        pass
    try:
        import gym

        if hasattr(gym, "wrappers") and hasattr(gym.wrappers, "RecordVideo"):
            return gym.wrappers.RecordVideo(env, str(video_folder), episode_trigger=lambda ep: True)
        if hasattr(gym, "wrappers") and hasattr(gym.wrappers, "Monitor"):
            return gym.wrappers.Monitor(
                env, str(video_folder), video_callable=lambda ep: True, force=True
            )
    except Exception:
        pass
    return env


def run_rollouts_and_save_viz(
    model: Any,
    env_name: str,
    state_mean: Optional[np.ndarray],
    state_std: Optional[np.ndarray],
    device: Any,
    run_dir: Path,
    step: int,
    num_rollouts: int = 3,
    max_episode_steps: int = 200,
    scale: float = 5000.0,
    save_video: bool = False,
) -> Dict[str, float]:
    """
    Run num_rollouts env episodes, save trajectory plots to run_dir/viz/samples/step_XXXXX/.
    If save_video is True, wrap env with RecordVideo and save one video per rollout to .../videos/.
    Returns metrics dict (eval/return_mean, eval/return_std, eval/len_mean).
    """
    env = _try_make_env(env_name)
    if env is None:
        try:
            from loguru import logger as _log
            from src.envs.libero_env import LIBERO_SUITES

            if env_name in LIBERO_SUITES:
                _log.warning(
                    "Eval rollouts skipped: LIBERO not installed for '{}'. "
                    "Install with: uv sync --extra libero (or pip install -e .[libero]). "
                    "Use offline eval (run_libero_eval.py) or set experiment.eval_every_steps=0 to disable.",
                    env_name,
                )
            else:
                _log.warning(
                    "Eval rollouts skipped: no env registered for '{}'. "
                    "Use offline eval or set experiment.eval_every_steps=0 to disable.",
                    env_name,
                )
        except Exception:
            print(f"Eval rollouts skipped: no env for '{env_name}'.")
        return {"eval/return_mean": 0.0}

    viz_dir = run_dir / "viz" / "samples" / f"step_{step:06d}"
    viz_dir.mkdir(parents=True, exist_ok=True)
    if save_video:
        video_folder = viz_dir / "videos"
        video_folder.mkdir(parents=True, exist_ok=True)
        print(f"Eval rollout videos will be saved to: {video_folder.resolve()}")
        env = _wrap_record_video(env, video_folder)

    state_mean_t = state_mean
    state_std_t = state_std
    if state_mean_t is None:
        state_mean_t = np.zeros(env.observation_space.shape[0])
    if state_std_t is None:
        state_std_t = np.ones(env.observation_space.shape[0])
    if hasattr(state_mean_t, "numpy"):
        state_mean_t = state_mean_t.numpy()
    if hasattr(state_std_t, "numpy"):
        state_std_t = state_std_t.numpy()
    state_mean_t = np.asarray(state_mean_t, dtype=np.float32)
    state_std_t = np.asarray(state_std_t, dtype=np.float32)

    import torch

    all_returns = []
    all_lengths = []
    all_states: List[np.ndarray] = []
    all_actions: List[np.ndarray] = []

    for ep in range(num_rollouts):
        obs, _ = env.reset(seed=step + ep)
        if isinstance(obs, tuple):
            obs = obs[0]
        context_dim = model.context_dim
        states = torch.from_numpy(obs).float().reshape(1, -1).to(device)
        contexts = torch.zeros(1, context_dim, device=device)
        actions_t = torch.zeros(0, model.act_dim, device=device)
        rewards_t = torch.zeros(0, device=device)
        returns_to_go = torch.tensor([[scale]], device=device)
        timesteps = torch.zeros(1, 1, dtype=torch.long, device=device)
        ep_states = [obs.copy()]
        ep_actions: List[np.ndarray] = []
        ep_return = 0.0
        for t in range(max_episode_steps):
            action = model.get_action(
                (states - torch.from_numpy(state_mean_t).to(device))
                / torch.from_numpy(state_std_t).to(device),
                contexts,
                actions_t,
                rewards_t,
                returns_to_go,
                timesteps,
                prompt=None,
                warm_train_steps=0,
                current_step=step,
            )
            action_np = action.detach().cpu().numpy().flatten()
            step_out = env.step(action_np)
            if len(step_out) == 5:
                next_obs, reward, done, truncated, _ = step_out
            else:
                next_obs, reward, done, _ = step_out[:4]
                truncated = False
            if isinstance(next_obs, tuple):
                next_obs = next_obs[0]
            ep_return += float(reward)
            ep_states.append(next_obs.copy())
            ep_actions.append(action_np.copy())
            actions_t = torch.cat([actions_t, action], dim=0)
            rewards_t = torch.cat([rewards_t, torch.tensor([reward], device=device)])
            states = torch.cat(
                [states, torch.from_numpy(next_obs).float().reshape(1, -1).to(device)], dim=0
            )
            returns_to_go = torch.cat(
                [returns_to_go, (returns_to_go[0, -1] - reward / scale).reshape(1, 1)], dim=1
            )
            timesteps = torch.cat([timesteps, torch.tensor([[t + 1]], device=device)], dim=1)
            if done or truncated:
                break
        all_returns.append(ep_return)
        all_lengths.append(len(ep_states) - 1)
        all_states.append(np.array(ep_states))
        all_actions.append(np.array(ep_actions))
    env.close()

    # Save visualizations
    viz_dir = run_dir / "viz" / "samples" / f"step_{step:06d}"
    viz_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for i in range(min(num_rollouts, len(all_states))):
            fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            S = all_states[i]
            A = all_actions[i]
            if S.ndim == 1:
                S = S.reshape(-1, 1)
            if A.ndim == 1:
                A = A.reshape(-1, 1)
            # Plot first few state dims
            for j in range(min(3, S.shape[1])):
                axes[0].plot(S[:, j], label=f"s{j}")
            axes[0].set_ylabel("state (sample dims)")
            axes[0].legend(loc="right", fontsize=6)
            axes[0].set_title(f"Rollout {i} (return={all_returns[i]:.1f})")
            for j in range(min(3, A.shape[1])):
                axes[1].plot(A[:, j], label=f"a{j}")
            axes[1].set_ylabel("action")
            axes[1].set_xlabel("step")
            axes[1].legend(loc="right", fontsize=6)
            fig.tight_layout()
            fig.savefig(viz_dir / f"rollout_{i}.png", dpi=100)
            plt.close(fig)
        # Summary of returns
        fig2, ax = plt.subplots(figsize=(4, 3))
        ax.bar(range(len(all_returns)), all_returns, color="steelblue")
        ax.axhline(
            np.mean(all_returns),
            color="red",
            linestyle="--",
            label=f"mean={np.mean(all_returns):.1f}",
        )
        ax.set_xlabel("rollout")
        ax.set_ylabel("return")
        ax.legend()
        ax.set_title(f"Eval step {step}")
        fig2.savefig(viz_dir / "returns.png", dpi=100)
        plt.close(fig2)
    except Exception:
        pass

    return {
        "eval/return_mean": float(np.mean(all_returns)),
        "eval/return_std": float(np.std(all_returns)),
        "eval/len_mean": float(np.mean(all_lengths)),
    }
