"""
Eval rollout visualization: run env rollouts and save state/action curves and returns to viz/samples/step_XXXXX/.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _try_make_env(env_name: str):
    """Create env: LIBERO suite (libero_10, ...) via make_libero_env, else Gymnasium/gym. Returns None if env or deps missing."""
    from src.envs.libero_env import LIBERO_SUITES, make_libero_env

    if env_name in LIBERO_SUITES:
        return make_libero_env(suite_name=env_name, task_id=0, state_dim=9, action_dim=7)
    
    import gymnasium as gym
    return gym.make(env_name)


class _GymnasiumToGymStepAdapter:
    """Wraps an env so step() returns 4 values (obs, reward, done, info). Use before gym's RecordVideo."""

    def __init__(self, env: Any):
        self._env = env

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    def step(self, action: Any):
        out = self._env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            return (obs, reward, bool(terminated or truncated), info)
        return out

    def reset(self, *args, **kwargs):
        return self._env.reset(*args, **kwargs)


def _wrap_record_video(env: Any, video_folder: Path) -> Tuple[Any, bool]:
    """Wrap env with RecordVideo (Gymnasium). Returns (wrapped_env, success)."""
    import gymnasium as gym
    if hasattr(gym, "wrappers") and hasattr(gym.wrappers, "RecordVideo"):
        return (
            gym.wrappers.RecordVideo(
                env, str(video_folder), episode_trigger=lambda ep: True
            ),
            True,
        )
    return env, False


def _write_trial_video(video_folder: Path, trial: int, frames: List[np.ndarray], fps: int = 20) -> None:
    """Write frames to video_folder/trial_{trial}.mp4."""
    if not frames:
        return
    import imageio
    path = video_folder / f"trial_{trial}.mp4"
    writer = imageio.get_writer(str(path), fps=fps)
    for f in frames:
        writer.append_data(np.asarray(f))
    writer.close()


def _add_trial_text(frame: np.ndarray, label: str) -> np.ndarray:
    """Draw label on frame (copy). Requires cv2."""
    import cv2
    out = np.asarray(frame).copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, min(w, h) / 400.0)
    thick = max(1, int(scale * 2))
    cv2.putText(out, label, (10, 30), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    cv2.putText(out, label, (10, 30), font, scale, (0, 0, 0), max(1, thick - 1), cv2.LINE_AA)
    return out


def _run_one_rollout(
    model: Any,
    env: Any,
    state_mean_t: np.ndarray,
    state_std_t: np.ndarray,
    device: Any,
    scale: float,
    max_episode_steps: int,
    step: int,
    ep: int,
    prompt: Optional[Tuple[Any, ...]],
    collect_frames: bool = False,
) -> Tuple[float, int, np.ndarray, np.ndarray, Dict[str, np.ndarray], List[np.ndarray]]:
    """Run one episode; return (return, length, states, actions, trajectory_dict, frames)."""
    import torch
    obs, _ = env.reset(seed=step + ep)
    if isinstance(obs, tuple):
        obs = obs[0]
    frames: List[np.ndarray] = []
    if collect_frames and hasattr(env, "render"):
        frame = env.render(mode="rgb_array")
        if frame is not None and isinstance(frame, np.ndarray):
            frames.append(np.asarray(frame))
    context_dim = model.context_dim
    states = torch.from_numpy(obs).float().reshape(1, -1).to(device)
    contexts = torch.zeros(1, context_dim, device=device)
    actions_t = torch.zeros(0, model.act_dim, device=device)
    rewards_t = torch.zeros(0, device=device)
    returns_to_go = torch.tensor([[scale]], device=device)
    timesteps = torch.zeros(1, 1, dtype=torch.long, device=device)
    ep_states = [obs.copy()]
    ep_actions: List[np.ndarray] = []
    ep_rewards: List[float] = []
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
            prompt=prompt,
            warm_train_steps=0,
            current_step=step,
        )
        if action.dim() == 1:
            action = action.unsqueeze(0)
        action_np = action.detach().cpu().numpy().flatten()
        step_out = env.step(action_np)
        if len(step_out) == 5:
            next_obs, reward, done, truncated, _ = step_out
        else:
            next_obs, reward, done, _ = step_out[:4]
            truncated = False
        if collect_frames and hasattr(env, "render"):
            frame = env.render(mode="rgb_array")
            if frame is not None and isinstance(frame, np.ndarray):
                frames.append(np.asarray(frame))
        if isinstance(next_obs, tuple):
            next_obs = next_obs[0]
        r = float(reward)
        ep_return += r
        ep_rewards.append(r)
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
    traj_dict = {
        "observations": np.array(ep_states, dtype=np.float32),
        "actions": np.array(ep_actions, dtype=np.float32),
        "rewards": np.array(ep_rewards, dtype=np.float32),
    }
    return (
        ep_return,
        len(ep_states) - 1,
        np.array(ep_states),
        np.array(ep_actions),
        traj_dict,
        frames,
    )


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
    eval_context_mode: str = "prompt",
    prompt_trajectories: Optional[List[Dict[str, np.ndarray]]] = None,
    eval_num_trials: int = 5,
    eval_context_k: Optional[int] = None,
    eval_reward_source: str = "env",
    eval_reward_model: Optional[str] = None,
    total_prompt_len: Optional[int] = None,
    max_prompt_trajectory_length: Optional[int] = None,
    task_description: Optional[str] = None,
    logger: Optional[Any] = None,
) -> Dict[str, float]:
    """
    Run one eval rollout (N trials). Each trial can be saved as trial_0.mp4, trial_1.mp4, ...
    For wandb: log one continuous video with "Trial 1", "Trial 2", ... overlaid.
    """
    env = _try_make_env(env_name)
    if env is None:
        from src.envs.libero_env import LIBERO_SUITES
        if env_name in LIBERO_SUITES:
            print(
                f"Eval rollouts skipped: LIBERO not installed for '{env_name}'. "
                "Install with: uv sync --extra libero (or pip install -e .[libero])."
            )
        else:
            print(f"Eval rollouts skipped: no env registered for '{env_name}'.")
        return {"eval/return_mean": 0.0}

    viz_dir = run_dir / "viz" / "samples" / f"step_{step:06d}"
    viz_dir.mkdir(parents=True, exist_ok=True)
    use_wandb_video = logger is not None and getattr(logger, "_wandb", None) is not None
    collect_frames = save_video or use_wandb_video
    video_folder = viz_dir / "videos"
    if collect_frames:
        video_folder.mkdir(parents=True, exist_ok=True)
        print(f"Eval rollout: saving per-trial videos to {video_folder.resolve()}")
    elif save_video:
        video_folder.mkdir(parents=True, exist_ok=True)
        env, video_ok = _wrap_record_video(env, video_folder)
        if video_ok:
            print(f"Eval rollout videos will be saved to: {video_folder.resolve()}")
        else:
            print(
                "Warning: save_eval_video=true but no RecordVideo/Monitor wrapper available."
            )
    all_frames_for_wandb: List[np.ndarray] = []

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

    from src.engine.eval_context import build_prompt_tuple
    from src.engine.reward_models import get_return_from_env, get_return_from_reward_model

    def _return_for_traj(traj_dict: Dict[str, np.ndarray], ep_return: float) -> float:
        if eval_reward_source == "env":
            return ep_return
        if eval_reward_source == "reward_model" and eval_reward_model:
            return get_return_from_reward_model(
                task_description or "",
                traj_dict,
                eval_reward_model,
            )
        return ep_return

    all_returns: List[float] = []
    all_lengths: List[int] = []
    all_states_list: List[np.ndarray] = []
    all_actions_list: List[np.ndarray] = []

    state_dim = getattr(model, "state_dim", env.observation_space.shape[0])
    act_dim = getattr(model, "act_dim", env.action_space.shape[0])
    K = eval_context_k
    total_len = total_prompt_len or 512
    max_traj_len = max_prompt_trajectory_length or 64

    if eval_context_mode == "zero_shot_adaptation":
        n_trials = eval_num_trials
        context_list: List[Tuple[Dict[str, np.ndarray], float]] = []
        for trial in range(n_trials):
            sorted_context = sorted(context_list, key=lambda x: x[1])
            trajs_for_prompt = [t for t, _ in sorted_context]
            returns_for_prompt = [r for _, r in sorted_context]
            if K is not None:
                trajs_for_prompt = trajs_for_prompt[-K:]
                returns_for_prompt = returns_for_prompt[-K:]
            prompt = None
            if trajs_for_prompt and total_len and max_traj_len:
                prompt = build_prompt_tuple(
                    trajs_for_prompt,
                    state_mean_t,
                    state_std_t,
                    total_len,
                    max_traj_len,
                    state_dim,
                    act_dim,
                    scale,
                    device,
                    sort_ascending=True,
                    trajectory_returns=returns_for_prompt or None,
                )
            ep_return, length, S, A, traj_dict, frames = _run_one_rollout(
                model, env, state_mean_t, state_std_t, device, scale, max_episode_steps,
                step, trial, prompt, collect_frames=collect_frames,
            )
            ret = _return_for_traj(traj_dict, ep_return)
            context_list.append((traj_dict, ret))
            if K is not None:
                context_list = context_list[-K:]  # keep most recent K trials
            all_returns.append(ep_return)
            all_lengths.append(length)
            all_states_list.append(S)
            all_actions_list.append(A)
            if collect_frames and frames:
                _write_trial_video(video_folder, trial, frames)
                for f in frames:
                    all_frames_for_wandb.append(_add_trial_text(f, f"Trial {trial + 1}"))
    elif eval_context_mode == "prompt" and prompt_trajectories:
        prompt = build_prompt_tuple(
            prompt_trajectories,
            state_mean_t,
            state_std_t,
            total_len,
            max_traj_len,
            state_dim,
            act_dim,
            scale,
            device,
            sort_ascending=True,
        )
        for ep in range(num_rollouts):
            ep_return, length, S, A, _, frames = _run_one_rollout(
                model, env, state_mean_t, state_std_t, device, scale, max_episode_steps,
                step, ep, prompt, collect_frames=collect_frames,
            )
            all_returns.append(ep_return)
            all_lengths.append(length)
            all_states_list.append(S)
            all_actions_list.append(A)
            if collect_frames and frames:
                _write_trial_video(video_folder, ep, frames)
                for f in frames:
                    all_frames_for_wandb.append(_add_trial_text(f, f"Trial {ep + 1}"))
    else:
        for ep in range(num_rollouts):
            ep_return, length, S, A, _, frames = _run_one_rollout(
                model, env, state_mean_t, state_std_t, device, scale, max_episode_steps,
                step, ep, None, collect_frames=collect_frames,
            )
            all_returns.append(ep_return)
            all_lengths.append(length)
            all_states_list.append(S)
            all_actions_list.append(A)
            if collect_frames and frames:
                _write_trial_video(video_folder, ep, frames)
                for f in frames:
                    all_frames_for_wandb.append(_add_trial_text(f, f"Trial {ep + 1}"))

    env.close()
    if logger and all_frames_for_wandb:
        logger.log_video("eval/rollout_video", all_frames_for_wandb, step=step, fps=20)

    # Save visualizations
    viz_dir = run_dir / "viz" / "samples" / f"step_{step:06d}"
    viz_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_eps = len(all_states_list)
    for i in range(min(n_eps, 20)):
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        S = all_states_list[i]
        A = all_actions_list[i]
        if S.ndim == 1:
            S = S.reshape(-1, 1)
        if A.ndim == 1:
            A = A.reshape(-1, 1)
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

    return {
        "eval/return_mean": float(np.mean(all_returns)),
        "eval/return_std": float(np.std(all_returns)),
        "eval/len_mean": float(np.mean(all_lengths)),
    }
