"""
Eval rollout visualization: run env rollouts and save state/action curves and returns to viz/samples/step_XXXXX/.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Dataset / config may use D4RL-era ids (e.g. HalfCheetah-v2). Core Gymnasium only registers
# modern MuJoCo envs (v4+); v2/v3 in gymnasium-robotics still need deprecated mujoco_py.
# Map to v5 for eval when observation/action shapes match (HalfCheetah: 17 / 6).
_GYMNASIUM_EVAL_ENV_ALIASES = {
    "HalfCheetah-v2": "HalfCheetah-v5",
    "halfcheetah-v2": "HalfCheetah-v5",
}
_EVAL_ENV_ALIAS_LOGGED: set[str] = set()


def _log_eval_transformer_seq(
    model: Any,
    prompt: Optional[Tuple[Any, ...]],
    *,
    env_name: str,
    max_episode_steps: int,
    tag: str,
) -> None:
    """
    Print how long the transformer sequence is on each eval get_action call.

    OOM during eval is usually from attention over (prompt + padded query): with condition_rtg,
    each env/query timestep becomes 3 tokens, so positions ≈ 3 * (prompt_timesteps + model.max_length).
    That is fixed for the whole rollout (not growing with env t) but huge if max_total_prompt_length is large.
    """
    condition_rtg = getattr(model, "_condition_rtg", True)
    tokens_per_step = 3 if condition_rtg else 2
    max_len = getattr(model, "max_length", None)
    if max_len is None:
        max_len = 0
    t_prompt = 0
    if prompt is not None and prompt[0] is not None:
        t_prompt = int(prompt[0].shape[1])
    total_tokens = tokens_per_step * (t_prompt + int(max_len))
    npos = None
    tr = getattr(model, "transformer", None)
    if tr is not None:
        cfg = getattr(tr, "config", None)
        if cfg is not None:
            npos = getattr(cfg, "n_positions", None)
    print(
        f"Eval transformer seq [{tag}] env={env_name!r}: "
        f"prompt_timesteps={t_prompt}, query_padded_to_max_length={max_len}, "
        f"tokens_per_dt_step={tokens_per_step} => "
        f"~{total_tokens} positions per get_action call; "
        f"rollout max_episode_steps={max_episode_steps}.",
        flush=True,
    )
    if npos is not None:
        print(f"  backbone n_positions={npos}", flush=True)
    if total_tokens > 8192:
        print(
            "  WARNING: very long eval context — consider lowering data.max_total_prompt_length "
            "(or a dedicated eval prompt cap), smaller model, or eval on CPU.",
            flush=True,
        )


def _render_rgb_frame(env: Any) -> Optional[np.ndarray]:
    """RGB frame from Gymnasium env (create with render_mode='rgb_array')."""
    frame = env.render()
    if frame is None or not isinstance(frame, np.ndarray):
        return None
    return np.asarray(frame)


def _try_make_env(
    env_name: str,
    render_both_views: bool = True,
    render_mode: Optional[str] = None,
):
    """Create env: LIBERO suite (libero_10, ...) via make_libero_env, else Gymnasium/gym. Returns None if env or deps missing.
    render_both_views is only used for LIBERO (primary + wrist); Gymnasium envs have a single camera.
    """
    from src.envs.libero_env import LIBERO_SUITES, make_libero_env

    if env_name in LIBERO_SUITES:
        return make_libero_env(
            suite_name=env_name,
            task_id=0,
            state_dim=9,
            action_dim=7,
            render_both_views=render_both_views,
        )

    import gymnasium as gym

    gym_id = _GYMNASIUM_EVAL_ENV_ALIASES.get(env_name, env_name)
    if gym_id != env_name and env_name not in _EVAL_ENV_ALIAS_LOGGED:
        _EVAL_ENV_ALIAS_LOGGED.add(env_name)
        print(
            f"Eval env alias: {env_name!r} -> {gym_id!r} "
            f"(core Gymnasium MuJoCo; same obs/act dims as D4RL/Minari HalfCheetah)"
        )

    try:
        if render_mode is not None:
            return gym.make(gym_id, render_mode=render_mode)
        return gym.make(gym_id)
    except Exception as e:
        msg = str(e).lower()
        # If alias failed or user passed another -v2 id, give a clear hint.
        if (
            "gymnasium-robotics" in msg
            or "mujoco v2" in msg
            or "mujoco v3" in msg
        ):
            raise RuntimeError(
                f"Could not create Gymnasium env {gym_id!r} (from config {env_name!r}). "
                f"For HalfCheetah, use MuJoCo-backed Gymnasium (HalfCheetah-v5) and "
                f"`uv sync --extra d4rl` so `mujoco` is installed. Original error: {e}"
            ) from e
        raise


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
            gym.wrappers.RecordVideo(env, str(video_folder), episode_trigger=lambda ep: True),
            True,
        )
    return env, False


def _write_trial_video(
    video_folder: Path, trial: int, frames: List[np.ndarray], fps: int = 20
) -> None:
    """Write frames to video_folder/trial_{trial}.mp4."""
    if not frames:
        return
    import imageio

    path = video_folder / f"trial_{trial}.mp4"
    writer = imageio.get_writer(str(path), fps=fps)
    for f in frames:
        writer.append_data(np.asarray(f))
    writer.close()


def _write_frames_video(
    video_folder: Path, filename: str, frames: List[np.ndarray], fps: int = 20
) -> None:
    """Write a list of frames to video_folder/{filename}.mp4."""
    if not frames:
        return
    import imageio

    path = video_folder / filename
    writer = imageio.get_writer(str(path), fps=fps)
    for f in frames:
        writer.append_data(np.asarray(f))
    writer.close()


def _add_trial_text(frame: np.ndarray, label: str, y_offset: int = 18) -> np.ndarray:
    """Draw label on frame (copy). Requires cv2. y_offset is vertical position from top."""
    import cv2

    out = np.asarray(frame).copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, min(w, h) / 400.0)
    thick = max(1, int(scale * 2))
    cv2.putText(out, label, (10, y_offset), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    cv2.putText(out, label, (10, y_offset), font, scale, (0, 0, 0), max(1, thick - 1), cv2.LINE_AA)
    return out


def _preprocess_frames_for_encoder(
    frames: List[np.ndarray],
    device: Any,
    size: Tuple[int, int] = (224, 224),
) -> Any:
    """Convert list of (H, W, 3) uint8 to (1, T, 3, H, W) float, ImageNet normalized. Matches precompute_libero_embeddings."""
    import torch

    if not frames:
        return None
    h, w = size
    try:
        import cv2

        resized = np.stack(
            [cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) for f in frames],
            axis=0,
        )
    except Exception:
        resized = np.stack([np.asarray(f) for f in frames], axis=0)
        if resized.shape[1:3] != (h, w):
            return None
    x = torch.from_numpy(resized).float().to(device)
    x = x.permute(0, 3, 1, 2)
    x = x / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x.unsqueeze(0)


def _encode_rollout_images(
    image_list: List[Tuple[Any, Any]],
    model: Any,
    device: Any,
) -> Optional[Any]:
    """Build (1, T, D) image embeddings from list of (primary, wrist) frames. Returns None if no encoder or no images."""
    import torch

    vision_encoder = getattr(model, "vision_encoder", None)
    if vision_encoder is None or not image_list:
        return None
    primary_frames = []
    wrist_frames = []
    for p, w in image_list:
        if p is not None:
            primary_frames.append(p)
        if w is not None:
            wrist_frames.append(w)
    if not primary_frames and not wrist_frames:
        return None
    if not primary_frames:
        primary_frames = wrist_frames
    if not wrist_frames:
        wrist_frames = primary_frames
    v0 = _preprocess_frames_for_encoder(primary_frames, device)
    v1 = _preprocess_frames_for_encoder(wrist_frames, device)
    if v0 is None or v1 is None:
        return None
    with torch.no_grad():
        emb = vision_encoder([v0, v1])
    if emb is not None and emb.dim() == 3:
        return emb
    return None


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
    env_name: str = "",
) -> Tuple[float, int, np.ndarray, np.ndarray, Dict[str, np.ndarray], List[np.ndarray]]:
    """Run one episode; return (return, length, states, actions, trajectory_dict, frames)."""
    import torch

    obs, _ = env.reset(seed=step + ep)
    if isinstance(obs, tuple):
        obs = obs[0]
    frames: List[np.ndarray] = []
    if collect_frames:
        frame = _render_rgb_frame(env)
        if frame is not None:
            frames.append(frame)
    use_vision = getattr(model, "vision_encoder", None) is not None
    get_images = getattr(env, "get_current_images", None)
    image_list: List[Tuple[Any, Any]] = []
    if use_vision and get_images is not None:
        prim, wrist = get_images()
        image_list.append((prim, wrist))
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
    ml_q = getattr(model, "max_length", None)
    for t in range(max_episode_steps):
        if t % 25 == 0 or t < 3:
            n_live = int(states.shape[0])
            print(
                f"[eval rollout] env={env_name!r} ep={ep} env_timestep={t}/{max_episode_steps} "
                f"live_query_length={n_live} states (query padded to max_length={ml_q} inside get_action)",
                flush=True,
            )
        image_embeddings = None
        if use_vision and image_list:
            image_embeddings = _encode_rollout_images(image_list, model, device)
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
            image_embeddings=image_embeddings,
        )
        if action.dim() == 1:
            action = action.unsqueeze(0)
        # Detach before feeding back into next get_action so we never chain autograd graphs
        # across env steps (OOM if eval runs without torch.inference_mode).
        action = action.detach()
        action_np = action.cpu().numpy().flatten()
        step_out = env.step(action_np)
        if len(step_out) == 5:
            next_obs, reward, done, truncated, _ = step_out
        else:
            next_obs, reward, done, _ = step_out[:4]
            truncated = False
        if use_vision and get_images is not None:
            prim, wrist = get_images()
            image_list.append((prim, wrist))
        if collect_frames:
            frame = _render_rgb_frame(env)
            if frame is not None:
                frames.append(frame)
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
    eval_render_both_views: bool = True,
) -> Dict[str, float]:
    """
    Run one eval rollout (N trials). Each trial can be saved as trial_0.mp4, trial_1.mp4, ...
    For wandb: log one continuous video with "Trial 1", "Trial 2", ... overlaid.
    """
    from src.envs.libero_env import LIBERO_SUITES

    use_wandb_video = logger is not None and getattr(logger, "_wandb", None) is not None
    collect_frames = save_video or use_wandb_video
    # Gymnasium needs render_mode="rgb_array" at construction for env.render() to return pixels.
    gym_render_mode: Optional[str] = None
    if collect_frames and env_name not in LIBERO_SUITES:
        gym_render_mode = "rgb_array"

    env = _try_make_env(
        env_name,
        render_both_views=eval_render_both_views,
        render_mode=gym_render_mode,
    )
    if env is None:
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
            print("Warning: save_eval_video=true but no RecordVideo/Monitor wrapper available.")
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
    max_traj_len = max_prompt_trajectory_length  # None = use full trajectory per demo

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
            if trajs_for_prompt and total_len:
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
            _log_eval_transformer_seq(
                model,
                prompt,
                env_name=env_name,
                max_episode_steps=max_episode_steps,
                tag=f"zero_shot trial {trial} / step {step}",
            )
            ep_return, length, S, A, traj_dict, frames = _run_one_rollout(
                model,
                env,
                state_mean_t,
                state_std_t,
                device,
                scale,
                max_episode_steps,
                step,
                trial,
                prompt,
                collect_frames=collect_frames,
                env_name=env_name,
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
                for t, f in enumerate(frames):
                    all_frames_for_wandb.append(_add_trial_text(f, f"Trial {trial + 1}  t={t}"))
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
        _log_eval_transformer_seq(
            model,
            prompt,
            env_name=env_name,
            max_episode_steps=max_episode_steps,
            tag=f"prompt mode step {step}",
        )
        for ep in range(num_rollouts):
            ep_return, length, S, A, _, frames = _run_one_rollout(
                model,
                env,
                state_mean_t,
                state_std_t,
                device,
                scale,
                max_episode_steps,
                step,
                ep,
                prompt,
                collect_frames=collect_frames,
                env_name=env_name,
            )
            all_returns.append(ep_return)
            all_lengths.append(length)
            all_states_list.append(S)
            all_actions_list.append(A)
            if collect_frames and frames:
                _write_trial_video(video_folder, ep, frames)
                for t, f in enumerate(frames):
                    all_frames_for_wandb.append(_add_trial_text(f, f"Trial {ep + 1}  t={t}"))
    else:
        if num_rollouts > 0:
            _log_eval_transformer_seq(
                model,
                None,
                env_name=env_name,
                max_episode_steps=max_episode_steps,
                tag=f"no prompt step {step}",
            )
        for ep in range(num_rollouts):
            ep_return, length, S, A, _, frames = _run_one_rollout(
                model,
                env,
                state_mean_t,
                state_std_t,
                device,
                scale,
                max_episode_steps,
                step,
                ep,
                None,
                collect_frames=collect_frames,
                env_name=env_name,
            )
            all_returns.append(ep_return)
            all_lengths.append(length)
            all_states_list.append(S)
            all_actions_list.append(A)
            if collect_frames and frames:
                _write_trial_video(video_folder, ep, frames)
                for t, f in enumerate(frames):
                    all_frames_for_wandb.append(_add_trial_text(f, f"Trial {ep + 1}  t={t}"))

    env.close()
    # Save combined video (all trials with labels) and each trial separately; then log to wandb
    if collect_frames and all_frames_for_wandb:
        all_trials_path = video_folder / "all_trials.mp4"
        _write_frames_video(video_folder, "all_trials.mp4", all_frames_for_wandb, fps=20)
        if logger is not None and getattr(logger, "_wandb", None) is not None:
            # Log main video from array so W&B respects fps (path-based logging ignores fps and warns)
            logger.log_video("eval/rollout_video", all_frames_for_wandb, step=step, fps=20)
            for i in range(len(all_returns)):
                trial_path = video_folder / f"trial_{i}.mp4"
                if trial_path.exists():
                    logger.log_video_from_path(
                        f"eval/rollout_video_trial_{i}", trial_path, step=step
                    )

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
    from matplotlib.ticker import MaxNLocator

    _font = {"family": "serif", "serif": ["Palatino", "Palatino Linotype", "DejaVu Serif"]}
    with plt.rc_context({"font.family": "serif", "font.serif": _font["serif"]}):
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
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
        ax.set_title(f"Eval step {step}")
        fig2.savefig(viz_dir / "returns.png", dpi=100)
        plt.close(fig2)

    # Cumulative return over trials with trial boundaries
    if len(all_returns) > 0:
        cum_returns = np.cumsum(all_returns)
        with plt.rc_context({"font.family": "serif", "font.serif": _font["serif"]}):
            fig3, ax3 = plt.subplots(figsize=(6, 3.5))
            x = np.arange(len(all_returns), dtype=int)
            ax3.plot(x, cum_returns, color="steelblue", linewidth=2, marker="o", markersize=4)
            for i in range(1, len(all_returns)):
                ax3.axvline(x=i - 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
            ax3.set_xlabel("trial")
            ax3.set_ylabel("cumulative return")
            ax3.set_ylim(0, 1)
            ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax3.set_title(f"Eval step {step}: cumulative return")
            ax3.grid(True, alpha=0.3)
        fig3.tight_layout()
        cum_path = viz_dir / "eval_cumulative_return.png"
        fig3.savefig(cum_path, dpi=100)
        plt.close(fig3)
        if logger is not None and getattr(logger, "_wandb", None) is not None:
            logger.log_image("eval/cumulative_return", str(cum_path), step=step)

    return {
        "eval/return_mean": float(np.mean(all_returns)),
        "eval/return_std": float(np.std(all_returns)),
        "eval/len_mean": float(np.mean(all_lengths)),
    }
