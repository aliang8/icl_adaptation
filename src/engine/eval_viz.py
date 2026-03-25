"""
Eval rollout visualization: run env rollouts and save state/action curves and returns to viz/samples/step_XXXXX/.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from src.data.reward_normalization import load_stats, normalize_reward_scalar

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
        if "gymnasium-robotics" in msg or "mujoco v2" in msg or "mujoco v3" in msg:
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


def _annotate_eval_frame(frame: np.ndarray, lines: List[str]) -> np.ndarray:
    """Draw multiple lines (white stroke + black fill) on frame copy. Requires cv2."""
    import cv2

    out = np.asarray(frame).copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    sc = max(0.45, min(w, h) / 480.0)
    thick = max(1, int(sc * 2))
    y = int(18 * sc + 12)
    for line in lines:
        if not line:
            continue
        cv2.putText(out, line, (10, y), font, sc, (255, 255, 255), thick, cv2.LINE_AA)
        cv2.putText(out, line, (10, y), font, sc, (0, 0, 0), max(1, thick - 1), cv2.LINE_AA)
        y += int(24 * sc + 10)
    return out


def _annotated_rollout_frames(
    model: Any,
    frames: List[np.ndarray],
    rtg_per_frame: List[float],
    trial_tag: str,
) -> List[np.ndarray]:
    """Overlay trial id, timestep, and RTG scalar seen by the model at that frame (before get_action)."""
    cond_rtg = getattr(model, "_condition_rtg", True)
    out: List[np.ndarray] = []
    for t, f in enumerate(frames):
        lines = [f"{trial_tag}  t={t}"]
        if cond_rtg and t < len(rtg_per_frame):
            lines.append(f"RTG (model input, /return_scale)={rtg_per_frame[t]:.4f}")
        elif not cond_rtg:
            lines.append("RTG conditioning: off")
        out.append(_annotate_eval_frame(f, lines))
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
    normalize_reward_for_rtg: Optional[Callable[[float], float]] = None,
    prompt_builder: Optional[
        Callable[
            [List[np.ndarray], List[np.ndarray], List[float]],
            Tuple[Optional[Tuple[Any, ...]], Dict[str, Any]],
        ]
    ] = None,
) -> Tuple[float, int, np.ndarray, np.ndarray, Dict[str, np.ndarray], List[np.ndarray], List[float]]:
    """Run one episode. Returns frames and rtg_for_frames (same length): RTG scalar at query tail
    before each get_action (model input in /scale units; rewards in prompt use same scale)."""
    import torch

    rtg_for_frames: List[float] = []
    obs, _ = env.reset(seed=step + ep)
    if isinstance(obs, tuple):
        obs = obs[0]
    frames: List[np.ndarray] = []
    # Initialize RTG early so debug video collection at t=0 can record it.
    returns_to_go = torch.tensor([[scale]], device=device)
    if collect_frames:
        frame = _render_rgb_frame(env)
        if frame is not None:
            frames.append(frame)
            rtg_for_frames.append(float(returns_to_go[0, -1].detach().cpu()))
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
    timesteps = torch.zeros(1, 1, dtype=torch.long, device=device)
    ep_states = [obs.copy()]
    ep_actions: List[np.ndarray] = []
    ep_rewards: List[float] = []
    ep_return = 0.0
    ml_q = getattr(model, "max_length", None)
    for t in range(max_episode_steps):
        rtg_snap = float(returns_to_go[0, -1].detach().cpu())
        prompt_now = prompt
        prompt_meta: Dict[str, Any] = {"mode": "static", "prev_trials": 0, "current_trial_steps": 0}
        if prompt_builder is not None:
            prompt_now, prompt_meta = prompt_builder(ep_states, ep_actions, ep_rewards)
        if t % 25 == 0 or t < 3:
            n_live = int(states.shape[0])
            p_steps = 0
            if prompt_now is not None and prompt_now[0] is not None:
                p_steps = int(prompt_now[0].shape[1])
            mode = prompt_meta.get("mode")
            if mode == "zero_shot_dynamic":
                br = prompt_meta.get("prior_per_trial_steps") or []
                br_s = "+".join(str(x) for x in br) if br else "0"
                print(
                    f"[eval rollout] env={env_name!r} trial={ep} t={t}/{max_episode_steps} "
                    f"query_live_steps={n_live} model_max_length={ml_q}; "
                    f"prompt_T={p_steps} prompt_cap={prompt_meta.get('total_prompt_cap')}; "
                    f"prior_trials={prompt_meta.get('prev_trials', 0)} "
                    f"prior_steps_by_trial=[{br_s}] prior_ctx_sum={prompt_meta.get('prior_context_steps')}; "
                    f"current_window_steps={prompt_meta.get('current_trial_window_steps')} "
                    f"raw_concat={prompt_meta.get('raw_concat_steps')}; "
                    f"approx_tf_pos={prompt_meta.get('approx_transformer_positions')}",
                    flush=True,
                )
            else:
                print(
                    f"[eval rollout] env={env_name!r} trial={ep} t={t}/{max_episode_steps} "
                    f"query_live_steps={n_live} (model max_length={ml_q}); "
                    f"prompt_steps={p_steps}; mode={mode}; "
                    f"prev_trials_in_prompt={prompt_meta.get('prev_trials', 0)}; "
                    f"current_trial_steps_in_prompt={prompt_meta.get('current_trial_steps', 0)}",
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
            prompt=prompt_now,
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
                rtg_for_frames.append(rtg_snap)
        if isinstance(next_obs, tuple):
            next_obs = next_obs[0]
        r_env = float(reward)
        r_model = normalize_reward_for_rtg(r_env) if normalize_reward_for_rtg else r_env
        ep_return += r_env
        ep_rewards.append(r_model)
        ep_states.append(next_obs.copy())
        ep_actions.append(action_np.copy())
        actions_t = torch.cat([actions_t, action], dim=0)
        rewards_t = torch.cat([rewards_t, torch.tensor([r_model], device=device)])
        states = torch.cat(
            [states, torch.from_numpy(next_obs).float().reshape(1, -1).to(device)], dim=0
        )
        returns_to_go = torch.cat(
            [returns_to_go, (returns_to_go[0, -1] - r_model / scale).reshape(1, 1)], dim=1
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
        rtg_for_frames,
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
    context_subsample_strategy: str = "none",
    task_description: Optional[str] = None,
    logger: Optional[Any] = None,
    eval_render_both_views: bool = True,
    reward_normalization: str = "none",
    reward_norm_constant: float = 1.0,
    reward_norm_epsilon: float = 1e-8,
    reward_normalization_stats_path: Optional[str] = None,
    wandb_defer_step_commit: bool = False,
) -> Dict[str, float]:
    """
    Run one eval rollout (N trials). Each trial can be saved as trial_0.mp4, trial_1.mp4, ...
    For wandb: log one continuous video with "Trial 1", "Trial 2", ... overlaid.
    When wandb_defer_step_commit=True (training eval), media uses commit=False so the trainer
    can log scalars at the same step with commit=True without W&B dropping out-of-order steps.
    """
    from src.envs.libero_env import LIBERO_SUITES

    use_wandb_video = logger is not None and logger._wandb is not None
    # Several media logs share one W&B step; never commit=True on each video or the step advances N times.
    wb_commit_video = False
    wb_commit_charts = not wandb_defer_step_commit
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

    mode = (reward_normalization or "none").strip().lower()
    reward_stats: Optional[Dict[str, Any]] = None
    stats_path = reward_normalization_stats_path
    if isinstance(stats_path, str) and stats_path.strip().lower() in ("", "none", "null", "~"):
        stats_path = None
    if mode in ("standardize", "dataset_std", "zscore", "minmax", "dataset_minmax"):
        if stats_path:
            sp = Path(stats_path)
            if sp.is_file():
                reward_stats = load_stats(sp)
                print(f"Eval reward normalization: loaded stats from {sp}", flush=True)
            else:
                raise FileNotFoundError(
                    f"reward_normalization_stats_path not found: {sp}. "
                    "For eval RTG normalization in standardize/minmax mode, provide stats JSON."
                )
        else:
            raise ValueError(
                "reward_normalization requires data.reward_normalization_stats_path for "
                "standardize/minmax at eval (to match training stats)."
            )

    def _normalize_reward_for_model(r: float) -> float:
        return normalize_reward_scalar(
            r,
            mode,
            reward_norm_constant=reward_norm_constant,
            epsilon=reward_norm_epsilon,
            stats=reward_stats,
        )

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
        current_trial_k = int(K) if K is not None else int(getattr(model, "max_length", 20))
        ml_model = getattr(model, "max_length", None)
        cond_rtg = getattr(model, "_condition_rtg", True)
        tps = 3 if cond_rtg else 2
        print(
            f"[zero_shot eval] Policy: [all completed prior trials] + "
            f"[last {current_trial_k} env steps of current trial]. "
            f"total_prompt_cap={total_len} max_prompt_traj_len={max_traj_len} "
            f"context_subsample_strategy={context_subsample_strategy} "
            f"query_pad_max_length={ml_model} tokens_per_dt_step={tps}. "
            f"Prompt tensor: left-pad, valid prompt timesteps right-aligned with query (matches train).",
            flush=True,
        )
        for trial in range(n_trials):
            sorted_context = sorted(context_list, key=lambda x: x[1])
            prev_trajs = [t for t, _ in sorted_context]
            prev_returns = [r for _, r in sorted_context]

            def _capped_traj_steps(traj: Dict[str, np.ndarray]) -> int:
                tlen = int(len(traj["rewards"]))
                if max_traj_len is not None and tlen > max_traj_len:
                    return int(max_traj_len)
                return tlen

            prior_per_trial_steps = [_capped_traj_steps(t) for t in prev_trajs]
            prior_ctx_steps = sum(prior_per_trial_steps)
            prior_steps_str = (
                "+".join(str(x) for x in prior_per_trial_steps) if prior_per_trial_steps else "0"
            )

            def _build_prompt_with_current(
                ep_states_now: List[np.ndarray],
                ep_actions_now: List[np.ndarray],
                ep_rewards_now: List[float],
            ) -> Tuple[Optional[Tuple[Any, ...]], Dict[str, Any]]:
                curr_steps = len(ep_actions_now)
                use_steps = min(current_trial_k, curr_steps)
                trajs = list(prev_trajs)
                rets = list(prev_returns)
                raw_concat = prior_ctx_steps + use_steps
                meta: Dict[str, Any] = {
                    "mode": "zero_shot_dynamic",
                    "prev_trials": len(prev_trajs),
                    "prior_context_steps": prior_ctx_steps,
                    "prior_per_trial_steps": list(prior_per_trial_steps),
                    "current_trial_window_steps": use_steps,
                    "current_trial_steps": use_steps,
                    "raw_concat_steps": raw_concat,
                    "total_prompt_cap": total_len,
                }
                if use_steps > 0:
                    start = curr_steps - use_steps
                    curr_obs = np.asarray(
                        ep_states_now[start : start + use_steps], dtype=np.float32
                    )
                    curr_act = np.asarray(ep_actions_now[start:curr_steps], dtype=np.float32)
                    curr_rew = np.asarray(ep_rewards_now[start:curr_steps], dtype=np.float32)
                    curr_traj = {
                        "observations": curr_obs,
                        "actions": curr_act,
                        "rewards": curr_rew,
                    }
                    trajs.append(curr_traj)
                    # Keep current-trial window as the final prompt chunk by assigning max sorting score.
                    base = max(rets) if rets else 0.0
                    rets.append(base + 1e-6)
                if not trajs or not total_len:
                    return None, meta
                prompt_dyn = build_prompt_tuple(
                    trajs,
                    state_mean_t,
                    state_std_t,
                    total_len,
                    max_traj_len,
                    state_dim,
                    act_dim,
                    scale,
                    device,
                    sort_ascending=True,
                    trajectory_returns=rets or None,
                    context_subsample_strategy=context_subsample_strategy,
                )
                pt = int(prompt_dyn[0].shape[1])
                meta["prompt_timesteps_in_model"] = pt
                ml = int(ml_model) if ml_model is not None else 0
                meta["approx_transformer_positions"] = tps * (pt + ml)
                return prompt_dyn, meta

            print(
                f"[zero_shot eval] --- trial {trial + 1}/{n_trials} --- "
                f"prior_completed_trials={len(prev_trajs)} "
                f"prior_steps_by_trial(capped)=[{prior_steps_str}] "
                f"prior_context_steps(sum)={prior_ctx_steps} "
                f"current_window_K={current_trial_k}",
                flush=True,
            )
            _log_eval_transformer_seq(
                model,
                None,
                env_name=env_name,
                max_episode_steps=max_episode_steps,
                tag=f"zero_shot trial {trial} / step {step} (prompt length varies during rollout)",
            )
            ep_return, length, S, A, traj_dict, frames, rtg_ff = _run_one_rollout(
                model,
                env,
                state_mean_t,
                state_std_t,
                device,
                scale,
                max_episode_steps,
                step,
                trial,
                None,
                collect_frames=collect_frames,
                env_name=env_name,
                normalize_reward_for_rtg=_normalize_reward_for_model,
                prompt_builder=_build_prompt_with_current,
            )
            ret = _return_for_traj(traj_dict, ep_return)
            context_list.append((traj_dict, ret))
            print(
                f"[zero_shot eval] finished trial={trial}: added full trial to prior context. "
                f"prior_trials_now={len(context_list)}",
                flush=True,
            )
            all_returns.append(ep_return)
            all_lengths.append(length)
            all_states_list.append(S)
            all_actions_list.append(A)
            if collect_frames and frames:
                ann = _annotated_rollout_frames(model, frames, rtg_ff, f"Trial {trial + 1}")
                _write_trial_video(video_folder, trial, ann)
                all_frames_for_wandb.extend(ann)
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
            context_subsample_strategy=context_subsample_strategy,
        )
        _log_eval_transformer_seq(
            model,
            prompt,
            env_name=env_name,
            max_episode_steps=max_episode_steps,
            tag=f"prompt mode step {step}",
        )
        for ep in range(num_rollouts):
            ep_return, length, S, A, _, frames, rtg_ff = _run_one_rollout(
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
                normalize_reward_for_rtg=_normalize_reward_for_model,
            )
            all_returns.append(ep_return)
            all_lengths.append(length)
            all_states_list.append(S)
            all_actions_list.append(A)
            if collect_frames and frames:
                ann = _annotated_rollout_frames(model, frames, rtg_ff, f"Trial {ep + 1}")
                _write_trial_video(video_folder, ep, ann)
                all_frames_for_wandb.extend(ann)
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
            ep_return, length, S, A, _, frames, rtg_ff = _run_one_rollout(
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
                normalize_reward_for_rtg=_normalize_reward_for_model,
            )
            all_returns.append(ep_return)
            all_lengths.append(length)
            all_states_list.append(S)
            all_actions_list.append(A)
            if collect_frames and frames:
                ann = _annotated_rollout_frames(model, frames, rtg_ff, f"Trial {ep + 1}")
                _write_trial_video(video_folder, ep, ann)
                all_frames_for_wandb.extend(ann)

    env.close()
    # Save combined video (all trials with labels) and each trial separately; then log to wandb
    if collect_frames and all_frames_for_wandb:
        all_trials_path = video_folder / "all_trials.mp4"
        _write_frames_video(video_folder, "all_trials.mp4", all_frames_for_wandb, fps=20)
        if logger is not None and logger._wandb is not None:
            # Log main video from array so W&B respects fps (path-based logging ignores fps and warns)
            logger.log_video(
                "eval/rollout_video",
                all_frames_for_wandb,
                step=step,
                fps=20,
                wandb_commit=wb_commit_video,
            )
            for i in range(len(all_returns)):
                trial_path = video_folder / f"trial_{i}.mp4"
                if trial_path.exists():
                    logger.log_video_from_path(
                        f"eval/rollout_video_trial_{i}",
                        trial_path,
                        step=step,
                        wandb_commit=wb_commit_video,
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
    if len(all_returns) > 0:
        cum_returns = np.cumsum(all_returns)
        x = np.arange(len(all_returns), dtype=int)
        with plt.rc_context({"font.family": "serif", "font.serif": _font["serif"]}):
            fig_sum, (ax_bar, ax_cum) = plt.subplots(
                1, 2, figsize=(9, 3.5), constrained_layout=True
            )
            ax_bar.bar(x, all_returns, color="steelblue")
            ax_bar.axhline(
                np.mean(all_returns),
                color="red",
                linestyle="--",
                label=f"mean={np.mean(all_returns):.1f}",
            )
            ax_bar.set_xlabel("trial")
            ax_bar.set_ylabel("return")
            ax_bar.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax_bar.legend(loc="upper right", fontsize=8)
            ax_bar.set_title("Return per trial")
            ax_bar.grid(True, axis="y", alpha=0.3)

            ax_cum.plot(x, cum_returns, color="steelblue", linewidth=2, marker="o", markersize=4)
            for i in range(1, len(all_returns)):
                ax_cum.axvline(x=i - 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
            ax_cum.set_xlabel("trial")
            ax_cum.set_ylabel("cumulative return")
            ax_cum.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax_cum.set_title("Cumulative return (all trials)")
            ax_cum.grid(True, alpha=0.3)
            fig_sum.suptitle(f"Eval step {step}", fontsize=11, y=1.02)

        summary_path = viz_dir / "eval_trials_returns.png"
        fig_sum.savefig(summary_path, dpi=100, bbox_inches="tight")
        plt.close(fig_sum)
        if logger is not None and logger._wandb is not None:
            import wandb

            p = str(summary_path)
            logger.log_wandb_dict(
                {
                    "eval/trials_returns": wandb.Image(p),
                    "eval/cumulative_return": wandb.Image(p),
                },
                step=step,
                wandb_commit=wb_commit_charts,
            )

    return {
        "eval/return_mean": float(np.mean(all_returns)),
        "eval/return_std": float(np.std(all_returns)),
        "eval/len_mean": float(np.mean(all_lengths)),
    }
