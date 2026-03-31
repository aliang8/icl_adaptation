"""
Eval rollout visualization: run env rollouts and save state/action curves and returns to viz/samples/step_XXXXX/.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from src.data.rtg import initial_rtg_token
from src.models.meta_dt import MetaDecisionTransformer

# Dataset / config may use D4RL-era ids (e.g. HalfCheetah-v2). Core Gymnasium only registers
# modern MuJoCo envs (v4+); v2/v3 in gymnasium-robotics still need deprecated mujoco_py.
# Map to v5 for eval when observation/action shapes match (HalfCheetah: 17 / 6).
_GYMNASIUM_EVAL_ENV_ALIASES = {
    "HalfCheetah-v2": "HalfCheetah-v5",
    "halfcheetah-v2": "HalfCheetah-v5",
}
_EVAL_ENV_ALIAS_LOGGED: set[str] = set()


def _action_prediction_stats_from_rollouts(
    rollouts_actions: List[np.ndarray],
) -> Dict[str, float]:
    """Mean / min / max over all predicted actions (env steps × rollouts)."""
    if not rollouts_actions:
        return {}
    parts: List[np.ndarray] = []
    for a in rollouts_actions:
        if a is None or a.size == 0:
            continue
        x = np.asarray(a, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        parts.append(x.reshape(-1, x.shape[-1]))
    if not parts:
        return {}
    A = np.concatenate(parts, axis=0)
    if A.size == 0:
        return {}
    return {
        "eval/action_pred_mean": float(np.mean(A)),
        "eval/action_pred_min": float(np.min(A)),
        "eval/action_pred_max": float(np.max(A)),
    }


def _log_eval_transformer_seq(
    model: Any,
    prompt: Optional[Tuple[Any, ...]],
    *,
    env_name: str,
    max_episode_steps: int,
    tag: str,
    query_window: Optional[int] = None,
) -> None:
    """
    Print how long the transformer sequence is on each eval get_action call.

    OOM during eval is usually from attention over (prompt + padded query): with condition_rtg,
    each env/query timestep becomes 3 tokens, so positions ≈ 3 * (prompt_timesteps + K).
    K is ``query_window`` when passed (data horizon / query_history_length), else ``model.max_length``.
    With no ICL prompt, the query uses left-pad only until history reaches K, then exactly K steps.
    """
    if not isinstance(model, MetaDecisionTransformer):
        raise TypeError(f"eval expects MetaDecisionTransformer, got {type(model)}")
    condition_rtg = model._condition_rtg
    tokens_per_step = 3 if condition_rtg else 2
    if query_window is not None:
        max_len = int(query_window)
    else:
        max_len = model.max_length if model.max_length is not None else 0
    t_prompt = 0
    if prompt is not None and prompt[0] is not None:
        t_prompt = int(prompt[0].shape[1])
    total_tokens = tokens_per_step * (t_prompt + int(max_len))
    tr = model.transformer
    npos = tr.config.n_positions
    print(
        f"Eval transformer seq [{tag}] env={env_name!r}: "
        f"prompt_timesteps={t_prompt}, query_window_K={max_len}, "
        f"tokens_per_dt_step={tokens_per_step} => "
        f"~{total_tokens} positions per get_action call; "
        f"rollout max_episode_steps={max_episode_steps}.",
        flush=True,
    )
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
    *,
    vd4rl_eval_pixel_hw: Optional[int] = None,
    vd4rl_eval_obs_downsample: Optional[int] = None,
    vd4rl_eval_seed: int = 0,
    minari_halfcheetah_dataset_id: Optional[str] = None,
):
    """Create env: LIBERO suite (libero_10, ...) via make_libero_env, else Gymnasium/gym. Returns None if env or deps missing.
    render_both_views is only used for LIBERO (primary + wrist); Gymnasium envs have a single camera.
    """
    from src.envs.libero_env import LIBERO_SUITES, make_libero_env

    if env_name.startswith("VD4RL/dmc/"):
        from src.envs.vd4rl_eval_env import make_vd4rl_dm_control_pixel_env

        if vd4rl_eval_pixel_hw is None or vd4rl_eval_obs_downsample is None:
            raise ValueError(
                f"Env {env_name!r} requires data.vd4rl_pixel_size and data.vd4rl_obs_downsample "
                "(passed as vd4rl_eval_pixel_hw / vd4rl_eval_obs_downsample from train)."
            )
        return make_vd4rl_dm_control_pixel_env(
            env_name,
            pixel_hw=int(vd4rl_eval_pixel_hw),
            obs_downsample=int(vd4rl_eval_obs_downsample),
            seed=int(vd4rl_eval_seed),
        )

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

    if minari_halfcheetah_dataset_id and gym_id == "HalfCheetah-v5":
        from src.envs.minari_halfcheetah_eval import make_halfcheetah_env_via_minari

        return make_halfcheetah_env_via_minari(
            minari_halfcheetah_dataset_id, render_mode=render_mode
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
        env = object.__getattribute__(self, "_env")
        return object.__getattribute__(env, name)

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

    return (
        gym.wrappers.RecordVideo(env, str(video_folder), episode_trigger=lambda ep: True),
        True,
    )


def _write_rollout_video(
    video_folder: Path, rollout_idx: int, frames: List[np.ndarray], fps: int = 20
) -> None:
    """Write frames to video_folder/rollout_{rollout_idx}.mp4 (one clip per eval rollout index)."""
    if not frames:
        return
    import imageio

    path = video_folder / f"rollout_{rollout_idx}.mp4"
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


def _pad_ragged_1d(arrays: List[np.ndarray], fill: float = float("nan")) -> np.ndarray:
    """Stack 1d arrays of different lengths into (N, T_max) with ``fill`` padding."""
    if not arrays:
        return np.zeros((0, 0), dtype=np.float64)
    T = max(int(np.asarray(a).shape[0]) for a in arrays)
    out = np.full((len(arrays), T), fill, dtype=np.float64)
    for i, a in enumerate(arrays):
        v = np.asarray(a, dtype=np.float64).reshape(-1)
        n = int(v.shape[0])
        out[i, :n] = v
    return out


def _save_eval_rtg_reward_figure(
    reward_rows: List[np.ndarray],
    rtg_rows: List[np.ndarray],
    *,
    rtg_scale: float,
    step: int,
    out_path: Path,
    condition_rtg: bool,
) -> bool:
    """Mean ± std over rollouts (ragged steps padded with NaN). Palatino styling. Returns False if nothing to plot."""
    if not reward_rows or all(np.asarray(r).size == 0 for r in reward_rows):
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R = _pad_ragged_1d(reward_rows)
    n_roll = int(R.shape[0])
    T = int(R.shape[1])
    x = np.arange(T, dtype=int)

    _serif = ["Palatino", "Palatino Linotype", "DejaVu Serif"]
    bg = "#FAFAF8"
    c_r = "#2E86AB"
    c_rtg = "#8B3A62"

    def _plot_reward(ax_r: Any, subtitle: str) -> None:
        if n_roll <= 1:
            ax_r.plot(x, R[0], color=c_r, linewidth=2.0, label="rollout")
            ax_r.set_title(subtitle)
        else:
            r_mean = np.nanmean(R, axis=0)
            r_std = np.nanstd(R, axis=0)
            r_std = np.nan_to_num(r_std, nan=0.0, posinf=0.0, neginf=0.0)
            ax_r.fill_between(x, r_mean - r_std, r_mean + r_std, color=c_r, alpha=0.28, linewidth=0)
            ax_r.plot(x, r_mean, color=c_r, linewidth=2.0, label="mean ± std")
            ax_r.set_title(subtitle)
        ax_r.set_ylabel("env reward")
        ax_r.grid(True, axis="y", alpha=0.4)
        ax_r.legend(loc="upper right", fontsize=8, framealpha=0.92)

    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": _serif,
            "axes.facecolor": "#FFFFFF",
            "figure.facecolor": bg,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "grid.alpha": 0.35,
            "grid.linestyle": "-",
        }
    ):
        if condition_rtg and rtg_rows and not all(np.asarray(t).size == 0 for t in rtg_rows):
            G = _pad_ragged_1d(rtg_rows)
            share_x = int(G.shape[1]) == int(R.shape[1])
            fig, (ax_r, ax_g) = plt.subplots(
                2,
                1,
                figsize=(9, 5.4),
                sharex=share_x,
                constrained_layout=True,
                gridspec_kw={"height_ratios": [1, 1], "hspace": 0.08},
            )
            r_sub = (
                "Reward per env step (single eval episode)"
                if n_roll <= 1
                else f"Reward per env step (mean ± std over {n_roll} eval episodes)"
            )
            _plot_reward(ax_r, r_sub)

            n_rtg = int(G.shape[0])
            if n_rtg <= 1:
                ax_g.plot(np.arange(G.shape[1]), G[0], color=c_rtg, linewidth=2.0, label="rollout")
                ax_g.set_title("RTG token (tail before each action)")
            else:
                g_mean = np.nanmean(G, axis=0)
                g_std = np.nanstd(G, axis=0)
                g_std = np.nan_to_num(g_std, nan=0.0, posinf=0.0, neginf=0.0)
                xg = np.arange(G.shape[1], dtype=int)
                ax_g.fill_between(
                    xg, g_mean - g_std, g_mean + g_std, color=c_rtg, alpha=0.28, linewidth=0
                )
                ax_g.plot(xg, g_mean, color=c_rtg, linewidth=2.0, label="mean ± std")
                ax_g.set_title(
                    f"RTG token (mean ± std, n={n_rtg}); update −r / rtg_scale (rtg_scale={rtg_scale:g})"
                )
            ax_g.set_ylabel("RTG token")
            ax_g.set_xlabel("environment step")
            ax_g.grid(True, axis="y", alpha=0.4)
            ax_g.legend(loc="upper right", fontsize=8, framealpha=0.92)
        else:
            fig, ax_r = plt.subplots(1, 1, figsize=(9, 3.4), constrained_layout=True)
            r_sub = (
                "Reward per env step (single eval episode); RTG conditioning off"
                if n_roll <= 1
                else f"Reward per env step (mean ± std over {n_roll} eval episodes); RTG conditioning off"
            )
            _plot_reward(ax_r, r_sub)
            ax_r.set_xlabel("environment step")

        fig.suptitle(f"Eval step {step} · rollout dynamics", fontsize=12, y=1.02, color="#1a1a1a")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=120, bbox_inches="tight", facecolor=bg)
        plt.close(fig)
    return True


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
    if not isinstance(model, MetaDecisionTransformer):
        raise TypeError(f"eval expects MetaDecisionTransformer, got {type(model)}")
    cond_rtg = model._condition_rtg
    out: List[np.ndarray] = []
    for t, f in enumerate(frames):
        lines = [f"{trial_tag}  t={t}"]
        if cond_rtg and t < len(rtg_per_frame):
            # Same units as training: RTG token = future_return / rtg_scale (~O(1)).
            lines.append(f"RTG token: {rtg_per_frame[t]:.4f}")
        elif not cond_rtg:
            lines.append("RTG: off")
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

    if not isinstance(model, MetaDecisionTransformer):
        raise TypeError(f"eval expects MetaDecisionTransformer, got {type(model)}")
    vision_encoder = model.vision_encoder
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
    rtg_scale: float,
    max_episode_steps: int,
    step: int,
    ep: int,
    prompt: Optional[Tuple[Any, ...]],
    collect_frames: bool = False,
    env_name: str = "",
    prompt_builder: Optional[
        Callable[
            [List[np.ndarray], List[np.ndarray], List[float]],
            Tuple[Optional[Tuple[Any, ...]], Dict[str, Any]],
        ]
    ] = None,
    eval_target_return: Optional[float] = None,
    query_trial_index: int = 0,
    query_window: Optional[int] = None,
    reset_seed: Optional[int] = None,
) -> Tuple[
    float,
    int,
    np.ndarray,
    np.ndarray,
    Dict[str, np.ndarray],
    List[np.ndarray],
    List[float],
    np.ndarray,
]:
    """Run one episode. Returns frames and rtg_for_frames: RTG **token** at query tail before each
    ``get_action`` — same as training: ``discount_cumsum(r_env)/rtg_scale`` (~O(1)).

    Step rewards in the query sequence are **raw** env rewards. ``eval_target_return`` (optional) is
    target future cumulative return in env units; initial token = ``G / rtg_scale``. Default
    initial token is ``1.0`` (``initial_rtg_token``).
    """
    import torch

    if not isinstance(model, MetaDecisionTransformer):
        raise TypeError(f"eval expects MetaDecisionTransformer, got {type(model)}")

    rtg_for_frames: List[float] = []
    rs = int(reset_seed) if reset_seed is not None else int(step + ep)
    obs, _ = env.reset(seed=rs)
    if isinstance(obs, tuple):
        obs = obs[0]
    frames: List[np.ndarray] = []
    rtg0 = initial_rtg_token(rtg_scale, eval_target_return=eval_target_return)
    returns_to_go = torch.tensor([[rtg0]], device=device, dtype=torch.float32)
    if collect_frames:
        frame = _render_rgb_frame(env)
        if frame is not None:
            frames.append(frame)
            rtg_for_frames.append(float(returns_to_go[0, -1].detach().cpu()))
    use_vision = model.vision_encoder is not None
    image_list: List[Tuple[Any, Any]] = []
    if use_vision:
        prim, wrist = env.get_current_images()
        image_list.append((prim, wrist))
    context_dim = model.context_dim
    states = torch.from_numpy(obs).float().reshape(1, -1).to(device)
    contexts = torch.zeros(
        1, context_dim, device=device
    )  # get_action expands to len(states) if needed
    actions_t = torch.zeros(0, model.act_dim, device=device)
    rewards_t = torch.zeros(0, device=device)
    timesteps = torch.zeros(1, 1, dtype=torch.long, device=device)
    ep_states = [obs.copy()]
    ep_actions: List[np.ndarray] = []
    ep_rewards: List[float] = []
    ep_return = 0.0
    ml_q = model.max_length
    rtg_tokens_trace: List[float] = []

    for t in range(max_episode_steps):
        rtg_snap = float(returns_to_go[0, -1].detach().cpu())
        rtg_tokens_trace.append(rtg_snap)
        prompt_now = prompt
        prompt_meta: Dict[str, Any] = {"mode": "static", "prev_trials": 0, "current_trial_steps": 0}
        if prompt_builder is not None:
            prompt_now, prompt_meta = prompt_builder(ep_states, ep_actions, ep_rewards)
        qti = query_trial_index
        if prompt_meta.get("query_trial_index") is not None:
            qti = int(prompt_meta["query_trial_index"])
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
        ga_kw: Dict[str, Any] = dict(
            image_embeddings=image_embeddings,
            query_trial_index=qti,
        )
        if query_window is not None:
            ga_kw["query_window"] = int(query_window)

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
            **ga_kw,
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
        if use_vision:
            prim, wrist = env.get_current_images()
            image_list.append((prim, wrist))
        if collect_frames:
            frame = _render_rgb_frame(env)
            if frame is not None:
                frames.append(frame)
                rtg_for_frames.append(rtg_snap)
        if isinstance(next_obs, tuple):
            next_obs = next_obs[0]
        r_env = float(reward)
        ep_return += r_env
        ep_rewards.append(r_env)
        ep_states.append(next_obs.copy())
        ep_actions.append(action_np.copy())
        actions_t = torch.cat([actions_t, action], dim=0)
        rewards_t = torch.cat([rewards_t, torch.tensor([r_env], device=device)])
        states = torch.cat(
            [states, torch.from_numpy(next_obs).float().reshape(1, -1).to(device)], dim=0
        )
        returns_to_go = torch.cat(
            [returns_to_go, (returns_to_go[0, -1] - r_env / rtg_scale).reshape(1, 1)], dim=1
        )
        timesteps = torch.cat([timesteps, torch.tensor([[t + 1]], device=device)], dim=1)
        if done or truncated:
            break
    traj_dict = {
        "observations": np.array(ep_states, dtype=np.float32),
        "actions": np.array(ep_actions, dtype=np.float32),
        "rewards": np.array(ep_rewards, dtype=np.float32),
    }
    rtg_arr = np.asarray(rtg_tokens_trace, dtype=np.float64)
    return (
        ep_return,
        len(ep_states) - 1,
        np.array(ep_states),
        np.array(ep_actions),
        traj_dict,
        frames,
        rtg_for_frames,
        rtg_arr,
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
    rtg_scale: float = 5000.0,
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
    wandb_defer_step_commit: bool = False,
    vd4rl_eval_pixel_hw: Optional[int] = None,
    vd4rl_eval_obs_downsample: Optional[int] = None,
    vd4rl_eval_seed: int = 0,
    eval_target_return: Optional[float] = None,
    query_window: Optional[int] = None,
    minari_halfcheetah_dataset_id: Optional[str] = None,
    num_eval_rollout_videos: Optional[int] = None,
    d4rl_score_ref: Optional[Tuple[float, float]] = None,
) -> Dict[str, float]:
    """
    Run eval rollouts (and, in zero-shot, ``eval_num_trials`` in-session episodes per session).
    Each recorded rollout index can be saved as rollout_0.mp4, rollout_1.mp4, ...
    For wandb: log one continuous video with per-clip overlays (session/trial labels in zero-shot).
    When wandb_defer_step_commit=True (training eval), media uses commit=False so the trainer
    can log scalars at the same step with commit=True without W&B dropping out-of-order steps.

    **``rtg_scale``** must equal ``data.rtg_scale`` from training. RTG **tokens** are
    ``discount_cumsum(r_env) / rtg_scale`` (see ``dataset.py``). Each rollout step uses raw env
    reward in the query tensor and decrements the RTG token by ``r_env / rtg_scale``.

    **``eval_target_return``** (optional) is target future cumulative return in **env reward units**;
    initial token is ``G / rtg_scale``. If omitted, initial token is ``1.0``.

    **``query_window``** (optional): query history cap K passed to ``get_action`` (training
    ``query_history_length`` or ``horizon``). When set, overrides ``model.max_length`` for
    trim/left-pad so no-prompt / query-only eval matches the data window.

    **``minari_halfcheetah_dataset_id``** (optional): e.g. ``mujoco/halfcheetah/medium-v0`` —
    build the eval env with Minari ``recover_environment()`` (falls back to ``HalfCheetah-v5``).

    **``num_eval_rollout_videos``** (optional): max number of rollouts (or zero-shot trials) to
    record frames for. ``None`` = record all when ``save_video`` or W&B video is on. Metrics still
    use every rollout. W&B: one **combined** clip (``eval/rollout_video``) from recorded frames plus
    **per-rollout** clips (``eval/rollout_video_i``) when saved — combined for a quick scan,
    separate files for one rollout’s recording.

    **Zero-shot** (``eval_context_mode=zero_shot_adaptation``): ``num_rollouts`` is the number of
    independent adaptation **sessions** (context reset each session); ``eval_num_trials`` is still
    the sequential trials **within** each session.

    **``d4rl_score_ref``** (optional): ``(ref_min, ref_max)`` for D4RL-style normalized return
    (100 * (R - ref_min) / (ref_max - ref_min)) per rollout; logs ``eval/return_mean_normalized`` and
    ``eval/return_std_normalized``. Use for MuJoCo HalfCheetah (D4RL / Minari); not for VD4RL
    dm_control tasks (different reward scale).
    """
    from src.envs.libero_env import LIBERO_SUITES

    if not isinstance(model, MetaDecisionTransformer):
        raise TypeError(
            f"run_rollouts_and_save_viz expects MetaDecisionTransformer, got {type(model)}"
        )

    use_wandb_video = logger is not None and logger._wandb is not None
    # Several media logs share one W&B step; never commit=True on each video or the step advances N times.
    wb_commit_video = False
    wb_commit_charts = not wandb_defer_step_commit
    collect_frames = save_video or use_wandb_video
    cap_raw = num_eval_rollout_videos
    video_cap: Optional[int]
    if cap_raw is None:
        video_cap = None
    else:
        video_cap = max(0, int(cap_raw))

    def _collect_frames_for_index(idx: int) -> bool:
        if not collect_frames:
            return False
        if video_cap is None:
            return True
        return idx < video_cap

    need_pixel_env = collect_frames and (video_cap is None or video_cap > 0)
    # Gymnasium needs render_mode="rgb_array" at construction for env.render() to return pixels.
    gym_render_mode: Optional[str] = None
    if need_pixel_env and env_name not in LIBERO_SUITES:
        gym_render_mode = "rgb_array"

    env = _try_make_env(
        env_name,
        render_both_views=eval_render_both_views,
        render_mode=gym_render_mode,
        vd4rl_eval_pixel_hw=vd4rl_eval_pixel_hw,
        vd4rl_eval_obs_downsample=vd4rl_eval_obs_downsample,
        vd4rl_eval_seed=vd4rl_eval_seed,
        minari_halfcheetah_dataset_id=minari_halfcheetah_dataset_id,
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

    if eval_context_mode == "zero_shot_adaptation":
        print(
            f"Eval rollouts: zero_shot_adaptation, num_eval_rollouts={num_rollouts} (sessions), "
            f"eval_num_trials={eval_num_trials} (trials per session)",
            flush=True,
        )
    else:
        print(
            f"Eval rollouts: mode={eval_context_mode!r}, num_rollouts={num_rollouts}",
            flush=True,
        )

    viz_dir = run_dir / "viz" / "samples" / f"step_{step:06d}"
    viz_dir.mkdir(parents=True, exist_ok=True)
    video_folder = viz_dir / "videos"
    if collect_frames and need_pixel_env:
        video_folder.mkdir(parents=True, exist_ok=True)
        cap_msg = (
            "all rollouts/trials" if video_cap is None else f"first {video_cap} rollout(s)/trial(s)"
        )
        print(
            f"Eval rollout: saving videos for {cap_msg} under {video_folder.resolve()}",
            flush=True,
        )
    elif save_video:
        video_folder.mkdir(parents=True, exist_ok=True)
        env, _ = _wrap_record_video(env, video_folder)
        print(f"Eval rollout videos will be saved to: {video_folder.resolve()}")
    all_frames_for_wandb: List[np.ndarray] = []

    import torch

    state_mean_t = state_mean
    state_std_t = state_std
    if state_mean_t is None:
        state_mean_t = np.zeros(env.observation_space.shape[0])
    if state_std_t is None:
        state_std_t = np.ones(env.observation_space.shape[0])
    if isinstance(state_mean_t, torch.Tensor):
        state_mean_t = state_mean_t.detach().cpu().numpy()
    if isinstance(state_std_t, torch.Tensor):
        state_std_t = state_std_t.detach().cpu().numpy()
    state_mean_t = np.asarray(state_mean_t, dtype=np.float32)
    state_std_t = np.asarray(state_std_t, dtype=np.float32)

    rtg0 = initial_rtg_token(rtg_scale, eval_target_return=eval_target_return)
    tgt_g = float(eval_target_return) if eval_target_return is not None else float(rtg_scale)
    print(
        f"Eval RTG: rtg_scale={rtg_scale}; eval_target_return_G={tgt_g} -> initial_rtg_token={rtg0:.6g} "
        f"(per-step env rewards are raw; token -= r/rtg_scale).",
        flush=True,
    )

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
    all_reward_traces: List[np.ndarray] = []
    all_rtg_traces: List[np.ndarray] = []

    state_dim = model.state_dim
    act_dim = model.act_dim
    K = eval_context_k
    total_len = total_prompt_len or 512
    max_traj_len = max_prompt_trajectory_length  # None = use full trajectory per demo

    if eval_context_mode == "zero_shot_adaptation":
        n_trials = eval_num_trials
        ml_fallback = model.max_length if model.max_length is not None else 20
        current_trial_k = int(K) if K is not None else int(ml_fallback)
        ml_model = model.max_length
        cond_rtg = model._condition_rtg
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
        for rep in range(num_rollouts):
            context_list: List[Tuple[Dict[str, np.ndarray], float]] = []
            for trial in range(n_trials):
                global_idx = rep * n_trials + trial
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
                    "+".join(str(x) for x in prior_per_trial_steps)
                    if prior_per_trial_steps
                    else "0"
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
                        "query_trial_index": len(prev_trajs),
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
                        rtg_scale,
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
                    f"[zero_shot eval] --- session {rep + 1}/{num_rollouts} trial {trial + 1}/{n_trials} --- "
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
                    tag=(
                        f"zero_shot session {rep} trial {trial} / step {step} "
                        "(prompt length varies during rollout)"
                    ),
                    query_window=query_window,
                )
                ep_return, length, S, A, traj_dict, frames, rtg_ff, rtg_trace = _run_one_rollout(
                    model,
                    env,
                    state_mean_t,
                    state_std_t,
                    device,
                    rtg_scale,
                    max_episode_steps,
                    step,
                    trial,
                    None,
                    collect_frames=_collect_frames_for_index(global_idx),
                    env_name=env_name,
                    prompt_builder=_build_prompt_with_current,
                    eval_target_return=eval_target_return,
                    query_trial_index=trial,
                    query_window=query_window,
                    reset_seed=step + rep * 100_000 + trial,
                )
                ret = _return_for_traj(traj_dict, ep_return)
                context_list.append((traj_dict, ret))
                print(
                    f"[zero_shot eval] finished session={rep} trial={trial}: "
                    f"added full trial to prior context. prior_trials_now={len(context_list)}",
                    flush=True,
                )
                all_returns.append(ep_return)
                all_lengths.append(length)
                all_states_list.append(S)
                all_actions_list.append(A)
                all_reward_traces.append(np.asarray(traj_dict["rewards"], dtype=np.float64))
                all_rtg_traces.append(np.asarray(rtg_trace, dtype=np.float64))
                if _collect_frames_for_index(global_idx) and frames:
                    ann = _annotated_rollout_frames(
                        model,
                        frames,
                        rtg_ff,
                        f"S{rep + 1} T{trial + 1}",
                    )
                    _write_rollout_video(video_folder, global_idx, ann)
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
            rtg_scale,
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
            query_window=query_window,
        )
        for ep in range(num_rollouts):
            ep_return, length, S, A, traj_dict, frames, rtg_ff, rtg_trace = _run_one_rollout(
                model,
                env,
                state_mean_t,
                state_std_t,
                device,
                rtg_scale,
                max_episode_steps,
                step,
                ep,
                prompt,
                collect_frames=_collect_frames_for_index(ep),
                env_name=env_name,
                eval_target_return=eval_target_return,
                query_trial_index=len(prompt_trajectories),
                query_window=query_window,
            )
            all_returns.append(ep_return)
            all_lengths.append(length)
            all_states_list.append(S)
            all_actions_list.append(A)
            all_reward_traces.append(np.asarray(traj_dict["rewards"], dtype=np.float64))
            all_rtg_traces.append(np.asarray(rtg_trace, dtype=np.float64))
            if _collect_frames_for_index(ep) and frames:
                ann = _annotated_rollout_frames(model, frames, rtg_ff, f"Rollout {ep + 1}")
                _write_rollout_video(video_folder, ep, ann)
                all_frames_for_wandb.extend(ann)
    else:
        if num_rollouts > 0:
            _log_eval_transformer_seq(
                model,
                None,
                env_name=env_name,
                max_episode_steps=max_episode_steps,
                tag=f"no prompt step {step}",
                query_window=query_window,
            )
        for ep in range(num_rollouts):
            ep_return, length, S, A, traj_dict, frames, rtg_ff, rtg_trace = _run_one_rollout(
                model,
                env,
                state_mean_t,
                state_std_t,
                device,
                rtg_scale,
                max_episode_steps,
                step,
                ep,
                None,
                collect_frames=_collect_frames_for_index(ep),
                env_name=env_name,
                eval_target_return=eval_target_return,
                query_window=query_window,
            )
            all_returns.append(ep_return)
            all_lengths.append(length)
            all_states_list.append(S)
            all_actions_list.append(A)
            all_reward_traces.append(np.asarray(traj_dict["rewards"], dtype=np.float64))
            all_rtg_traces.append(np.asarray(rtg_trace, dtype=np.float64))
            if _collect_frames_for_index(ep) and frames:
                ann = _annotated_rollout_frames(model, frames, rtg_ff, f"Rollout {ep + 1}")
                _write_rollout_video(video_folder, ep, ann)
                all_frames_for_wandb.extend(ann)

    env.close()
    # Save combined video (all trials with labels) and each trial separately; then log to wandb
    if all_frames_for_wandb:
        _write_frames_video(video_folder, "all_rollouts.mp4", all_frames_for_wandb, fps=20)
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
                rollout_path = video_folder / f"rollout_{i}.mp4"
                if rollout_path.exists():
                    logger.log_video_from_path(
                        f"eval/rollout_video_{i}",
                        rollout_path,
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
        _ep_lab = "Episode" if eval_context_mode == "zero_shot_adaptation" else "Rollout"
        axes[0].set_title(f"{_ep_lab} {i} (return={all_returns[i]:.1f})")
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
    # Per-trial / grouped return plots only when K>1; for K==1 scalars (eval/return_mean, std) suffice.
    if len(all_returns) > 0 and int(eval_num_trials) > 1:
        arr = np.asarray(all_returns, dtype=np.float64)
        n_pts = int(arr.shape[0])
        is_zs = eval_context_mode == "zero_shot_adaptation"
        K = max(1, int(eval_num_trials))
        N = int(num_rollouts)
        grouped_zs = bool(is_zs and K >= 1 and N >= 1 and n_pts == N * K)

        with plt.rc_context({"font.family": "serif", "font.serif": _font["serif"]}):
            fig_sum, (ax_bar, ax_cum) = plt.subplots(
                1, 2, figsize=(9, 4.2), constrained_layout=True
            )

            if grouped_zs:
                R = arr.reshape(N, K)
                bar_mean = np.mean(R, axis=1)
                bar_std = np.std(R, axis=1, ddof=0)
                bar_x = np.arange(N, dtype=int)
                ax_bar.bar(
                    bar_x,
                    bar_mean,
                    yerr=bar_std,
                    capsize=4,
                    color="steelblue",
                    ecolor="#333333",
                    edgecolor="#1a5276",
                    linewidth=0.6,
                    error_kw={"linewidth": 1.2, "capthick": 1.2},
                )
                ax_bar.axhline(
                    float(np.mean(arr)),
                    color="coral",
                    linestyle="--",
                    linewidth=1.2,
                    label=f"global mean={float(np.mean(arr)):.1f}",
                )
                ax_bar.set_xlabel("rollout index")
                ax_bar.set_ylabel("return (env sum per episode)")
                ax_bar.set_title(f"Mean return ± std over K={K} in-session trials per rollout")
                suptitle_sub = (
                    f"{N} rollouts × {K} trials/rollout; bars aggregate within each rollout"
                )

                cum_mat = np.cumsum(R, axis=1)
                cum_x = np.arange(K, dtype=int)
                cum_mean = np.mean(cum_mat, axis=0)
                cum_std = np.std(cum_mat, axis=0, ddof=0)
                if K > 1:
                    ax_cum.fill_between(
                        cum_x,
                        cum_mean - cum_std,
                        cum_mean + cum_std,
                        color="steelblue",
                        alpha=0.28,
                        linewidth=0,
                    )
                ax_cum.plot(
                    cum_x,
                    cum_mean,
                    color="#1a5276",
                    linewidth=2.0,
                    marker="o",
                    markersize=5,
                    label="mean ± std across rollouts",
                )
                ax_cum.errorbar(
                    cum_x,
                    cum_mean,
                    yerr=cum_std,
                    fmt="none",
                    ecolor="#333333",
                    capsize=3,
                    elinewidth=1.0,
                    zorder=5,
                )
                ax_cum.set_xlabel(
                    "in-session trial index j (cumulative = trials 0…j within a rollout)"
                )
                ax_cum.set_ylabel("cumulative return")
                ax_cum.set_title("Within-rollout cumulative return: mean ± std across rollouts")
            else:
                if not is_zs and n_pts > 1:
                    m = float(np.mean(arr))
                    s = float(np.std(arr, ddof=0))
                    ax_bar.bar(
                        [0],
                        [m],
                        width=0.45,
                        yerr=[s],
                        capsize=5,
                        color="steelblue",
                        ecolor="#333333",
                        edgecolor="#1a5276",
                        linewidth=0.8,
                        error_kw={"linewidth": 1.2, "capthick": 1.2},
                    )
                    ax_bar.set_xticks([0])
                    ax_bar.set_xticklabels([f"{n_pts} rollouts"])
                    ax_bar.set_title(
                        "Mean return ± std across independent rollouts (1 episode each)"
                    )
                    ax_bar.set_xlabel("aggregate")
                    suptitle_sub = f"{n_pts} independent rollout(s), one bar = mean ± std"
                else:
                    bar_x = np.arange(n_pts, dtype=int)
                    ax_bar.bar(
                        bar_x,
                        arr,
                        color="steelblue",
                        edgecolor="#1a5276",
                        linewidth=0.6,
                    )
                    ax_bar.axhline(
                        float(np.mean(arr)),
                        color="coral",
                        linestyle="--",
                        linewidth=1.2,
                        label=f"mean={float(np.mean(arr)):.1f}",
                    )
                    if is_zs and n_pts != N * K:
                        ax_bar.set_title(
                            f"Return per episode (expected {N}×{K}={N * K} points, got {n_pts}; no grouped stats)"
                        )
                        suptitle_sub = (
                            f"{n_pts} episode(s); check num_eval_rollouts × eval_num_trials"
                        )
                    else:
                        ax_bar.set_title("Return per rollout (single episode per bar)")
                        suptitle_sub = f"{n_pts} rollout(s)" + (
                            f", K={K} trial(s)/rollout configured" if is_zs else ""
                        )
                    ax_bar.set_xlabel(
                        "rollout index (one episode per rollout)" if not is_zs else "episode index"
                    )
                ax_bar.set_ylabel("return (env sum per episode)")

                cum_x = np.arange(n_pts, dtype=int)
                cum_y = np.cumsum(arr)
                ax_cum.plot(
                    cum_x,
                    cum_y,
                    color="#1a5276",
                    linewidth=2.0,
                    marker="o",
                    markersize=4,
                    label="observed cumsum",
                )
                if not is_zs and n_pts > 1:
                    cum_mu = (np.arange(1, n_pts + 1, dtype=np.float64)) * m
                    ax_cum.plot(
                        cum_x,
                        cum_mu,
                        color="coral",
                        linestyle="--",
                        linewidth=1.2,
                        label="i × mean(return) (reference)",
                    )
                    ax_cum.set_title(
                        "Cumulative return over rollouts (one path; dashed = i × mean per rollout)"
                    )
                else:
                    ax_cum.set_title("Cumulative return (sequential sum in eval order)")
                ax_cum.set_xlabel(
                    "flattened episode index"
                    if is_zs
                    else "rollout index (running sum over rollouts)"
                )
                ax_cum.set_ylabel("cumulative return")

            ax_bar.xaxis.set_major_locator(MaxNLocator(integer=True))
            _handles, _labels = ax_bar.get_legend_handles_labels()
            if _labels:
                ax_bar.legend(loc="upper right", fontsize=8)
            ax_bar.grid(True, axis="y", alpha=0.3)

            ax_cum.xaxis.set_major_locator(MaxNLocator(integer=True))
            _h2, _l2 = ax_cum.get_legend_handles_labels()
            if _l2:
                ax_cum.legend(loc="lower right", fontsize=8, framealpha=0.92)
            ax_cum.grid(True, alpha=0.3)
            fig_sum.suptitle(f"Eval step {step}\n{suptitle_sub}", fontsize=10, y=1.05)

        summary_path = viz_dir / "eval_returns_summary.png"
        fig_sum.savefig(summary_path, dpi=100, bbox_inches="tight")
        plt.close(fig_sum)
        if logger is not None and logger._wandb is not None:
            import wandb

            p = str(summary_path)
            logger.log_wandb_dict(
                {"eval/returns_summary": wandb.Image(p)},
                step=step,
                wandb_commit=wb_commit_charts,
            )

    rtg_dyn_path = viz_dir / "eval_rtg_reward_dynamics.png"
    if _save_eval_rtg_reward_figure(
        all_reward_traces,
        all_rtg_traces,
        rtg_scale=rtg_scale,
        step=step,
        out_path=rtg_dyn_path,
        condition_rtg=bool(model._condition_rtg),
    ):
        if logger is not None and logger._wandb is not None:
            import wandb

            logger.log_wandb_dict(
                {"eval/rtg_reward_dynamics": wandb.Image(str(rtg_dyn_path))},
                step=step,
                wandb_commit=wb_commit_charts,
            )

    if len(all_returns) == 0:
        metrics = {
            "eval/return_mean": 0.0,
            "eval/return_std": 0.0,
            "eval/len_mean": 0.0,
        }
    else:
        metrics = {
            "eval/return_mean": float(np.mean(all_returns)),
            "eval/return_std": float(np.std(all_returns)),
            "eval/len_mean": float(np.mean(all_lengths)),
        }
        if d4rl_score_ref is not None:
            from src.envs.d4rl_normalized_score import d4rl_normalize_returns

            rmin, rmax = float(d4rl_score_ref[0]), float(d4rl_score_ref[1])
            norm = d4rl_normalize_returns(all_returns, rmin, rmax)
            metrics["eval/return_mean_normalized"] = float(np.mean(norm))
            metrics["eval/return_std_normalized"] = float(np.std(norm))
    ap_stats = _action_prediction_stats_from_rollouts(all_actions_list)
    metrics.update(ap_stats)
    if ap_stats:
        print(
            f"[eval action_pred] step={step} mean={ap_stats['eval/action_pred_mean']:.4f} "
            f"min={ap_stats['eval/action_pred_min']:.4f} max={ap_stats['eval/action_pred_max']:.4f}",
            flush=True,
        )
    return metrics
