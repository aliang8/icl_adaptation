"""Eval rollouts: env episodes, logging, and sample viz under run_dir/viz/."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.data.rtg import initial_rtg_token
from src.engine.eval_context import build_prompt_tuple
from src.engine.eval_visuals import (
    annotated_rollout_frames,
    cum_return_per_frame,
    encode_rollout_images,
    raise_missing_eval_vision_images,
    save_eval_rtg_reward_figure,
)
from src.engine.reward_models import get_return_from_reward_model
from src.envs.eval_gym import render_rgb_frame, try_make_eval_env, wrap_record_video
from src.envs.libero_env import LIBERO_SUITES
from src.models.meta_dt import MetaDecisionTransformer
from src.utils.eval_utils import (
    action_prediction_stats_from_rollouts,
    compose_grid_frames_sequence,
    default_eval_scene_seeds,
    eval_episode_reset_seed,
    grid_layout_dims,
    pack_flat_clips_to_grid,
    resolve_per_trial_eval_target_returns,
    write_frames_video,
)


def _log_eval_transformer_seq(
    model: Any,
    prompt: Optional[Tuple[Any, ...]],
    *,
    env_name: str,
    max_episode_steps: int,
    tag: str,
    query_window: Optional[int] = None,
) -> None:
    if not isinstance(model, MetaDecisionTransformer):
        raise TypeError(f"eval expects MetaDecisionTransformer, got {type(model)}")
    layout = model._sequence_token_layout
    tokens_per_step = 3 if layout in ("rtg_state_action", "state_action_reward") else 2
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


def _finalize_eval_rollout_videos(
    logger: Optional[Any],
    video_folder: Path,
    *,
    step: int,
    fps: int,
    wandb_commit: bool,
    clips_grid: List[List[Optional[List[np.ndarray]]]],
    n_rows: int,
    n_cols: int,
) -> None:
    grid_frames = compose_grid_frames_sequence(clips_grid, n_rows=n_rows, n_cols=n_cols)
    if grid_frames:
        write_frames_video(video_folder, "all_rollouts.mp4", grid_frames, fps=fps)
    if logger is None or getattr(logger, "_wandb", None) is None:
        return
    if not grid_frames:
        return
    mp4_path = video_folder / "all_rollouts.mp4"
    # File-based Video avoids extra client-side encoding passes that can show up as duplicate rows.
    if mp4_path.is_file() and hasattr(logger, "log_video_from_path"):
        logger.log_video_from_path(
            "eval/rollout_video_grid",
            mp4_path,
            step=step,
            wandb_commit=wandb_commit,
        )
    elif hasattr(logger, "log_video"):
        logger.log_video(
            "eval/rollout_video_grid",
            grid_frames,
            step=step,
            fps=fps,
            wandb_commit=wandb_commit,
        )


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
    query_trial_index: int = 1,
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
    List[float],
]:
    """Run one episode. Returns ``rtg_for_frames`` (RTG **token** before each ``get_action``) and
    ``cum_return_for_frames`` (cumulative env return after each rendered step, same units as rewards).

    RTG tokens match training: ``future_return/rtg_scale``. Step rewards are raw env rewards.
    """
    if not isinstance(model, MetaDecisionTransformer):
        raise TypeError(f"eval expects MetaDecisionTransformer, got {type(model)}")

    rtg_for_frames: List[float] = []
    rs = int(reset_seed) if reset_seed is not None else int(step + ep)
    obs, _ = env.reset(seed=rs)
    if isinstance(obs, tuple):
        obs = obs[0]
    frames: List[np.ndarray] = []
    layout = model._sequence_token_layout
    use_rtg_rollout = layout == "rtg_state_action"
    if use_rtg_rollout:
        rtg0 = initial_rtg_token(rtg_scale, eval_target_return=eval_target_return)
        returns_to_go = torch.tensor([[rtg0]], device=device, dtype=torch.float32)
    else:
        returns_to_go = torch.zeros((1, 1, 1), device=device, dtype=torch.float32)
    if collect_frames:
        frame = render_rgb_frame(env)
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
    rewards_t = torch.zeros(0, device=device, dtype=torch.float32)
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

        # if prompt_now is not None:
        #     import ipdb; ipdb.set_trace()

        if t % 100 == 0 or t < 3:
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
        if use_vision:
            if image_list:
                image_embeddings = encode_rollout_images(image_list, model, device)
            if image_embeddings is None:
                raise_missing_eval_vision_images(
                    model=model,
                    env_name=env_name,
                    env=env,
                    timestep=t,
                )
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
            frame = render_rgb_frame(env)
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
        if use_rtg_rollout:
            returns_to_go = torch.cat(
                [returns_to_go, (returns_to_go[0, -1] - r_env / rtg_scale).reshape(1, 1)], dim=1
            )
        else:
            returns_to_go = torch.cat(
                [returns_to_go, torch.zeros(1, 1, 1, device=device, dtype=torch.float32)], dim=1
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
    cum_return_for_frames = cum_return_per_frame(frames, ep_rewards)
    return (
        ep_return,
        len(ep_states) - 1,
        np.array(ep_states),
        np.array(ep_actions),
        traj_dict,
        frames,
        rtg_for_frames,
        rtg_arr,
        cum_return_for_frames,
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
    context_style: str = "subsampled",
    task_description: Optional[str] = None,
    logger: Optional[Any] = None,
    eval_render_both_views: bool = True,
    wandb_defer_step_commit: bool = False,
    vd4rl_eval_pixel_hw: Optional[int] = None,
    vd4rl_eval_obs_downsample: Optional[int] = None,
    vd4rl_eval_seed: int = 0,
    eval_target_return: Optional[float] = None,
    eval_target_returns: Optional[List[float]] = None,
    num_context_trajectories: Optional[int] = None,
    query_window: Optional[int] = None,
    minari_halfcheetah_dataset_id: Optional[str] = None,
    num_eval_rollout_videos: Optional[int] = None,
    d4rl_score_ref: Optional[Tuple[float, float]] = None,
    maniskill_sim_backend: Optional[str] = None,
    maniskill_reward_mode: Optional[str] = None,
    maniskill_control_mode: Optional[str] = None,
    maniskill_state_obs_slice: Optional[slice] = None,
    eval_scene_seeds: Optional[List[int]] = None,
    randomize_scene_between_trials: bool = False,
) -> Dict[str, float]:
    """Run rollouts, write ``run_dir/viz/samples/step_*/`` plots, optional composite video/W&B.

    Keep ``rtg_scale``, ``context_style``, and prompt/query caps aligned with training config.
    Zero-shot: ``num_rollouts`` sessions × ``eval_num_trials`` episodes; trial 1 is query-only, later
    trials use ICL from completed priors. ``num_context_trajectories`` ≤ 0 or ``None`` keeps all priors.
    ``wandb_defer_step_commit``: log charts/videos with ``commit=False`` so the trainer can commit once.
    """
    if not isinstance(model, MetaDecisionTransformer):
        raise TypeError(
            f"run_rollouts_and_save_viz expects MetaDecisionTransformer, got {type(model)}"
        )

    _t_rollout_start = time.perf_counter()

    resolved_scene_seeds: List[int]
    if eval_scene_seeds:
        resolved_scene_seeds = [int(x) for x in eval_scene_seeds if x is not None]
    else:
        resolved_scene_seeds = []
    eval_scene_seeds_explicit = len(resolved_scene_seeds) > 0
    if not resolved_scene_seeds:
        resolved_scene_seeds = default_eval_scene_seeds(
            eval_context_mode=eval_context_mode,
            num_rollouts=num_rollouts,
            eval_num_trials=eval_num_trials,
            randomize_scene_between_trials=randomize_scene_between_trials,
            seed_base=int(vd4rl_eval_seed),
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
    # Gymnasium / ManiSkill need render_mode="rgb_array" at construction for env.render() pixels.
    gym_render_mode: Optional[str] = None
    need_maniskill_vision_render = model.vision_encoder is not None and str(env_name).startswith(
        "ManiSkill/"
    )
    if env_name not in LIBERO_SUITES and (need_pixel_env or need_maniskill_vision_render):
        gym_render_mode = "rgb_array"

    env = try_make_eval_env(
        env_name,
        render_both_views=eval_render_both_views,
        render_mode=gym_render_mode,
        vd4rl_eval_pixel_hw=vd4rl_eval_pixel_hw,
        vd4rl_eval_obs_downsample=vd4rl_eval_obs_downsample,
        vd4rl_eval_seed=vd4rl_eval_seed,
        minari_halfcheetah_dataset_id=minari_halfcheetah_dataset_id,
        maniskill_sim_backend=maniskill_sim_backend,
        maniskill_reward_mode=maniskill_reward_mode,
        maniskill_control_mode=maniskill_control_mode,
        maniskill_state_obs_slice=maniskill_state_obs_slice,
    )
    if env is None:
        if env_name in LIBERO_SUITES:
            print(
                f"Eval rollouts skipped: LIBERO not installed for '{env_name}'. "
                "Install with: uv sync --extra libero (or pip install -e .[libero])."
            )
        else:
            print(f"Eval rollouts skipped: no env registered for '{env_name}'.")
        _wall = time.perf_counter() - _t_rollout_start
        print(
            f"[eval rollout] timing: wall={_wall:.3f}s (no env; skipped)",
            flush=True,
        )
        return {
            "eval/return_mean": 0.0,
            "eval/rollout_wall_s": float(_wall),
        }

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
    print(
        f"[eval] scene reset: seed_pool_len={len(resolved_scene_seeds)} "
        f"source={'config' if eval_scene_seeds_explicit else 'auto'} "
        f"randomize_scene_between_trials={randomize_scene_between_trials}",
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
        env, _ = wrap_record_video(env, video_folder)
        print(f"Eval rollout videos will be saved to: {video_folder.resolve()}")

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

    per_trial_eval_G_zs: Optional[List[Optional[float]]] = None
    _lay_m = model._sequence_token_layout
    if _lay_m == "state_action_reward":
        print(
            "Eval: state_action_reward layout — (s,a,r) policy; eval_target_return / RTG rollouts unused.",
            flush=True,
        )
    elif eval_context_mode == "zero_shot_adaptation" and int(eval_num_trials) > 0:
        per_trial_eval_G_zs = resolve_per_trial_eval_target_returns(
            int(eval_num_trials), eval_target_return, eval_target_returns
        )
        _toks = [initial_rtg_token(rtg_scale, eval_target_return=g) for g in per_trial_eval_G_zs]
        print(
            f"[zero_shot] per-trial eval target G={per_trial_eval_G_zs} -> initial_rtg_token={_toks} "
            f"rtg_scale={rtg_scale} (env step rewards raw; token -= r/rtg_scale).",
            flush=True,
        )
    else:
        rtg0 = initial_rtg_token(rtg_scale, eval_target_return=eval_target_return)
        tgt_g = float(eval_target_return) if eval_target_return is not None else float(rtg_scale)
        print(
            f"Eval RTG: rtg_scale={rtg_scale}; eval_target_return_G={tgt_g} -> initial_rtg_token={rtg0:.6g} "
            f"(per-step env rewards are raw; token -= r/rtg_scale).",
            flush=True,
        )

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
        _lay = model._sequence_token_layout
        tps = 3 if _lay in ("rtg_state_action", "state_action_reward") else 2
        ctx_cap: Optional[int] = None
        if num_context_trajectories is not None:
            ctx_cap = max(0, int(num_context_trajectories))
        ctx_desc = (
            f"up to {ctx_cap} most recent completed trials (then sorted asc. by return)"
            if ctx_cap is not None and ctx_cap > 0
            else "all completed prior trials (sorted asc. by return)"
        )
        if _lay == "state_action_reward":
            print(
                f"[zero_shot eval] Algorithm distillation: trials 2+ use ``build_prompt_tuple`` prior demos; "
                f"``MetaDecisionTransformer.forward`` concatenates prior demos then query on the time axis "
                f"and runs one ``embed_*`` pass (same as training). Live episode stays in the query; last {current_trial_k} "
                f"steps after concat come from ``get_action`` (``query_window`` / horizon). "
                f"Sorted by return: [{ctx_desc}]. max_episode_steps={max_episode_steps} "
                f"query_pad_max_length={ml_model} tokens_per_dt_step={tps}.",
                flush=True,
            )
        else:
            print(
                f"[zero_shot eval] Policy: trial 1 = query-only; trials 2+ = ICL prompt from [{ctx_desc}] "
                f"only (fixed during the episode). Query segment: last {current_trial_k} env steps via get_action. "
                f"total_prompt_cap={total_len} max_prompt_traj_len={max_traj_len} "
                f"context_style={context_style} max_episode_steps={max_episode_steps} "
                f"context_subsample_strategy={context_subsample_strategy} "
                f"query_pad_max_length={ml_model} tokens_per_dt_step={tps}. "
                f"ICL prompt layout matches training (dataset padding for this context_style).",
                flush=True,
            )
        zs_clip_grid: List[List[Optional[List[np.ndarray]]]] = [
            [None] * n_trials for _ in range(num_rollouts)
        ]
        for rep in range(num_rollouts):
            context_list: List[Tuple[Dict[str, np.ndarray], float]] = []
            for trial in range(n_trials):
                global_idx = rep * n_trials + trial
                _zs_prompt_cache: Optional[Tuple[Optional[Tuple[Any, ...]], Dict[str, Any]]] = None
                if ctx_cap is None:
                    recent = list(context_list)
                elif ctx_cap <= 0:
                    # ``data.num_context_trajectories=0`` (common for AD) must not zero out eval priors:
                    # otherwise ``prev_trajs`` is always empty and ``prompt_now`` stays None for trial 1+.
                    recent = list(context_list)
                else:
                    recent = (
                        context_list[-ctx_cap:]
                        if len(context_list) > ctx_cap
                        else list(context_list)
                    )
                sorted_recent = sorted(recent, key=lambda x: x[1])
                prev_trajs = [t for t, _ in sorted_recent]
                prev_returns = [r for _, r in sorted_recent]

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

                def _build_prompt_from_prior_only(
                    _ep_states_now: List[np.ndarray],
                    _ep_actions_now: List[np.ndarray],
                    _ep_rewards_now: List[float],
                ) -> Tuple[Optional[Tuple[Any, ...]], Dict[str, Any]]:
                    nonlocal _zs_prompt_cache
                    # Trial 0: query only. Trial 1+: fixed ICL prompt from completed trials only —
                    # current episode stays in the query segment (grows in get_action), not in the prompt.
                    if trial == 0:
                        return None, {
                            "mode": "zero_shot_dynamic",
                            "prev_trials": 0,
                            "prior_context_steps": 0,
                            "prior_per_trial_steps": [],
                            "current_trial_window_steps": 0,
                            "current_trial_steps": 0,
                            "raw_concat_steps": 0,
                            "total_prompt_cap": total_len,
                            "query_trial_index": 1,
                        }
                    if _zs_prompt_cache is not None:
                        return _zs_prompt_cache
                    trajs = list(prev_trajs)
                    rets = list(prev_returns)
                    meta: Dict[str, Any] = {
                        "mode": "zero_shot_dynamic",
                        "prev_trials": len(prev_trajs),
                        "prior_context_steps": prior_ctx_steps,
                        "prior_per_trial_steps": list(prior_per_trial_steps),
                        "current_trial_window_steps": 0,
                        "current_trial_steps": 0,
                        "raw_concat_steps": prior_ctx_steps,
                        "total_prompt_cap": total_len,
                        "query_trial_index": len(prev_trajs) + 1,
                    }
                    if not trajs or not total_len:
                        _zs_prompt_cache = (None, meta)
                        return _zs_prompt_cache
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
                        context_style=context_style,
                        max_episode_steps=max_episode_steps,
                    )
                    pt = int(prompt_dyn[0].shape[1])
                    meta["prompt_timesteps_in_model"] = pt
                    ml = int(ml_model) if ml_model is not None else 0
                    meta["approx_transformer_positions"] = tps * (pt + ml)
                    _zs_prompt_cache = (prompt_dyn, meta)
                    return _zs_prompt_cache

                print(
                    f"[zero_shot eval] --- session {rep + 1}/{num_rollouts} trial {trial + 1}/{n_trials} --- "
                    f"prior_completed_trials={len(prev_trajs)} "
                    f"prior_steps_by_trial(capped)=[{prior_steps_str}] "
                    f"prior_context_steps(sum)={prior_ctx_steps} "
                    f"query_history_K={current_trial_k}",
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
                _qti_rollout = 1 if _lay == "state_action_reward" and trial == 0 else trial
                ep_return, length, S, A, traj_dict, frames, rtg_ff, rtg_trace, cum_ret_ff = (
                    _run_one_rollout(
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
                        prompt_builder=_build_prompt_from_prior_only,
                        eval_target_return=(
                            per_trial_eval_G_zs[trial]
                            if per_trial_eval_G_zs is not None
                            else eval_target_return
                        ),
                        query_trial_index=_qti_rollout,
                        query_window=query_window,
                        reset_seed=eval_episode_reset_seed(
                            step=step,
                            session_rep=rep,
                            trial=trial,
                            n_trials_in_session=n_trials,
                            eval_scene_seeds=resolved_scene_seeds,
                            randomize_scene_between_trials=randomize_scene_between_trials,
                        ),
                    )
                )
                ret = _return_for_traj(traj_dict, ep_return)
                context_list.append((traj_dict, ret))
                print(
                    f"[zero_shot eval] finished session={rep} trial={trial}: "
                    f"added full trial to prior context. prior_trials_now={len(context_list)} "
                    f"(context prompt uses up to {ctx_cap if ctx_cap is not None else 'all'} recent)",
                    flush=True,
                )
                all_returns.append(ep_return)
                all_lengths.append(length)
                all_states_list.append(S)
                all_actions_list.append(A)
                all_reward_traces.append(np.asarray(traj_dict["rewards"], dtype=np.float64))
                all_rtg_traces.append(np.asarray(rtg_trace, dtype=np.float64))
                if _collect_frames_for_index(global_idx) and frames:
                    ann = annotated_rollout_frames(
                        model,
                        frames,
                        rtg_ff,
                        f"S{rep + 1} T{trial + 1}",
                        cum_return_per_frame_vals=cum_ret_ff,
                    )
                    zs_clip_grid[rep][trial] = ann
        if collect_frames and any(
            zs_clip_grid[r][c] is not None for r in range(num_rollouts) for c in range(n_trials)
        ):
            if n_trials > 1:
                _finalize_eval_rollout_videos(
                    logger,
                    video_folder,
                    step=step,
                    fps=20,
                    wandb_commit=wb_commit_video,
                    clips_grid=zs_clip_grid,
                    n_rows=num_rollouts,
                    n_cols=n_trials,
                )
            else:
                zs_flat = [zs_clip_grid[r][0] for r in range(num_rollouts)]
                zr, zc = grid_layout_dims(num_rollouts, 1)
                zcg = pack_flat_clips_to_grid(zs_flat, zr, zc)
                _finalize_eval_rollout_videos(
                    logger,
                    video_folder,
                    step=step,
                    fps=20,
                    wandb_commit=wb_commit_video,
                    clips_grid=zcg,
                    n_rows=zr,
                    n_cols=zc,
                )
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
            context_style=context_style,
            max_episode_steps=max_episode_steps,
        )
        _log_eval_transformer_seq(
            model,
            prompt,
            env_name=env_name,
            max_episode_steps=max_episode_steps,
            tag=f"prompt mode step {step}",
            query_window=query_window,
        )
        prompt_clips: List[Optional[List[np.ndarray]]] = [None] * num_rollouts
        for ep in range(num_rollouts):
            ep_return, length, S, A, traj_dict, frames, rtg_ff, rtg_trace, cum_ret_ff = (
                _run_one_rollout(
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
                    query_trial_index=len(prompt_trajectories) + 1,
                    query_window=query_window,
                    reset_seed=eval_episode_reset_seed(
                        step=step,
                        session_rep=ep,
                        trial=0,
                        n_trials_in_session=1,
                        eval_scene_seeds=resolved_scene_seeds,
                        randomize_scene_between_trials=randomize_scene_between_trials,
                    ),
                )
            )
            all_returns.append(ep_return)
            all_lengths.append(length)
            all_states_list.append(S)
            all_actions_list.append(A)
            all_reward_traces.append(np.asarray(traj_dict["rewards"], dtype=np.float64))
            all_rtg_traces.append(np.asarray(rtg_trace, dtype=np.float64))
            if _collect_frames_for_index(ep) and frames:
                ann = annotated_rollout_frames(
                    model,
                    frames,
                    rtg_ff,
                    f"Rollout {ep + 1}",
                    cum_return_per_frame_vals=cum_ret_ff,
                )
                prompt_clips[ep] = ann
        if collect_frames and any(c is not None for c in prompt_clips):
            pr, pc = grid_layout_dims(num_rollouts, 1)
            pcg = pack_flat_clips_to_grid(prompt_clips, pr, pc)
            _finalize_eval_rollout_videos(
                logger,
                video_folder,
                step=step,
                fps=20,
                wandb_commit=wb_commit_video,
                clips_grid=pcg,
                n_rows=pr,
                n_cols=pc,
            )
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
        bare_clips: List[Optional[List[np.ndarray]]] = [None] * num_rollouts
        for ep in range(num_rollouts):
            ep_return, length, S, A, traj_dict, frames, rtg_ff, rtg_trace, cum_ret_ff = (
                _run_one_rollout(
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
                    reset_seed=eval_episode_reset_seed(
                        step=step,
                        session_rep=ep,
                        trial=0,
                        n_trials_in_session=1,
                        eval_scene_seeds=resolved_scene_seeds,
                        randomize_scene_between_trials=randomize_scene_between_trials,
                    ),
                )
            )
            all_returns.append(ep_return)
            all_lengths.append(length)
            all_states_list.append(S)
            all_actions_list.append(A)
            all_reward_traces.append(np.asarray(traj_dict["rewards"], dtype=np.float64))
            all_rtg_traces.append(np.asarray(rtg_trace, dtype=np.float64))
            if _collect_frames_for_index(ep) and frames:
                ann = annotated_rollout_frames(
                    model,
                    frames,
                    rtg_ff,
                    f"Rollout {ep + 1}",
                    cum_return_per_frame_vals=cum_ret_ff,
                )
                bare_clips[ep] = ann
        if collect_frames and any(c is not None for c in bare_clips):
            br, bc = grid_layout_dims(num_rollouts, 1)
            bcg = pack_flat_clips_to_grid(bare_clips, br, bc)
            _finalize_eval_rollout_videos(
                logger,
                video_folder,
                step=step,
                fps=20,
                wandb_commit=wb_commit_video,
                clips_grid=bcg,
                n_rows=br,
                n_cols=bc,
            )

    env.close()

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
    wb_eval_charts: Dict[str, Any] = {}
    # Per-trial / grouped return plots only when K>1; for K==1 scalars (eval/return_mean, std) suffice.
    if len(all_returns) > 0 and int(eval_num_trials) > 1:
        arr = np.asarray(all_returns, dtype=np.float64)
        n_pts = int(arr.shape[0])
        is_zs = eval_context_mode == "zero_shot_adaptation"
        K = max(1, int(eval_num_trials))
        N = int(num_rollouts)
        grouped_zs = bool(is_zs and K >= 1 and N >= 1 and n_pts == N * K)

        with plt.rc_context({"font.family": "serif", "font.serif": _font["serif"]}):
            fig_sum, (ax_per, ax_cum) = plt.subplots(
                1,
                2,
                figsize=(10, 4.2),
                sharex=True,
                constrained_layout=True,
            )

            if grouped_zs:
                R = arr.reshape(N, K)
                suptitle_sub = f"{N} rollouts × {K} trials/rollout · mean return & cum. return ± std across rollouts"

                cum_x = np.arange(K, dtype=int)
                per_mean = np.mean(R, axis=0)
                per_std = np.std(R, axis=0, ddof=0)
                ax_per.plot(
                    cum_x,
                    per_mean,
                    color="#2874a6",
                    linewidth=2.0,
                    marker="s",
                    markersize=5,
                )
                ax_per.errorbar(
                    cum_x,
                    per_mean,
                    yerr=per_std,
                    fmt="none",
                    ecolor="#333333",
                    capsize=3,
                    elinewidth=1.0,
                    zorder=5,
                )
                ax_per.set_ylabel("return (per trial)")
                ax_per.grid(True, alpha=0.3)

                cum_mat = np.cumsum(R, axis=1)
                cum_mean = np.mean(cum_mat, axis=0)
                cum_std = np.std(cum_mat, axis=0, ddof=0)
                ax_cum.plot(
                    cum_x,
                    cum_mean,
                    color="#1a5276",
                    linewidth=2.0,
                    marker="o",
                    markersize=5,
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
                ax_per.set_xlabel("trial index")
                ax_cum.set_xlabel("trial index")
                ax_cum.set_ylabel("cumulative return")
            else:
                if not is_zs and n_pts > 1:
                    m = float(np.mean(arr))
                    suptitle_sub = (
                        f"{n_pts} independent rollout(s) · per-rollout return & running cumulative"
                    )
                else:
                    if is_zs and n_pts != N * K:
                        ax_cum.set_title(
                            f"Expected {N}×{K}={N * K} points, got {n_pts}; no grouped stats"
                        )
                        suptitle_sub = (
                            f"{n_pts} episode(s); check num_eval_rollouts × eval_num_trials"
                        )
                    else:
                        suptitle_sub = f"{n_pts} rollout(s)" + (
                            f", K={K} trial(s)/rollout configured" if is_zs else ""
                        )

                cum_x = np.arange(n_pts, dtype=int)
                ax_per.plot(
                    cum_x,
                    arr,
                    color="#2874a6",
                    linewidth=2.0,
                    marker="s",
                    markersize=4,
                    label="return",
                )
                ax_per.set_ylabel("return (per episode)")
                ax_per.grid(True, alpha=0.3)

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
                    )
                _xlab = (
                    "flattened episode index"
                    if is_zs
                    else "rollout index (running sum over rollouts)"
                )
                ax_per.set_xlabel(_xlab)
                ax_cum.set_xlabel(_xlab)
                ax_cum.set_ylabel("cumulative return")

            for ax in (ax_per, ax_cum):
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            _h2, _l2 = ax_cum.get_legend_handles_labels()
            if _l2:
                ax_cum.legend(loc="lower right", fontsize=8, framealpha=0.92)
            _h3, _l3 = ax_per.get_legend_handles_labels()
            if _l3:
                ax_per.legend(loc="lower right", fontsize=8, framealpha=0.92)
            ax_cum.grid(True, alpha=0.3)
            fig_sum.suptitle(f"Eval step {step}\n{suptitle_sub}", fontsize=10, y=1.05)

        summary_path = viz_dir / "eval_returns_summary.png"
        fig_sum.savefig(summary_path, dpi=100, bbox_inches="tight")
        plt.close(fig_sum)
        if logger is not None and logger._wandb is not None:
            import wandb

            wb_eval_charts["eval/returns_summary"] = wandb.Image(str(summary_path))

    rtg_dyn_path = viz_dir / "eval_rtg_reward_dynamics.png"
    rtg_dyn_paths = save_eval_rtg_reward_figure(
        all_reward_traces,
        all_rtg_traces,
        rtg_scale=rtg_scale,
        step=step,
        out_path=rtg_dyn_path,
        condition_rtg=bool(model._condition_rtg),
        eval_num_trials=int(eval_num_trials),
        num_rollouts=int(num_rollouts),
        eval_context_mode=str(eval_context_mode),
    )
    # W&B: RTG/reward dynamics figure (disabled — duplicate Media panels; PNGs still under viz/).
    # if rtg_dyn_paths and logger is not None and logger._wandb is not None:
    #     import wandb
    #
    #     wb_rtg_path: Path
    #     wb_rtg_path = Path(rtg_dyn_paths[-1])
    #     for p in rtg_dyn_paths:
    #         if "_mean_std" in Path(p).stem:
    #             wb_rtg_path = Path(p)
    #             break
    #     wb_eval_charts["eval/rtg_reward_dynamics"] = wandb.Image(str(wb_rtg_path))

    if wb_eval_charts and logger is not None and logger._wandb is not None:
        payload: Dict[str, Any] = {"train/global_step": int(step), **wb_eval_charts}
        logger.log_wandb_dict(payload, step=step, wandb_commit=wb_commit_charts)

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
    ap_stats = action_prediction_stats_from_rollouts(all_actions_list)
    metrics.update(ap_stats)
    if ap_stats:
        print(
            f"[eval action_pred] step={step} mean={ap_stats['eval/action_pred_mean']:.4f} "
            f"min={ap_stats['eval/action_pred_min']:.4f} max={ap_stats['eval/action_pred_max']:.4f}",
            flush=True,
        )

    _wall = time.perf_counter() - _t_rollout_start
    metrics["eval/rollout_wall_s"] = float(_wall)
    _total_env_steps = int(sum(all_lengths)) if all_lengths else 0
    if _wall > 1e-9:
        metrics["eval/rollout_env_steps_per_s"] = float(_total_env_steps) / float(_wall)
    else:
        metrics["eval/rollout_env_steps_per_s"] = 0.0
    print(
        f"[eval rollout] timing: wall={_wall:.3f}s env_steps={_total_env_steps} "
        f"({metrics['eval/rollout_env_steps_per_s']:.1f} env steps/s)",
        flush=True,
    )
    return metrics
