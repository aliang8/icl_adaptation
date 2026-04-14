"""
ManiSkill-specific helpers for ICL trajectory export and rollouts.

The shared **replay buffer** on-disk format (HDF5 read/write/append, PPO buffer stitching, RGB batching)
is in :mod:`src.data.ic_replay_buffer_hdf5`. This module adds:

- :func:`resolve_maniskill_trajectory_paths` → resolves ``data.trajectory_hdf5_paths`` under
  ``data_root/maniskill/<task>/`` (no automatic shard discovery; list must be set in config);
- Gymnasium / ManiSkill ``final_info`` → ``episode_meta`` helpers;
- :func:`collect_episodes_vector_env` for vectorized ManiSkill rollouts.

Replay buffer I/O lives in :mod:`src.data.ic_replay_buffer_hdf5`; this module is ManiSkill conveniences only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from src.data.ic_replay_buffer_hdf5 import (
    finalize_trajectory_dict,
    render_batch_to_rgb_list,
)


def resolve_maniskill_trajectory_paths(
    data_root: Union[str, Path],
    env_id: str,
    trajectory_hdf5_paths: Optional[Sequence[str]] = None,
    *,
    repo_root: Optional[Path] = None,
) -> List[Path]:
    """
    Flat v2 ``.h5`` paths for a ManiSkill task id (e.g. ``PickCube-v1``).

    ``trajectory_hdf5_paths`` must be a non-empty list (e.g. from ``data.trajectory_hdf5_paths`` in
    YAML). Each entry is resolved with :func:`src.data.ic_replay_buffer_hdf5.resolve_trajectory_hdf5_path_entries`
    (absolute path, ``cwd``, ``data_root / entry``, or under ``<data_root>/maniskill/<env_id>/``).
    If ``repo_root`` is set, also tries ``<repo_root>/datasets/<entry>`` so local shards under the
    repo ``datasets/`` tree are found when ``paths.data_root`` points elsewhere (e.g. shared NFS).
    """
    root = Path(data_root).expanduser().resolve()
    safe = env_id.replace("/", "_").replace(" ", "_")
    task_dir = root / "maniskill" / safe
    if not trajectory_hdf5_paths:
        raise ValueError(
            "ManiSkill training requires a non-empty data.trajectory_hdf5_paths list "
            f"(flat v2 .h5 shards). Example relative to paths.data_root:\n"
            f"  - maniskill/{safe}/trajectories_shard_00000.h5\n"
            f"  - maniskill/{safe}/trajectories_shard_00001.h5\n"
            f"  ...\n"
            f"Expected data under: {task_dir}"
        )
    from src.data.ic_replay_buffer_hdf5 import resolve_trajectory_hdf5_path_entries

    extras: tuple[Path, ...] = ()
    if repo_root is not None:
        extras = (Path(repo_root).expanduser().resolve() / "datasets",)
    return resolve_trajectory_hdf5_path_entries(
        trajectory_hdf5_paths,
        data_root=root,
        search_dirs=(task_dir, root),
        extra_roots=extras,
    )


def resolve_maniskill_trajectory_path(
    data_root: Union[str, Path],
    env_id: str,
    trajectory_hdf5_paths: Sequence[str],
    *,
    repo_root: Optional[Path] = None,
) -> Path:
    """First path from :func:`resolve_maniskill_trajectory_paths`."""
    return resolve_maniskill_trajectory_paths(
        data_root, env_id, trajectory_hdf5_paths, repo_root=repo_root
    )[0]

def episode_meta_from_final_info(
    final_info_episode: Dict[str, Any],
    env_idx: int,
) -> Dict[str, Any]:
    """
    Pull per-env scalars from ManiSkillVectorEnv ``infos[\"final_info\"][\"episode\"]`` (batched tensors).
    Returns JSON-friendly Python scalars suitable for pickle (e.g. ``success_once``, ``return``, ``episode_len``).
    """
    meta: Dict[str, Any] = {}
    if not isinstance(final_info_episode, dict):
        return meta
    for k, v in final_info_episode.items():
        try:
            if torch.is_tensor(v):
                ev = v[env_idx].detach().cpu()
                if ev.numel() == 1:
                    if ev.dtype == torch.bool:
                        meta[k] = bool(ev.item())
                    else:
                        meta[k] = float(ev.item()) if ev.is_floating_point() else int(ev.item())
                else:
                    meta[k] = ev.numpy().tolist()
            elif isinstance(v, np.ndarray):
                if v.ndim == 0:
                    meta[k] = v.item()
                elif v.shape[0] > env_idx:
                    el = np.asarray(v[env_idx])
                    meta[k] = el.item() if el.ndim == 0 else el.tolist()
            elif isinstance(v, (bool, np.bool_)):
                meta[k] = bool(v)
            elif isinstance(v, (int, np.integer)):
                meta[k] = int(v)
            elif isinstance(v, (float, np.floating)):
                meta[k] = float(v)
            else:
                meta[k] = v
        except (IndexError, KeyError, RuntimeError, ValueError, TypeError):
            continue
    return meta


def scale_episode_meta_for_icl_export(
    meta: Dict[str, Any],
    *,
    reward_scale: float,
    success_reward_bonus: float,
    success: bool,
) -> Dict[str, Any]:
    """
    Adjust env ``episode`` scalars so they match the **same** shaping as PPO / HDF5 step rewards:
    add optional terminal ``success_reward_bonus`` to cumulative return when ``success``, then
    multiply by ``reward_scale``. Also scales mean ``reward`` by ``reward_scale`` (terminal bonus is
    not re-distributed into the mean).

    Typical keys from Gymnasium / ManiSkill ``infos[\"final_info\"][\"episode\"]``: ``return``,
    ``r`` (duplicate total), ``reward`` (often mean return per step), lengths / flags unchanged.
    """
    if not meta:
        return {}
    rs = float(reward_scale)
    bonus = float(success_reward_bonus) if success else 0.0

    def _is_real(x: Any) -> bool:
        return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)

    out = dict(meta)
    raw_total: Optional[float] = None
    if "return" in out and _is_real(out["return"]):
        raw_total = float(out["return"])
    elif "r" in out and _is_real(out["r"]):
        raw_total = float(out["r"])
    if raw_total is not None:
        scaled_total = (raw_total + bonus) * rs
        if "return" in out and _is_real(out.get("return")):
            out["return"] = float(scaled_total)
        if "r" in out and _is_real(out.get("r")):
            out["r"] = float(scaled_total)
    if "reward" in out and _is_real(out["reward"]):
        out["reward"] = float(out["reward"]) * rs
    return out


def episode_success_from_batched_final_info(final_info_episode: Dict[str, Any], env_idx: int) -> bool:
    """True iff ManiSkill batched ``infos[\"final_info\"][\"episode\"]`` marks success for env ``env_idx``."""
    if not isinstance(final_info_episode, dict):
        return False
    for key in ("success_once", "success_at_end", "success"):
        if key not in final_info_episode:
            continue
        v = final_info_episode[key]
        try:
            if torch.is_tensor(v):
                if env_idx >= int(v.shape[0]):
                    continue
                x = v[env_idx]
                if x.numel() != 1:
                    continue
                return bool(x.item()) if x.dtype == torch.bool else bool(float(x.item()) > 0.5)
            if isinstance(v, np.ndarray):
                if env_idx >= int(v.shape[0]):
                    continue
                x = np.asarray(v[env_idx])
                if x.ndim != 0 and x.size != 1:
                    continue
                return bool(x.reshape(-1)[0])
            if isinstance(v, (bool, np.bool_)):
                return bool(v)
        except (IndexError, RuntimeError, ValueError, TypeError):
            continue
    return False

def collect_episodes_vector_env(
    env: Any,
    agent: torch.nn.Module,
    device: torch.device,
    num_episodes: int,
    max_steps_per_episode: int,
    action_space_low: torch.Tensor,
    action_space_high: torch.Tensor,
    rgb_resize_hw: Optional[int] = None,
    *,
    success_reward_bonus: float = 0.0,
    reward_scale: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Roll out ``num_episodes`` completed episodes on a ``ManiSkillVectorEnv``, deterministic policy.

    Uses **all** ``env.num_envs`` slots in parallel when ``num_envs > 1`` (GPU sim + batched render),
    which is much faster than ``num_envs==1`` for RGB logging. Requires ``reconfiguration_freq=0``
    when ``num_envs > 1`` (see PPO script env factory).

    If ``rgb_resize_hw`` is set (e.g. 256), each frame is resized before storage when needed; if the env
    human-render camera already matches (see PPO ``human_render_camera_configs``), this is a no-op.

    If ``success_reward_bonus`` is non-zero, that amount is added to the **last** stored step reward
    when ``final_info`` reports success for that env (same semantics as PPO ``--success-reward-bonus``).
    Then every stored step reward is multiplied by ``reward_scale`` (default 1.0) so HDF5 matches PPO.
    """
    agent.eval()
    n_envs = int(getattr(env, "num_envs", 1))
    if n_envs < 1:
        raise ValueError("env.num_envs must be >= 1")

    def clip_action(a: torch.Tensor) -> torch.Tensor:
        lo = action_space_low.to(device=a.device, dtype=a.dtype)
        hi = action_space_high.to(device=a.device, dtype=a.dtype)
        return torch.clamp(a.detach(), lo, hi)

    trajectories: List[Dict[str, Any]] = []
    obs, _ = env.reset()
    buffers: List[Dict[str, List]] = [
        {"obs": [], "act": [], "rew": [], "rgb": []} for _ in range(n_envs)
    ]
    steps_ct = [0] * n_envs

    while len(trajectories) < num_episodes:
        try:
            frame = env.render()
            rgb_rows = render_batch_to_rgb_list(frame, n_envs, rgb_resize_hw)
        except Exception:
            rgb_rows = [None] * n_envs

        with torch.no_grad():
            action = agent.get_action(obs, deterministic=True)
        action = clip_action(action)
        next_obs, reward, terminations, truncations, infos = env.step(action)

        partial_reset_idx: List[int] = []
        for e in range(n_envs):
            o_np = obs[e].detach().cpu().numpy().astype(np.float32).reshape(-1)
            a_np = action[e].detach().cpu().numpy().reshape(-1)
            r = float(reward.reshape(-1)[e].item())
            buffers[e]["obs"].append(o_np)
            buffers[e]["act"].append(a_np)
            buffers[e]["rew"].append(r)
            buffers[e]["rgb"].append(rgb_rows[e])
            steps_ct[e] += 1

            nat_done = bool(terminations.reshape(-1)[e].item()) or bool(
                truncations.reshape(-1)[e].item()
            )
            cap_done = steps_ct[e] >= max_steps_per_episode
            if not (nat_done or cap_done):
                continue

            if len(trajectories) < num_episodes:
                ep_meta = None
                ep_success = False
                if isinstance(infos, dict) and "final_info" in infos:
                    fi = infos["final_info"]
                    if isinstance(fi, dict) and "episode" in fi:
                        ep_meta = episode_meta_from_final_info(fi["episode"], e)
                        ep_success = episode_success_from_batched_final_info(fi["episode"], e)
                sb = float(success_reward_bonus)
                if sb != 0.0 and buffers[e]["rew"] and ep_success:
                    buffers[e]["rew"][-1] = float(buffers[e]["rew"][-1]) + sb
                rsf = float(reward_scale)
                if buffers[e]["rew"]:
                    buffers[e]["rew"] = [float(x) * rsf for x in buffers[e]["rew"]]
                if ep_meta is not None:
                    ep_meta = scale_episode_meta_for_icl_export(
                        ep_meta,
                        reward_scale=rsf,
                        success_reward_bonus=float(success_reward_bonus),
                        success=ep_success,
                    )
                try:
                    b = buffers[e]
                    trajectories.append(
                        finalize_trajectory_dict(
                            b["obs"],
                            b["act"],
                            b["rew"],
                            b["rgb"],
                            episode_meta=ep_meta,
                        )
                    )
                except ValueError:
                    pass

            buffers[e] = {"obs": [], "act": [], "rew": [], "rgb": []}
            steps_ct[e] = 0
            if cap_done and not nat_done:
                partial_reset_idx.append(e)

        obs = next_obs
        if partial_reset_idx:
            idx = torch.tensor(partial_reset_idx, device=obs.device, dtype=torch.long)
            obs, _ = env.reset(options=dict(env_idx=idx))

    return trajectories[:num_episodes]
