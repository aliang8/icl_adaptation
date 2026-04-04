"""
Build trajectory dicts compatible with ``ICLTrajectoryDataset`` / ``get_icl_trajectory_dataset``.

Expected keys per trajectory:
  - observations: (T, state_dim) float32  (ManiSkill state vector)
  - actions: (T, act_dim) float32
  - rewards: (T,) float32
  - terminals: (T,) float32  (1.0 on last step of episode, else 0.0)
  - images: optional list of one ``(T, H, W, 3)`` uint8 RGB array (VD4RL / ICRT convention)
  - episode_meta: optional dict of scalar episode metrics when available (e.g. ManiSkill
    ``success_once``, ``success_at_end``, ``fail_once``, ``fail_at_end``, ``return``, ``episode_len``)

**On-disk format:** episodes are stored in **HDF5** (``.h5``) — compressed, streamable, and the usual
choice for offline RL / imitation datasets. Legacy ``trajectories.pkl`` is still loaded if present.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

import h5py

from src.data.trajectories import sort_trajectories_by_return

HDF5_TRAJ_FILE_ATTR = "icl_adaptation_maniskill_trajectories"
HDF5_TRAJ_VERSION = 1

# Chunked storage required for shuffle filter; gzip alone is portable and still shrinks float series well.
_H5_DATASET_KW: Dict[str, Any] = dict(compression="gzip", compression_opts=4, chunks=True)


def _episode_meta_to_attr(meta: Dict[str, Any]) -> str:
    return json.dumps(meta, default=str)


def _episode_meta_from_attr(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, bytes):
        s = raw.decode("utf-8")
    else:
        s = str(raw)
    out = json.loads(s)
    return out if isinstance(out, dict) else {}


def save_trajectories_hdf5(
    trajectories: List[Dict[str, Any]],
    out_path: Path,
    *,
    sort_by_return: bool = True,
) -> None:
    """Write trajectory list to HDF5 (one group per episode). Preferred format for ICL training."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tr: List[Dict[str, Any]] = list(trajectories)
    if sort_by_return and tr:
        tr = sort_trajectories_by_return(tr, ascending=False)

    with h5py.File(out_path, "w") as f:
        f.attrs["format"] = HDF5_TRAJ_FILE_ATTR
        f.attrs["version"] = HDF5_TRAJ_VERSION
        f.attrs["num_episodes"] = len(tr)
        for i, t in enumerate(tr):
            grp = f.create_group(f"ep_{i:06d}")
            grp.create_dataset(
                "observations",
                data=np.asarray(t["observations"], dtype=np.float32),
                **_H5_DATASET_KW,
            )
            grp.create_dataset(
                "actions",
                data=np.asarray(t["actions"], dtype=np.float32),
                **_H5_DATASET_KW,
            )
            grp.create_dataset(
                "rewards",
                data=np.asarray(t["rewards"], dtype=np.float32).reshape(-1),
                **_H5_DATASET_KW,
            )
            term = t.get("terminals")
            if term is None:
                term = t.get("dones")
            if term is None:
                T = int(np.asarray(t["rewards"]).shape[0])
                term = np.zeros(T, dtype=np.float32)
                term[-1] = 1.0
            grp.create_dataset(
                "terminals",
                data=np.asarray(term, dtype=np.float32).reshape(-1),
                **_H5_DATASET_KW,
            )
            imgs = t.get("images")
            if isinstance(imgs, list) and imgs:
                ig = grp.create_group("images")
                for vi, arr in enumerate(imgs):
                    a = np.asarray(arr, dtype=np.uint8)
                    chunks = (1,) + tuple(a.shape[1:]) if a.ndim == 4 else True
                    ig.create_dataset(
                        f"view_{vi}",
                        data=a,
                        compression="gzip",
                        compression_opts=4,
                        chunks=chunks,
                    )
            em = t.get("episode_meta")
            if isinstance(em, dict) and em:
                grp.attrs["episode_meta_json"] = _episode_meta_to_attr(em)


def load_trajectories_hdf5(path: Path) -> List[Dict[str, Any]]:
    """Load episodes from ``save_trajectories_hdf5`` (or compatible) file."""
    path = Path(path)
    out: List[Dict[str, Any]] = []
    with h5py.File(path, "r") as f:
        keys = sorted(k for k in f.keys() if k.startswith("ep_"))
        for k in keys:
            grp = f[k]
            d: Dict[str, Any] = {
                "observations": np.asarray(grp["observations"], dtype=np.float32),
                "actions": np.asarray(grp["actions"], dtype=np.float32),
                "rewards": np.asarray(grp["rewards"], dtype=np.float32).reshape(-1),
                "terminals": np.asarray(grp["terminals"], dtype=np.float32).reshape(-1),
            }
            if "episode_meta_json" in grp.attrs:
                meta = _episode_meta_from_attr(grp.attrs["episode_meta_json"])
                if meta:
                    d["episode_meta"] = meta
            if "images" in grp:
                views = sorted(
                    grp["images"].keys(),
                    key=lambda x: int(str(x).split("_")[-1]),
                )
                d["images"] = [np.asarray(grp["images"][vn], dtype=np.uint8) for vn in views]
            out.append(d)
    return out


def load_trajectories_file(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load ``.h5`` / ``.hdf5`` or legacy ``.pkl`` trajectory bundles."""
    p = Path(path)
    suf = p.suffix.lower()
    if suf in (".h5", ".hdf5"):
        return load_trajectories_hdf5(p)
    if suf == ".pkl":
        return load_trajectories_pickle(p)
    raise ValueError(f"Unsupported trajectory file type: {p} (use .h5 or .pkl)")


def resolve_maniskill_trajectory_path(data_root: Union[str, Path], env_id: str) -> Path:
    """
    Return ``trajectories.h5`` if it exists under ``<data_root>/maniskill/<env_id>/``,
    else ``trajectories.pkl`` (may not exist yet).
    """
    root = Path(data_root)
    safe = env_id.replace("/", "_").replace(" ", "_")
    d = root / "maniskill" / safe
    h5 = d / "trajectories.h5"
    if h5.is_file():
        return h5
    return d / "trajectories.pkl"


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


def _finalize_traj(
    obs_l: List[np.ndarray],
    act_l: List[np.ndarray],
    rew_l: List[float],
    rgb_l: List[Optional[np.ndarray]],
    episode_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    T = len(obs_l)
    if T == 0:
        raise ValueError("empty episode")
    terminals = np.zeros(T, dtype=np.float32)
    terminals[-1] = 1.0
    out: Dict[str, Any] = {
        "observations": np.stack(obs_l, axis=0).astype(np.float32),
        "actions": np.stack(act_l, axis=0).astype(np.float32),
        "rewards": np.asarray(rew_l, dtype=np.float32).reshape(-1),
        "terminals": terminals,
    }
    if rgb_l and all(x is not None for x in rgb_l):
        rgb = np.stack([np.asarray(x, dtype=np.uint8) for x in rgb_l], axis=0)
        if rgb.ndim == 4 and rgb.shape[-1] == 3:
            out["images"] = [rgb]
    if episode_meta is not None:
        out["episode_meta"] = dict(episode_meta)
    return out


def append_ppo_rollout_to_episode_buffers(
    obs_np: np.ndarray,
    actions_np: np.ndarray,
    rewards_np: np.ndarray,
    done_after_np: np.ndarray,
    reward_scale: float,
    buffers: List[Dict[str, List]],
    out_complete: List[Dict[str, Any]],
    episode_meta_grid: Optional[np.ndarray] = None,
) -> None:
    """
    Stitch one PPO rollout block ``(num_steps, num_envs, ...)`` into episode dicts.

    ``done_after_np[t, e]`` is True iff env ``e`` ended the episode on step ``t`` (after
    ``env.step``). ``rewards_np`` is as stored in PPO (often scaled); we divide by
    ``reward_scale`` when exporting so ``rewards`` match env units when scale ≠ 0.

    Does not record RGB (training rollouts do not render); trajectories are state-only.

    ``episode_meta_grid`` optional ``(S, N)`` ``dtype=object`` array: at each step/env where an
    episode ends, the cell may hold a dict from ``episode_meta_from_final_info``; otherwise ``None``.
    """
    if obs_np.ndim < 2 or actions_np.ndim < 2:
        raise ValueError("obs_np and actions_np need at least (S, N, ...)")
    S, N = int(obs_np.shape[0]), int(obs_np.shape[1])
    if rewards_np.shape != (S, N) or done_after_np.shape != (S, N):
        raise ValueError("rewards_np and done_after_np must be (S, N)")
    if episode_meta_grid is not None:
        em = np.asarray(episode_meta_grid, dtype=object)
        if em.shape != (S, N):
            raise ValueError(f"episode_meta_grid shape {em.shape} != {(S, N)}")
        episode_meta_grid = em
    if len(buffers) != N:
        raise ValueError(f"buffers length {len(buffers)} != num_envs {N}")
    rs = float(reward_scale)
    rdiv = rs if abs(rs) > 1e-12 else 1.0

    for e in range(N):
        b = buffers[e]
        for t in range(S):
            o = np.asarray(obs_np[t, e], dtype=np.float32).reshape(-1)
            a = np.asarray(actions_np[t, e], dtype=np.float32).reshape(-1)
            rw = float(rewards_np[t, e]) / rdiv
            b["obs"].append(o)
            b["act"].append(a)
            b["rew"].append(rw)
            if bool(done_after_np[t, e]):
                if b["obs"]:
                    em = None
                    if episode_meta_grid is not None:
                        cell = episode_meta_grid[t, e]
                        if isinstance(cell, dict):
                            em = dict(cell)
                        else:
                            em = {}
                    out_complete.append(
                        _finalize_traj(b["obs"], b["act"], b["rew"], [], episode_meta=em)
                    )
                b["obs"] = []
                b["act"] = []
                b["rew"] = []


def flush_episode_buffers(
    buffers: List[Dict[str, List]],
    out_complete: List[Dict[str, Any]],
) -> None:
    """Finalize any in-progress segments (e.g. end of training)."""
    for b in buffers:
        if b["obs"]:
            out_complete.append(_finalize_traj(b["obs"], b["act"], b["rew"], []))
            b["obs"] = []
            b["act"] = []
            b["rew"] = []


def collect_episodes_vector_env(
    env: Any,
    agent: torch.nn.Module,
    device: torch.device,
    num_episodes: int,
    max_steps_per_episode: int,
    action_space_low: torch.Tensor,
    action_space_high: torch.Tensor,
) -> List[Dict[str, Any]]:
    """
    Roll out ``num_episodes`` on a ManiSkillVectorEnv with ``num_envs==1``, deterministic policy.
    Records state observations, clipped actions, rewards, and RGB frames from ``env.render()``.
    """
    agent.eval()

    def clip_action(a: torch.Tensor) -> torch.Tensor:
        return torch.clamp(a.detach(), action_space_low, action_space_high)

    trajectories: List[Dict[str, Any]] = []
    obs, _ = env.reset()
    obs_l: List[np.ndarray] = []
    act_l: List[np.ndarray] = []
    rew_l: List[float] = []
    rgb_l: List[Optional[np.ndarray]] = []

    episodes_done = 0
    steps = 0

    while episodes_done < num_episodes:
        rgb = None
        try:
            frame = env.render()
            if isinstance(frame, np.ndarray):
                rgb = frame
            elif isinstance(frame, torch.Tensor):
                rgb = frame.detach().cpu().numpy()
            if isinstance(rgb, np.ndarray) and rgb.ndim == 4 and rgb.shape[0] == 1:
                rgb = rgb[0]
        except Exception:
            rgb = None

        with torch.no_grad():
            action = agent.get_action(obs, deterministic=True)
        action = clip_action(action)
        next_obs, reward, terminations, truncations, infos = env.step(action)
        r = float(reward.view(-1)[0].item())
        done = bool(terminations.view(-1)[0].item() or truncations.view(-1)[0].item())

        o_np = obs.detach().cpu().numpy().reshape(1, -1)[0].astype(np.float32)
        a_np = action.detach().cpu().numpy().reshape(1, -1)[0].astype(np.float32)
        obs_l.append(o_np)
        act_l.append(a_np)
        rew_l.append(r)
        rgb_l.append(rgb)

        steps += 1
        obs = next_obs

        if done or steps >= max_steps_per_episode:
            ep_meta = None
            if isinstance(infos, dict) and "final_info" in infos:
                fi = infos["final_info"]
                if isinstance(fi, dict) and "episode" in fi:
                    ep_meta = episode_meta_from_final_info(fi["episode"], 0)
            try:
                trajectories.append(
                    _finalize_traj(obs_l, act_l, rew_l, rgb_l, episode_meta=ep_meta)
                )
            except ValueError:
                pass
            episodes_done += 1
            obs_l, act_l, rew_l, rgb_l = [], [], [], []
            steps = 0
            if episodes_done < num_episodes:
                obs, _ = env.reset()

    return trajectories


def save_trajectories_pickle(
    trajectories: List[Dict[str, Any]],
    out_path: Path,
    sort_by_return: bool = True,
) -> None:
    """Legacy pickle export (no compression). Prefer :func:`save_trajectories_hdf5` for new data."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tr = trajectories
    if sort_by_return and tr:
        tr = sort_trajectories_by_return(list(tr), ascending=False)
    with open(out_path, "wb") as f:
        pickle.dump(tr, f, protocol=pickle.HIGHEST_PROTOCOL)


def default_icl_export_path(icl_data_root: str, env_id: str) -> Path:
    """``<icl_data_root>/maniskill/<env_id>/trajectories.h5`` (flat env_id, safe for paths)."""
    safe = env_id.replace("/", "_").replace(" ", "_")
    return Path(icl_data_root) / "maniskill" / safe / "trajectories.h5"


def icl_image_snapshots_dir(icl_data_root: str, env_id: str) -> Path:
    """Directory for periodic RGB snapshot pickles: ``.../maniskill/<env_id>/image_snapshots/``."""
    safe = env_id.replace("/", "_").replace(" ", "_")
    return Path(icl_data_root) / "maniskill" / safe / "image_snapshots"


def icl_image_snapshot_path(icl_data_root: str, env_id: str, train_global_step: int) -> Path:
    """``image_snapshots/trajectories_step_XXXXXXXX.h5`` (8-digit step boundary)."""
    d = icl_image_snapshots_dir(icl_data_root, env_id)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"trajectories_step_{int(train_global_step):08d}.h5"


def load_trajectories_pickle(path: Path) -> List[Dict[str, Any]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list of trajectories in {path}")
    return data
