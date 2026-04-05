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

**On-disk format:** **HDF5** (``.h5``) with a single flat time axis: ``observations``, ``actions``,
``rewards``, ``terminals`` shaped ``(total_timesteps, ...)``, plus ``episode_starts`` and
``episode_lengths`` (``int64``, length ``num_episodes``) and ``episode_meta_json`` (variable-length UTF-8,
one string per episode, possibly empty). If any episode has RGB, ``images_view_*`` datasets share the same
time axis; episodes without RGB are stored as zero-filled frames so shapes stay aligned.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

import h5py

import torch.nn.functional as F

from src.data.trajectories import sort_trajectories_by_return

HDF5_TRAJ_FILE_ATTR = "icl_adaptation_maniskill_trajectories"
HDF5_TRAJ_VERSION = 2

# Row chunk size along the flat time axis (timesteps per HDF5 chunk).
_H5_FLAT_CHUNK_ROWS = 8192


def _h5_chunks_1d(total: int) -> tuple[int, ...]:
    r = min(_H5_FLAT_CHUNK_ROWS, max(1, int(total)))
    return (r,)


def _h5_chunks_2d(total: int, feat: int) -> tuple[int, int]:
    r = min(_H5_FLAT_CHUNK_ROWS, max(1, int(total)))
    return (r, int(feat))


def _h5_chunks_4d(total: int, h: int, w: int, c: int) -> tuple[int, int, int, int]:
    r = min(256, max(1, int(total)))
    return (r, int(h), int(w), int(c))


def _flat_h5_kw_1d(total: int) -> Dict[str, Any]:
    return dict(compression="gzip", compression_opts=4, chunks=_h5_chunks_1d(total))


def _flat_h5_kw_2d(total: int, feat: int) -> Dict[str, Any]:
    return dict(compression="gzip", compression_opts=4, chunks=_h5_chunks_2d(total, feat))


def _image_view_dataset_kwargs(
    arr_shape: tuple[int, int, int, int],
    compression: str,
    *,
    gzip_level: int = 4,
) -> Dict[str, Any]:
    """Kwargs for ``create_dataset`` on ``(T,H,W,C)`` uint8 image stacks."""
    t, h, w, c = (int(arr_shape[0]), int(arr_shape[1]), int(arr_shape[2]), int(arr_shape[3]))
    chunks = _h5_chunks_4d(t, h, w, c)
    key = str(compression).strip().lower()
    if key in ("none", "off", "false", "0"):
        return {"chunks": chunks}
    if key == "lzf":
        return {"compression": "lzf", "chunks": chunks}
    if key == "gzip":
        return {
            "compression": "gzip",
            "compression_opts": int(gzip_level),
            "chunks": chunks,
        }
    raise ValueError(
        f"image_hdf5_compression must be 'gzip', 'lzf', or 'none'; got {compression!r}"
    )


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


def _terminals_for_traj(t: Dict[str, Any]) -> np.ndarray:
    term = t.get("terminals")
    if term is None:
        term = t.get("dones")
    if term is None:
        T = int(np.asarray(t["rewards"]).shape[0])
        term = np.zeros(T, dtype=np.float32)
        term[-1] = 1.0
    return np.asarray(term, dtype=np.float32).reshape(-1)


def save_trajectories_hdf5(
    trajectories: List[Dict[str, Any]],
    out_path: Path,
    *,
    sort_by_return: bool = True,
    image_hdf5_compression: str = "gzip",
    image_gzip_level: int = 4,
) -> None:
    """Write trajectory list to flat HDF5 (v2): one time axis + ``episode_starts`` / ``episode_lengths``.

    ``image_hdf5_compression``: ``gzip`` (default, smaller), ``lzf`` (faster writes), or ``none``
    (fastest / largest). Scalar datasets keep gzip for compatibility.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tr: List[Dict[str, Any]] = list(trajectories)
    if sort_by_return and tr:
        tr = sort_trajectories_by_return(tr, ascending=False)

    if not tr:
        with h5py.File(out_path, "w") as f:
            f.attrs["format"] = HDF5_TRAJ_FILE_ATTR
            f.attrs["version"] = HDF5_TRAJ_VERSION
            f.attrs["num_episodes"] = 0
            f.attrs["total_timesteps"] = 0
            z = np.zeros((0,), dtype=np.int64)
            f.create_dataset("episode_starts", data=z)
            f.create_dataset("episode_lengths", data=z)
            f.create_dataset("observations", data=np.zeros((0, 1), dtype=np.float32))
            f.create_dataset("actions", data=np.zeros((0, 1), dtype=np.float32))
            f.create_dataset("rewards", data=np.zeros((0,), dtype=np.float32))
            f.create_dataset("terminals", data=np.zeros((0,), dtype=np.float32))
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset("episode_meta_json", shape=(0,), dtype=dt)
        return

    lengths = [int(np.asarray(t["observations"]).shape[0]) for t in tr]
    episode_lengths = np.asarray(lengths, dtype=np.int64)
    starts = np.zeros(len(tr), dtype=np.int64)
    if len(tr) > 1:
        starts[1:] = np.cumsum(episode_lengths[:-1])
    total_t = int(episode_lengths.sum())

    obs_list = [np.asarray(t["observations"], dtype=np.float32) for t in tr]
    act_list = [np.asarray(t["actions"], dtype=np.float32) for t in tr]
    rew_list = [np.asarray(t["rewards"], dtype=np.float32).reshape(-1) for t in tr]
    term_list = [_terminals_for_traj(t) for t in tr]

    obs_all = np.concatenate(obs_list, axis=0)
    act_all = np.concatenate(act_list, axis=0)
    rew_all = np.concatenate(rew_list, axis=0)
    term_all = np.concatenate(term_list, axis=0)
    if obs_all.shape[0] != total_t or act_all.shape[0] != total_t:
        raise ValueError("observations/actions length mismatch vs episode lengths")

    img_views: List[np.ndarray] = []
    has_any_img = any(isinstance(t.get("images"), list) and len(t["images"]) > 0 for t in tr)
    if has_any_img:
        ref_i = next(
            i
            for i, t in enumerate(tr)
            if isinstance(t.get("images"), list) and len(t["images"]) > 0
        )
        ref = tr[ref_i]
        n_view = len(ref["images"])
        hwc_list: List[tuple[int, int, int]] = []
        for v in range(n_view):
            a = np.asarray(ref["images"][v], dtype=np.uint8)
            if a.ndim != 4 or a.shape[-1] != 3:
                raise ValueError(f"images[{v}] must be (T, H, W, 3) uint8, got {a.shape}")
            hwc_list.append((int(a.shape[1]), int(a.shape[2]), int(a.shape[3])))
        for i, t in enumerate(tr):
            T = lengths[i]
            imgs = t.get("images")
            if isinstance(imgs, list) and len(imgs) > 0:
                if len(imgs) != n_view:
                    raise ValueError(f"Episode {i}: expected {n_view} image views, got {len(imgs)}")
                for v in range(n_view):
                    a = np.asarray(imgs[v], dtype=np.uint8)
                    h, w, c = hwc_list[v]
                    if a.shape[0] != T or tuple(a.shape[1:]) != (h, w, c):
                        raise ValueError(
                            f"Episode {i} view {v}: shape {a.shape} vs length {T} / HWC ({h},{w},{c})"
                        )
        for v in range(n_view):
            h, w, c = hwc_list[v]
            chunks: List[np.ndarray] = []
            for i, t in enumerate(tr):
                T = lengths[i]
                imgs = t.get("images")
                if isinstance(imgs, list) and len(imgs) == n_view:
                    chunks.append(np.asarray(imgs[v], dtype=np.uint8))
                else:
                    chunks.append(np.zeros((T, h, w, c), dtype=np.uint8))
            img_views.append(np.concatenate(chunks, axis=0))

    meta_strings: List[str] = []
    for t in tr:
        em = t.get("episode_meta")
        if isinstance(em, dict) and em:
            meta_strings.append(_episode_meta_to_attr(em))
        else:
            meta_strings.append("")

    with h5py.File(out_path, "w") as f:
        f.attrs["format"] = HDF5_TRAJ_FILE_ATTR
        f.attrs["version"] = HDF5_TRAJ_VERSION
        f.attrs["num_episodes"] = len(tr)
        f.attrs["total_timesteps"] = total_t
        f.create_dataset("episode_starts", data=starts)
        f.create_dataset("episode_lengths", data=episode_lengths)
        f.create_dataset("observations", data=obs_all, **_flat_h5_kw_2d(*obs_all.shape))
        f.create_dataset("actions", data=act_all, **_flat_h5_kw_2d(*act_all.shape))
        f.create_dataset("rewards", data=rew_all, **_flat_h5_kw_1d(total_t))
        f.create_dataset("terminals", data=term_all, **_flat_h5_kw_1d(total_t))
        dt = h5py.special_dtype(vlen=str)
        meta_ds = f.create_dataset("episode_meta_json", shape=(len(tr),), dtype=dt)
        for i, s in enumerate(meta_strings):
            meta_ds[i] = s
        for v, arr in enumerate(img_views):
            img_kw = _image_view_dataset_kwargs(
                tuple(arr.shape),
                image_hdf5_compression,
                gzip_level=image_gzip_level,
            )
            f.create_dataset(f"images_view_{v}", data=arr, **img_kw)


def load_trajectories_hdf5(path: Path) -> List[Dict[str, Any]]:
    """Load episodes from flat ``save_trajectories_hdf5`` v2 file into a list of trajectory dicts."""
    path = Path(path)
    out: List[Dict[str, Any]] = []
    with h5py.File(path, "r") as f:
        if f.attrs.get("format") != HDF5_TRAJ_FILE_ATTR:
            raise ValueError(f"{path}: not a ManiSkill ICL trajectory file (missing format attr)")
        ver = int(f.attrs.get("version", 0))
        if ver != HDF5_TRAJ_VERSION:
            raise ValueError(f"{path}: expected HDF5 version {HDF5_TRAJ_VERSION}, got {ver}")
        for name in (
            "episode_starts",
            "episode_lengths",
            "observations",
            "actions",
            "rewards",
            "terminals",
        ):
            if name not in f:
                raise ValueError(f"{path}: missing dataset {name!r}")
        starts = np.asarray(f["episode_starts"], dtype=np.int64).reshape(-1)
        lens = np.asarray(f["episode_lengths"], dtype=np.int64).reshape(-1)
        n_ep = int(starts.shape[0])
        if lens.shape[0] != n_ep:
            raise ValueError(f"{path}: episode_starts and episode_lengths length mismatch")
        obs = f["observations"]
        act = f["actions"]
        rew = f["rewards"]
        term = f["terminals"]
        total_t = int(obs.shape[0])
        if (
            int(act.shape[0]) != total_t
            or int(rew.shape[0]) != total_t
            or int(term.shape[0]) != total_t
        ):
            raise ValueError(f"{path}: flat array length mismatch")
        if n_ep == 0:
            return []
        if int(starts[0]) != 0:
            raise ValueError(f"{path}: episode_starts[0] must be 0, got {int(starts[0])}")
        if int(lens.sum()) != total_t:
            raise ValueError(f"{path}: sum(episode_lengths) != total_timesteps")
        for i in range(n_ep - 1):
            if int(starts[i] + lens[i]) != int(starts[i + 1]):
                raise ValueError(f"{path}: episode_starts not consecutive at episode {i}")

        meta_ds = f["episode_meta_json"] if "episode_meta_json" in f else None
        view_keys = sorted(
            (k for k in f.keys() if k.startswith("images_view_")),
            key=lambda x: int(str(x).split("_")[-1]),
        )

        obs_full = np.asarray(obs, dtype=np.float32)
        act_full = np.asarray(act, dtype=np.float32)
        rew_full = np.asarray(rew, dtype=np.float32).reshape(-1)
        term_full = np.asarray(term, dtype=np.float32).reshape(-1)
        img_full = [np.asarray(f[k], dtype=np.uint8) for k in view_keys] if view_keys else []

        for i in range(n_ep):
            s = int(starts[i])
            ln = int(lens[i])
            e = s + ln
            d: Dict[str, Any] = {
                "observations": obs_full[s:e].copy(),
                "actions": act_full[s:e].copy(),
                "rewards": rew_full[s:e].copy(),
                "terminals": term_full[s:e].copy(),
            }
            if meta_ds is not None:
                raw = meta_ds[i]
                if isinstance(raw, bytes):
                    raw_s = raw.decode("utf-8")
                else:
                    raw_s = str(raw)
                if raw_s:
                    meta = _episode_meta_from_attr(raw_s)
                    if meta:
                        d["episode_meta"] = meta
            if img_full:
                d["images"] = [im[s:e].copy() for im in img_full]
            out.append(d)
    return out


def load_trajectories_file(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load ManiSkill trajectory bundle from ``.h5`` / ``.hdf5`` (flat v2 only)."""
    p = Path(path)
    suf = p.suffix.lower()
    if suf in (".h5", ".hdf5"):
        return load_trajectories_hdf5(p)
    raise ValueError(f"Unsupported trajectory file type: {p} (expected .h5 or .hdf5)")


def _assert_trajectories_compatible_for_merge(trajs: List[Dict[str, Any]]) -> None:
    """Require consistent state/action dims and matching image view shapes when RGB is present."""
    obs_d: Optional[int] = None
    act_d: Optional[int] = None
    img_layout: Optional[Tuple[int, Tuple[Tuple[int, int, int], ...]]] = None
    for i, t in enumerate(trajs):
        o = np.asarray(t["observations"], dtype=np.float32)
        a = np.asarray(t["actions"], dtype=np.float32)
        od = int(o.shape[-1]) if o.ndim >= 2 else int(o.size)
        ad = int(a.shape[-1]) if a.ndim >= 2 else int(a.size)
        if obs_d is None:
            obs_d, act_d = od, ad
        elif od != obs_d or ad != act_d:
            raise ValueError(
                f"Episode {i}: expected state_dim={obs_d} action_dim={act_d}, got {od}, {ad}"
            )
        imgs = t.get("images")
        if not isinstance(imgs, list) or not imgs:
            continue
        n_view = len(imgs)
        hwc_tuples: List[Tuple[int, int, int]] = []
        for v in range(n_view):
            im = np.asarray(imgs[v], dtype=np.uint8)
            if im.ndim != 4 or int(im.shape[-1]) != 3:
                raise ValueError(f"Episode {i} view {v}: expected (T,H,W,3) uint8, got {im.shape}")
            hwc_tuples.append((int(im.shape[1]), int(im.shape[2]), int(im.shape[3])))
        layout = (n_view, tuple(hwc_tuples))
        if img_layout is None:
            img_layout = layout
        elif layout != img_layout:
            raise ValueError(
                f"Episode {i}: image layout {layout} != {img_layout} "
                "(cannot merge different resolutions or view counts)"
            )


def merge_maniskill_trajectory_hdf5(
    input_paths: Sequence[Union[str, Path]],
    out_path: Union[str, Path],
    *,
    sort_by_return: bool = True,
    image_hdf5_compression: str = "gzip",
    image_gzip_level: int = 4,
) -> Tuple[int, int]:
    """
    Load multiple flat v2 ManiSkill ICL ``.h5`` files (same format as ``save_trajectories_hdf5``),
    concatenate episodes in argument order, optionally sort by return, and write a single HDF5.

    Episodes without RGB in some files are fine if at least one file has images: missing RGB is
    stored as zeros (same as ``save_trajectories_hdf5``). All non-empty image episodes must share
    the same view count and (H, W, C).

    Returns:
        ``(num_episodes, total_timesteps)``.
    """
    merged: List[Dict[str, Any]] = []
    for p in tqdm(input_paths, desc="Loading trajectories", unit="file"):
        path = Path(p)
        if not path.is_file():
            raise FileNotFoundError(f"Not a file: {path}")
        part = load_trajectories_hdf5(path)
        merged.extend(part)
    if merged:
        _assert_trajectories_compatible_for_merge(merged)
    save_trajectories_hdf5(
        merged,
        Path(out_path),
        sort_by_return=sort_by_return,
        image_hdf5_compression=image_hdf5_compression,
        image_gzip_level=image_gzip_level,
    )
    if not merged:
        return 0, 0
    lengths = [int(np.asarray(t["observations"]).shape[0]) for t in merged]
    return len(merged), int(sum(lengths))


def resolve_maniskill_trajectory_path(data_root: Union[str, Path], env_id: str) -> Path:
    """``<data_root>/maniskill/<env_id>/trajectories.h5``."""
    root = Path(data_root)
    safe = env_id.replace("/", "_").replace(" ", "_")
    return root / "maniskill" / safe / "trajectories.h5"


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


def _render_batch_to_rgb_list(
    frame: Any, n_envs: int, rgb_resize_hw: Optional[int]
) -> List[Optional[np.ndarray]]:
    """Split ``env.render()`` output into ``n_envs`` (H,W,3) uint8 arrays; optional square resize."""
    rgb = frame
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()
    if rgb is None:
        return [None] * n_envs
    rgb = np.asarray(rgb)
    if rgb.ndim == 3:
        if n_envs != 1:
            raise ValueError(f"Render is (H,W,C) but n_envs={n_envs} (expected 1)")
        rows = [rgb]
    elif rgb.ndim == 4:
        if int(rgb.shape[0]) != n_envs:
            raise ValueError(f"Render batch {rgb.shape[0]} != n_envs {n_envs}")
        rows = [rgb[i] for i in range(n_envs)]
    else:
        raise ValueError(f"Unexpected render shape {rgb.shape}")
    out: List[Optional[np.ndarray]] = []
    for row in rows:
        x = np.asarray(row, dtype=np.uint8)
        if rgb_resize_hw is not None and rgb_resize_hw > 0:
            x = _resize_rgb_uint8_to_square(x, int(rgb_resize_hw))
        out.append(x)
    return out


def _resize_rgb_uint8_to_square(rgb: np.ndarray, hw: int) -> np.ndarray:
    """Down/upscale RGB uint8 (H,W,3) to (hw,hw,3) for smaller HDF5 / faster IO."""
    if hw <= 0 or rgb is None or not isinstance(rgb, np.ndarray):
        return rgb
    if rgb.ndim != 3 or int(rgb.shape[-1]) != 3:
        return rgb
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    if h == hw and w == hw:
        return rgb
    t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    t = F.interpolate(t, size=(hw, hw), mode="bilinear", align_corners=False)
    out = (t.squeeze(0).permute(1, 2, 0) * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
    return np.asarray(out, dtype=np.uint8)


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
    rgb_resize_hw: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Roll out ``num_episodes`` completed episodes on a ``ManiSkillVectorEnv``, deterministic policy.

    Uses **all** ``env.num_envs`` slots in parallel when ``num_envs > 1`` (GPU sim + batched render),
    which is much faster than ``num_envs==1`` for RGB logging. Requires ``reconfiguration_freq=0``
    when ``num_envs > 1`` (see PPO script env factory).

    If ``rgb_resize_hw`` is set (e.g. 256), each frame is resized before storage when needed; if the env
    human-render camera already matches (see PPO ``human_render_camera_configs``), this is a no-op.
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
            rgb_rows = _render_batch_to_rgb_list(frame, n_envs, rgb_resize_hw)
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
                if isinstance(infos, dict) and "final_info" in infos:
                    fi = infos["final_info"]
                    if isinstance(fi, dict) and "episode" in fi:
                        ep_meta = episode_meta_from_final_info(fi["episode"], e)
                try:
                    b = buffers[e]
                    trajectories.append(
                        _finalize_traj(
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
