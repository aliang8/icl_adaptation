"""
ICL **replay buffer** on-disk format and HDF5 I/O (environment-agnostic).

Binary layout: one contiguous time axis per file with ``episode_starts`` / ``episode_lengths``,
``observations``, ``actions``, ``rewards``, ``terminals``, optional ``episode_meta_json``,
and optional ``images_view_*``. The HDF5 ``format`` attribute is historical
(**``icl_adaptation_maniskill_trajectories``**); any exporter following this schema may use it.

**This module** holds save/load/append, PPO rollout stitching, optional directory discovery
(:func:`discover_ic_replay_buffer_paths`), and training helpers (:func:`resolve_trajectory_hdf5_path_entries`,
:func:`load_ic_replay_buffer_bundle`). Windowed training without loading full trajectories into RAM:
:class:`src.data.ic_replay_buffer_dataset.ICReplayBufferDataset`.

ManiSkill path wiring lives in :mod:`src.data.maniskill_io`.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

import h5py

import torch.nn.functional as F

from loguru import logger as log

from src.data.trajectories import sort_trajectories_by_return, trajectory_return

HDF5_TRAJ_FILE_ATTR = "icl_adaptation_maniskill_trajectories"
HDF5_TRAJ_VERSION = 2

# Episode metadata keys often logged as 0/1 scalars (e.g. Gymnasium / ManiSkill).
_BOOLISH_EPISODE_META_KEYS = frozenset(
    {"success_once", "success_at_end", "fail_once", "fail_at_end"}
)

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


def _episode_vec_dataset_kw(n_ep: int) -> Dict[str, Any]:
    r = min(_H5_FLAT_CHUNK_ROWS, max(1, int(n_ep)))
    return dict(compression="gzip", compression_opts=4, chunks=(r,), maxshape=(None,))


def _extendable_flat_kw_1d(total: int) -> Dict[str, Any]:
    d = _flat_h5_kw_1d(total)
    d["maxshape"] = (None,)
    return d


def _extendable_flat_kw_2d(total: int, feat: int) -> Dict[str, Any]:
    d = _flat_h5_kw_2d(total, feat)
    d["maxshape"] = (None, int(feat))
    return d


def _flatten_traj_list_to_arrays(
    tr: List[Dict[str, Any]],
    *,
    sort_by_return: bool,
    force_image_template: Optional[Tuple[int, List[Tuple[int, int, int]]]] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[List[np.ndarray]],
    List[str],
]:
    """Pack trajectory dicts into flat arrays for HDF5 (one contiguous time axis).

    ``force_image_template``: ``(n_view, [(h,w,c), ...])`` when appending to a file that already
    has ``images_view_*`` — pads missing RGB with zeros so shapes match.
    """
    tr = list(tr)
    if sort_by_return and tr:
        tr = sort_trajectories_by_return(tr, ascending=False)
    if not tr:
        raise ValueError("_flatten_traj_list_to_arrays: empty trajectory list")

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

    img_views: Optional[List[np.ndarray]] = None
    if force_image_template is not None:
        n_view, hwc_list = force_image_template
        if n_view != len(hwc_list):
            raise ValueError("force_image_template n_view mismatch vs hwc_list")
        img_views = []
        for v in range(n_view):
            h, w, c = hwc_list[v]
            chunks: List[np.ndarray] = []
            for i, t in enumerate(tr):
                T = lengths[i]
                imgs = t.get("images")
                if isinstance(imgs, list) and len(imgs) == n_view:
                    a = np.asarray(imgs[v], dtype=np.uint8)
                    if a.shape[0] != T or tuple(a.shape[1:]) != (h, w, c):
                        raise ValueError(
                            f"Episode {i} view {v}: shape {a.shape} vs length {T} / HWC ({h},{w},{c})"
                        )
                    chunks.append(a)
                else:
                    chunks.append(np.zeros((T, h, w, c), dtype=np.uint8))
            img_views.append(np.concatenate(chunks, axis=0))
    else:
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
            img_views = []
            for v in range(n_view):
                h, w, c = hwc_list[v]
                chunks = []
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

    return (
        episode_lengths,
        starts,
        obs_all,
        act_all,
        rew_all,
        term_all,
        img_views,
        meta_strings,
    )


def _sorted_images_view_keys(f: h5py.File) -> List[str]:
    out = [k for k in f.keys() if k.startswith("images_view_")]

    def _view_index(k: str) -> int:
        return int(k.rsplit("_", 1)[-1])

    return sorted(out, key=_view_index)


def _hdf5_image_template_from_file(f: h5py.File) -> Optional[Tuple[int, List[Tuple[int, int, int]]]]:
    keys = _sorted_images_view_keys(f)
    if not keys:
        return None
    hwc_list: List[Tuple[int, int, int]] = []
    for k in keys:
        d = f[k]
        if d.ndim != 4 or int(d.shape[-1]) != 3:
            raise ValueError(f"{k}: expected (T,H,W,3) uint8 layout")
        hwc_list.append((int(d.shape[1]), int(d.shape[2]), int(d.shape[3])))
    return (len(keys), hwc_list)


def append_trajectories_hdf5(
    path: Path,
    trajectories: List[Dict[str, Any]],
) -> None:
    """Append episodes to an existing v2 trajectory HDF5 written by ``save_trajectories_hdf5``.

    Requires datasets created with an extendible time axis (``maxshape``). Older shards without
    ``maxshape`` cannot be grown; re-export or delete the file.
    """
    path = Path(path)
    tr = list(trajectories)
    if not tr:
        return
    with h5py.File(path, "a") as f:
        if f.attrs.get("format") != HDF5_TRAJ_FILE_ATTR:
            raise ValueError(f"{path}: not an ICL flat v2 trajectory file (missing format attr)")
        if int(f.attrs.get("version", 0)) != HDF5_TRAJ_VERSION:
            raise ValueError(f"{path}: unsupported trajectory version for append")

        old_ep = int(f.attrs.get("num_episodes", 0))
        old_T = int(f.attrs.get("total_timesteps", 0))
        if old_T == 0 and old_ep == 0:
            raise ValueError(
                f"{path}: cannot append to an empty trajectory file; use save_trajectories_hdf5 first"
            )

        obs_ds = f["observations"]
        act_ds = f["actions"]
        obs_dim = int(obs_ds.shape[1])
        act_dim = int(act_ds.shape[1])
        img_templ = _hdf5_image_template_from_file(f)

        (
            ep_len_new,
            starts_rel,
            obs_new,
            act_new,
            rew_new,
            term_new,
            img_views_new,
            meta_strings,
        ) = _flatten_traj_list_to_arrays(
            tr,
            sort_by_return=False,
            force_image_template=img_templ,
        )

        if obs_new.shape[1] != obs_dim or act_new.shape[1] != act_dim:
            raise ValueError(
                f"{path}: observation/action dims mismatch file ({obs_dim},{act_dim}) "
                f"vs chunk ({obs_new.shape[1]},{act_new.shape[1]})"
            )
        if img_templ is not None:
            if img_views_new is None:
                raise ValueError(f"{path}: file has images_view_* but chunk has no image data")
            n_v, hwc_exp = img_templ
            if len(img_views_new) != n_v:
                raise ValueError("image view count mismatch on append")
            for v, arr in enumerate(img_views_new):
                h, w, c = hwc_exp[v]
                if tuple(arr.shape[1:]) != (h, w, c):
                    raise ValueError(f"append view {v}: HWC mismatch")
        elif img_views_new is not None:
            raise ValueError(f"{path}: chunk has RGB but HDF5 has no images_view_* datasets")

        new_T = int(obs_new.shape[0])
        new_ep = int(len(tr))
        starts_global = int(old_T) + starts_rel

        def _require_resize(ds: h5py.Dataset, name: str) -> None:
            ms = ds.maxshape
            if ms is None or len(ms) < 1 or ms[0] is not None:
                raise ValueError(
                    f"{path}: dataset {name!r} is not extendible along the time axis; "
                    "re-save with a current save_trajectories_hdf5 or start a new shard file"
                )

        _require_resize(obs_ds, "observations")
        _require_resize(act_ds, "actions")
        _require_resize(f["rewards"], "rewards")
        _require_resize(f["terminals"], "terminals")
        es = f["episode_starts"]
        el = f["episode_lengths"]
        _require_resize(es, "episode_starts")
        _require_resize(el, "episode_lengths")
        em = f["episode_meta_json"]
        _require_resize(em, "episode_meta_json")

        obs_ds.resize((old_T + new_T, obs_dim))
        obs_ds[old_T:] = obs_new
        act_ds.resize((old_T + new_T, act_dim))
        act_ds[old_T:] = act_new
        r_ds = f["rewards"]
        r_ds.resize((old_T + new_T,))
        r_ds[old_T:] = rew_new
        t_ds = f["terminals"]
        t_ds.resize((old_T + new_T,))
        t_ds[old_T:] = term_new

        es.resize((old_ep + new_ep,))
        es[old_ep:] = starts_global
        el.resize((old_ep + new_ep,))
        el[old_ep:] = ep_len_new

        em.resize((old_ep + new_ep,))
        if h5py.check_dtype(vlen=em.dtype) is not str:
            raise ValueError(f"{path}: episode_meta_json must be variable-length UTF-8 strings")
        for i, s in enumerate(meta_strings):
            em[old_ep + i] = s

        if img_views_new is not None:
            for v, arr in enumerate(img_views_new):
                key = f"images_view_{v}"
                img_ds = f[key]
                _require_resize(img_ds, key)
                h, w, c = (int(arr.shape[1]), int(arr.shape[2]), int(arr.shape[3]))
                img_ds.resize((old_T + new_T, h, w, c))
                img_ds[old_T:] = arr

        f.attrs["num_episodes"] = old_ep + new_ep
        f.attrs["total_timesteps"] = old_T + new_T


def save_trajectories_hdf5(
    trajectories: List[Dict[str, Any]],
    out_path: Path,
    *,
    sort_by_return: bool = True,
    image_hdf5_compression: str = "gzip",
    image_gzip_level: int = 4,
) -> None:
    """Write trajectory list to flat HDF5 (v2): one time axis + ``episode_starts`` / ``episode_lengths``.

    ``image_hdf5_compression`` for ``images_view_*``: ``gzip`` (default, smaller on disk), ``lzf``
    (faster writes), or ``none`` (fastest / largest). **Gzip on ``images_view_*`` is single-threaded
    and slow** when ``total_timesteps × H × W`` is large. Scalar datasets keep gzip.
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
            f.create_dataset("episode_starts", data=z, **_episode_vec_dataset_kw(0))
            f.create_dataset("episode_lengths", data=z, **_episode_vec_dataset_kw(0))
            f.create_dataset(
                "observations",
                data=np.zeros((0, 1), dtype=np.float32),
                **_extendable_flat_kw_2d(1, 1),
            )
            f.create_dataset(
                "actions",
                data=np.zeros((0, 1), dtype=np.float32),
                **_extendable_flat_kw_2d(1, 1),
            )
            f.create_dataset("rewards", data=np.zeros((0,), dtype=np.float32), **_extendable_flat_kw_1d(1))
            f.create_dataset("terminals", data=np.zeros((0,), dtype=np.float32), **_extendable_flat_kw_1d(1))
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset("episode_meta_json", shape=(0,), dtype=dt, maxshape=(None,))
        return

    (
        episode_lengths,
        starts,
        obs_all,
        act_all,
        rew_all,
        term_all,
        img_views,
        meta_strings,
    ) = _flatten_traj_list_to_arrays(tr, sort_by_return=False)
    total_t = int(episode_lengths.sum())
    has_any_img = img_views is not None

    with h5py.File(out_path, "w") as f:
        f.attrs["format"] = HDF5_TRAJ_FILE_ATTR
        f.attrs["version"] = HDF5_TRAJ_VERSION
        f.attrs["num_episodes"] = len(tr)
        f.attrs["total_timesteps"] = total_t
        f.create_dataset("episode_starts", data=starts, **_episode_vec_dataset_kw(len(starts)))
        f.create_dataset("episode_lengths", data=episode_lengths, **_episode_vec_dataset_kw(len(episode_lengths)))
        f.create_dataset("observations", data=obs_all, **_extendable_flat_kw_2d(*obs_all.shape))
        f.create_dataset("actions", data=act_all, **_extendable_flat_kw_2d(*act_all.shape))
        f.create_dataset("rewards", data=rew_all, **_extendable_flat_kw_1d(total_t))
        f.create_dataset("terminals", data=term_all, **_extendable_flat_kw_1d(total_t))
        dt = h5py.special_dtype(vlen=str)
        meta_ds = f.create_dataset(
            "episode_meta_json",
            shape=(len(tr),),
            dtype=dt,
            maxshape=(None,),
        )
        for i, s in enumerate(meta_strings):
            meta_ds[i] = s
        if has_any_img:
            for v, arr in enumerate(img_views):
                img_kw = _image_view_dataset_kwargs(
                    tuple(arr.shape),
                    image_hdf5_compression,
                    gzip_level=image_gzip_level,
                )
                h, w, c = (int(arr.shape[1]), int(arr.shape[2]), int(arr.shape[3]))
                img_kw["maxshape"] = (None, h, w, c)
                f.create_dataset(f"images_view_{v}", data=arr, **img_kw)


def load_trajectories_hdf5(path: Path) -> List[Dict[str, Any]]:
    """Load episodes from flat ``save_trajectories_hdf5`` v2 file into a list of trajectory dicts."""
    path = Path(path)
    out: List[Dict[str, Any]] = []
    with h5py.File(path, "r") as f:
        if f.attrs.get("format") != HDF5_TRAJ_FILE_ATTR:
            raise ValueError(f"{path}: not an ICL flat v2 trajectory file (missing format attr)")
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
        images_store = str(f.attrs.get("images_store", "hdf5")).strip().lower()
        if images_store == "zarr":
            raise ValueError(
                f"{path}: legacy Zarr RGB sidecar (images_store=zarr) is no longer supported; "
                "re-export or merge trajectories so RGB lives in HDF5 ``images_view_*`` datasets."
            )
        view_keys = sorted(
            (k for k in f.keys() if k.startswith("images_view_")),
            key=lambda x: int(str(x).split("_")[-1]),
        )
        img_src: List[Any] = [f[k] for k in view_keys]

        obs_full = np.asarray(obs, dtype=np.float32)
        act_full = np.asarray(act, dtype=np.float32)
        rew_full = np.asarray(rew, dtype=np.float32).reshape(-1)
        term_full = np.asarray(term, dtype=np.float32).reshape(-1)

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
            if img_src:
                d["images"] = [np.asarray(im[s:e], dtype=np.uint8).copy() for im in img_src]
            out.append(d)
    return out


def load_trajectory_plot_stats_hdf5(path: Union[str, Path]) -> Dict[str, Any]:
    """Per-episode stats for plotting / summaries without loading observations, actions, or images.

    Reads only ``episode_starts``, ``episode_lengths``, ``rewards``, and optionally
    ``episode_meta_json`` (for ``success_once``). Same format/version checks as
    :func:`load_trajectories_hdf5` for the datasets that are read.

    Returns a dict with:

    - ``path``: resolved path string
    - ``n_episodes``: int
    - ``returns``: ``(n_episodes,)`` float64, sum of step rewards per episode
    - ``episode_lengths``: ``(n_episodes,)`` int64
    - ``success_once_known``: ``(n_episodes,)`` bool — meta present and key found
    - ``success_once``: ``(n_episodes,)`` bool — only meaningful where ``success_once_known``
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        if f.attrs.get("format") != HDF5_TRAJ_FILE_ATTR:
            raise ValueError(f"{path}: not an ICL flat v2 trajectory file (missing format attr)")
        ver = int(f.attrs.get("version", 0))
        if ver != HDF5_TRAJ_VERSION:
            raise ValueError(f"{path}: expected HDF5 version {HDF5_TRAJ_VERSION}, got {ver}")
        for name in ("episode_starts", "episode_lengths", "rewards"):
            if name not in f:
                raise ValueError(f"{path}: missing dataset {name!r}")
        starts = np.asarray(f["episode_starts"], dtype=np.int64).reshape(-1)
        lens = np.asarray(f["episode_lengths"], dtype=np.int64).reshape(-1)
        n_ep = int(starts.shape[0])
        if lens.shape[0] != n_ep:
            raise ValueError(f"{path}: episode_starts and episode_lengths length mismatch")
        rew = f["rewards"]
        total_t = int(rew.shape[0])
        if n_ep == 0:
            return {
                "path": str(path.resolve()),
                "n_episodes": 0,
                "returns": np.zeros(0, dtype=np.float64),
                "episode_lengths": np.zeros(0, dtype=np.int64),
                "success_once_known": np.zeros(0, dtype=bool),
                "success_once": np.zeros(0, dtype=bool),
            }
        if int(starts[0]) != 0:
            raise ValueError(f"{path}: episode_starts[0] must be 0, got {int(starts[0])}")
        if int(lens.sum()) != total_t:
            raise ValueError(f"{path}: sum(episode_lengths) != total_timesteps")
        for i in range(n_ep - 1):
            if int(starts[i] + lens[i]) != int(starts[i + 1]):
                raise ValueError(f"{path}: episode_starts not consecutive at episode {i}")

        rew_full = np.asarray(rew, dtype=np.float64).reshape(-1)
        if rew_full.shape[0] != total_t:
            raise ValueError(f"{path}: rewards length mismatch")

        cum = np.empty(total_t + 1, dtype=np.float64)
        cum[0] = 0.0
        np.cumsum(rew_full, out=cum[1:])
        ends = starts + lens
        rets = cum[ends] - cum[starts]

        known = np.zeros(n_ep, dtype=bool)
        succ = np.zeros(n_ep, dtype=bool)
        meta_ds = f["episode_meta_json"] if "episode_meta_json" in f else None
        if meta_ds is not None:
            for i in range(n_ep):
                raw = meta_ds[i]
                if isinstance(raw, bytes):
                    raw_s = raw.decode("utf-8")
                else:
                    raw_s = str(raw)
                if not raw_s.strip():
                    continue
                meta = _episode_meta_from_attr(raw_s)
                if not isinstance(meta, dict) or "success_once" not in meta:
                    continue
                known[i] = True
                succ[i] = bool(meta["success_once"])

    return {
        "path": str(path.resolve()),
        "n_episodes": n_ep,
        "returns": rets,
        "episode_lengths": lens.copy(),
        "success_once_known": known,
        "success_once": succ,
    }


def load_trajectories_file(
    path: Union[str, Path],
    *,
    min_episode_length: Optional[int] = None,
    log_summary: bool = False,
) -> List[Dict[str, Any]]:
    """Load ICL flat v2 trajectory bundle from ``.h5`` / ``.hdf5``.

    If ``min_episode_length`` is set and positive, keep only trajectories with reward length
    ``T >= min_episode_length`` (typical AD / training: drop very short episodes). When
    ``log_summary`` is true, logs a formatted summary before and after filtering (if any).
    """
    p = Path(path)
    suf = p.suffix.lower()
    if suf not in (".h5", ".hdf5"):
        raise ValueError(f"Unsupported trajectory file type: {p} (expected .h5 or .hdf5)")
    tr = load_trajectories_hdf5(p)
    hint = str(p)
    if log_summary:
        log.info(
            "{}\n{}",
            f"ICL flat trajectory bundle (raw): {p.name}",
            format_ic_replay_buffer_summary(tr, source_hint=hint),
        )
    if min_episode_length is not None and int(min_episode_length) > 0:
        ml = int(min_episode_length)
        tr, n_before, n_after = filter_trajectories_min_episode_length(tr, ml)
        log.info(
            "ICL flat length filter: kept {}/{} episodes with T >= {}",
            n_after,
            n_before,
            ml,
        )
        if log_summary:
            log.info(
                "{}\n{}",
                f"ICL flat trajectory bundle (after T>={ml}): {p.name}",
                format_ic_replay_buffer_summary(tr, source_hint=hint),
            )
    return tr


def episode_length_from_trajectory(t: Dict[str, Any]) -> int:
    """Episode length ``T`` from ``rewards`` (fallback ``observations``)."""
    r = t.get("rewards")
    if r is not None:
        return int(np.asarray(r, dtype=np.float32).reshape(-1).shape[0])
    o = t.get("observations")
    if o is not None:
        return int(np.asarray(o, dtype=np.float32).shape[0])
    raise ValueError("trajectory missing rewards and observations")


def filter_trajectories_episode_length_eq(
    trajectories: List[Dict[str, Any]],
    target_length: int,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Keep episodes with ``T == target_length``. Returns ``(filtered, n_before, n_after)``."""
    n_before = len(trajectories)
    tl = int(target_length)
    out: List[Dict[str, Any]] = []
    for t in trajectories:
        if episode_length_from_trajectory(t) == tl:
            out.append(t)
    return out, n_before, len(out)


def filter_trajectories_min_episode_length(
    trajectories: List[Dict[str, Any]],
    min_length: int,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Keep episodes with ``T >= min_length``. Returns ``(filtered, n_before, n_after)``."""
    n_before = len(trajectories)
    ml = int(min_length)
    out: List[Dict[str, Any]] = []
    for t in trajectories:
        if episode_length_from_trajectory(t) >= ml:
            out.append(t)
    return out, n_before, len(out)


def _collect_episode_meta_scalars(
    em: Dict[str, Any],
    meta_bool: Dict[str, List[float]],
    meta_num: Dict[str, List[float]],
) -> None:
    for k, v in em.items():
        if isinstance(v, (bool, np.bool_)):
            meta_bool.setdefault(k, []).append(float(bool(v)))
            continue
        if (
            k in _BOOLISH_EPISODE_META_KEYS
            and isinstance(v, (int, np.integer, float, np.floating))
            and not isinstance(v, bool)
        ):
            fv = float(v)
            if fv in (0.0, 1.0):
                meta_bool.setdefault(k, []).append(fv)
                continue
        if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
            meta_num.setdefault(k, []).append(float(v))
            continue
        if isinstance(v, (float, np.floating)):
            meta_num.setdefault(k, []).append(float(v))


def format_ic_replay_buffer_summary(
    trajs: List[Dict[str, Any]],
    *,
    source_hint: str = "",
) -> str:
    """
    Human-readable stats: counts, length / return distribution, RGB presence, ``episode_meta``
    aggregates (success rates, numeric means).
    """
    n = len(trajs)
    lines: List[str] = []
    if source_hint:
        lines.append(f"  source: {source_hint}")
    if n == 0:
        lines.append("  (empty)")
        return "\n".join(lines)

    lens: List[int] = []
    rets: List[float] = []
    n_img = 0
    img_shape: Optional[Tuple[int, ...]] = None
    obs_dim: Optional[int] = None
    act_dim: Optional[int] = None
    n_meta = 0
    meta_bool: Dict[str, List[float]] = {}
    meta_num: Dict[str, List[float]] = {}

    for t in trajs:
        o = t.get("observations")
        a = t.get("actions")
        r = t.get("rewards")
        if o is None or a is None or r is None:
            continue
        oa = np.asarray(o, dtype=np.float32)
        aa = np.asarray(a, dtype=np.float32)
        lens.append(int(oa.shape[0]))
        rets.append(trajectory_return(t))
        if obs_dim is None:
            obs_dim = int(oa.shape[-1]) if oa.ndim >= 2 else int(oa.size)
        if act_dim is None:
            act_dim = int(aa.shape[-1]) if aa.ndim >= 2 else int(aa.size)
        imgs = t.get("images")
        if isinstance(imgs, list) and len(imgs) > 0:
            n_img += 1
            if img_shape is None:
                img_shape = tuple(np.asarray(imgs[0]).shape)
        em = t.get("episode_meta")
        if isinstance(em, dict) and em:
            n_meta += 1
            _collect_episode_meta_scalars(em, meta_bool, meta_num)

    if not lens:
        lines.append("  (no valid trajectories: missing obs/actions/rewards)")
        return "\n".join(lines)

    la = np.asarray(lens, dtype=np.float64)
    ra = np.asarray(rets, dtype=np.float64)
    lines.append(f"  episodes:           {len(lens)}")
    lines.append(
        f"  episode length:     mean={la.mean():.2f}  std={la.std():.2f}  "
        f"min={int(la.min())}  max={int(la.max())}"
    )
    lines.append(
        f"  return (sum r):      mean={ra.mean():.6g}  std={ra.std():.6g}  "
        f"min={ra.min():.6g}  max={ra.max():.6g}"
    )
    if obs_dim is not None:
        lines.append(f"  state_dim:          {obs_dim}")
    if act_dim is not None:
        lines.append(f"  action_dim:         {act_dim}")
    lines.append(f"  with images:        {n_img}/{len(lens)}")
    if img_shape is not None:
        lines.append(f"  image shape (1st):  {img_shape}")
    lines.append(f"  with episode_meta:  {n_meta}/{len(lens)}")
    if n_meta < len(lens):
        lines.append(f"  episode_meta missing: {len(lens) - n_meta}/{len(lens)}")

    if meta_bool:
        lines.append("  episode_meta (boolean / 0-1):")
        for k in sorted(meta_bool.keys()):
            vals = np.asarray(meta_bool[k], dtype=np.float64)
            lines.append(
                f"    {k:20s}  true_rate={vals.mean():.4f}  (n={len(vals)} episodes with key)"
            )

    if meta_num:
        lines.append("  episode_meta (numeric):")
        for k in sorted(meta_num.keys()):
            vals = np.asarray(meta_num[k], dtype=np.float64)
            lines.append(
                f"    {k:20s}  mean={vals.mean():.4f}  std={vals.std():.4f}  "
                f"min={vals.min():.4f}  max={vals.max():.4f}  (n={len(vals)})"
            )

    return "\n".join(lines)


def summarize_ic_replay_buffer(
    trajs: List[Dict[str, Any]],
    *,
    title: str = "ICL replay buffer",
    source_hint: str = "",
) -> None:
    """Log :func:`format_ic_replay_buffer_summary` under ``title``."""
    log.info("{}\n{}", title, format_ic_replay_buffer_summary(trajs, source_hint=source_hint))


def paths_from_icl_shards_manifest(task_dir: Path) -> Optional[List[Path]]:
    """If ``icl_shards_manifest.json`` exists, return listed ``.h5`` paths (all must exist)."""
    mp = task_dir / "icl_shards_manifest.json"
    if not mp.is_file():
        return None
    try:
        doc = json.loads(mp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        log.warning("Ignoring {}: {}", mp, e)
        return None
    files = doc.get("h5_files")
    if not isinstance(files, list) or not files:
        return None
    out: List[Path] = []
    for name in files:
        stem = Path(str(name)).name
        p = task_dir / stem
        if not p.is_file():
            raise FileNotFoundError(
                f"{mp.name} lists {stem!r} but that file is missing under {task_dir}"
            )
        out.append(p.resolve())
    return out

def discover_ic_replay_buffer_paths(
    task_dir: Path,
    *,
    log_prefix: str = "ICL replay buffer",
) -> list[Path]:
    """
    Discover flat v2 ``.h5`` files under ``task_dir`` (one task / export folder).

    Order: ``icl_shards_manifest.json`` ``h5_files`` if present; else all
    ``trajectories_shard_*.h5`` (sorted); else first monolithic match among
    ``trajectories_state_5M.h5``, ``trajectories.h5``, ``trajectories_state.h5``;
    else ``trajectories_image_shard_*.h5`` (sorted).

    Shards are preferred before a lone ``trajectories.h5`` so shards are not shadowed.
    """
    task_dir = Path(task_dir).expanduser().resolve()
    manifest_paths = paths_from_icl_shards_manifest(task_dir)
    if manifest_paths:
        log.info(
            "{}: using {} HDF5 path(s) from {}",
            log_prefix,
            len(manifest_paths),
            (task_dir / "icl_shards_manifest.json").resolve(),
        )
        return manifest_paths
    shards = sorted(task_dir.glob("trajectories_shard_*.h5"))
    if shards:
        resolved = [p.resolve() for p in shards]
        log.info("{}: using {} shard file(s) under {}", log_prefix, len(resolved), task_dir)
        return resolved
    for c in (
        task_dir / "trajectories_state_5M.h5",
        task_dir / "trajectories.h5",
        task_dir / "trajectories_state.h5",
    ):
        if c.is_file():
            log.info("{}: using monolithic {}", log_prefix, c.resolve())
            return [c.resolve()]
    img_shards = sorted(task_dir.glob("trajectories_image_shard_*.h5"))
    if img_shards:
        resolved = [p.resolve() for p in img_shards]
        log.info("{}: using {} image shard file(s) under {}", log_prefix, len(resolved), task_dir)
        return resolved
    raise FileNotFoundError(
        f"No ICL replay buffer files under {task_dir}. "
        "Expected icl_shards_manifest.json + listed shards, trajectories_shard_*.h5, "
        "trajectories.h5, … or pass explicit paths via "
        ":func:`resolve_trajectory_hdf5_path_entries`."
    )

def finalize_trajectory_dict(
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
    buffers: List[Dict[str, List]],
    out_complete: List[Dict[str, Any]],
    episode_meta_grid: Optional[np.ndarray] = None,
    rgb_grid: Optional[np.ndarray] = None,
    *,
    episode_keep_fraction: float = 1.0,
) -> None:
    """
    Stitch one PPO rollout block ``(num_steps, num_envs, ...)`` into episode dicts.

    ``done_after_np[t, e]`` is True iff env ``e`` ended the episode on step ``t`` (after
    ``env.step``). ``rewards_np`` must match what PPO uses for learning (e.g. env reward with any
    terminal success bonus, then multiplied by ``reward_scale``). Those values are stored **as-is**
    in each trajectory's ``rewards`` so HDF5 rollout exports match the replay / learner signal.

    If ``rgb_grid`` is set, it must be ``uint8`` with shape ``(S, N, H, W, 3)`` (one frame per
    timestep/env, aligned with ``obs_np``); episodes include ``images`` in HDF5 when all frames are present.

    ``episode_meta_grid`` optional ``(S, N)`` ``dtype=object`` array: at each step/env where an
    episode ends, the cell may hold a dict from ``episode_meta_from_final_info``; otherwise ``None``.

    ``episode_keep_fraction``: if in ``(0, 1)``, each **completed** episode is appended to
    ``out_complete`` with that probability (uses the global ``random`` module — seed in the caller).
    ``1.0`` keeps all. Does not affect buffers: dropped episodes are still cleared from per-env state.
    """
    kf = float(episode_keep_fraction)
    if kf <= 0.0 or kf > 1.0:
        raise ValueError(f"episode_keep_fraction must be in (0, 1], got {episode_keep_fraction!r}")
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
    rg: Optional[np.ndarray] = None
    if rgb_grid is not None:
        rg = np.asarray(rgb_grid, dtype=np.uint8)
        if rg.ndim != 5 or int(rg.shape[-1]) != 3:
            raise ValueError(f"rgb_grid must be (S,N,H,W,3) uint8, got {rg.shape}")
        if int(rg.shape[0]) != S or int(rg.shape[1]) != N:
            raise ValueError(f"rgb_grid leading dims {(rg.shape[0], rg.shape[1])} != (S,N)={(S, N)}")

    for e in range(N):
        b = buffers[e]
        for t in range(S):
            o = np.asarray(obs_np[t, e], dtype=np.float32).reshape(-1)
            a = np.asarray(actions_np[t, e], dtype=np.float32).reshape(-1)
            rw = float(rewards_np[t, e])
            b["obs"].append(o)
            b["act"].append(a)
            b["rew"].append(rw)
            if rg is not None:
                b.setdefault("rgb", []).append(np.asarray(rg[t, e], dtype=np.uint8))
            if bool(done_after_np[t, e]):
                if b["obs"]:
                    em: Optional[Dict[str, Any]] = None
                    if episode_meta_grid is not None:
                        cell = episode_meta_grid[t, e]
                        if isinstance(cell, dict) and cell:
                            em = dict(cell)
                    rgb_lane = b.get("rgb", []) if rg is not None else []
                    if kf >= 1.0 or random.random() < kf:
                        out_complete.append(
                            finalize_trajectory_dict(b["obs"], b["act"], b["rew"], rgb_lane, episode_meta=em)
                        )
                b["obs"] = []
                b["act"] = []
                b["rew"] = []
                if rg is not None:
                    b["rgb"] = []


def render_batch_to_rgb_list(
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
            x = resize_rgb_uint8_to_square(x, int(rgb_resize_hw))
        out.append(x)
    return out


def resize_rgb_uint8_to_square(rgb: np.ndarray, hw: int) -> np.ndarray:
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
            rgb_lane = b.get("rgb", [])
            out_complete.append(finalize_trajectory_dict(b["obs"], b["act"], b["rew"], rgb_lane))
            b["obs"] = []
            b["act"] = []
            b["rew"] = []
            if "rgb" in b:
                b["rgb"] = []


# ---------------------------------------------------------------------------
# Path resolution + multi-file bundles (training stack)
# ---------------------------------------------------------------------------


def resolve_trajectory_hdf5_path_entries(
    entries: Sequence[str],
    *,
    data_root: Union[str, Path],
    search_dirs: Sequence[Union[str, Path]],
    extra_roots: Sequence[Union[str, Path]] = (),
) -> List[Path]:
    """
    Resolve each non-empty string to an existing ``.h5`` / ``.hdf5`` replay buffer file.

    Tries, in order: path as given (absolute or cwd-relative), ``cwd / entry``, ``data_root / entry``,
    then for each ``d`` in ``search_dirs``: ``d / basename(entry)`` and ``d / entry`` when entry has
    more than a basename, then for each ``r`` in ``extra_roots``: ``r / entry`` (relative entries only).
    """
    root = Path(data_root).expanduser().resolve()
    dirs = [Path(d).expanduser().resolve() for d in search_dirs]
    extras = [Path(x).expanduser().resolve() for x in extra_roots]
    out: List[Path] = []
    for raw in entries:
        e = str(raw).strip()
        if not e:
            continue
        out.append(_resolve_one_trajectory_hdf5(e, data_root=root, search_dirs=dirs, extra_roots=extras))
    if not out:
        raise ValueError("trajectory_hdf5_paths: no non-empty entries after filtering")
    return out


def _resolve_one_trajectory_hdf5(
    entry: str,
    *,
    data_root: Path,
    search_dirs: Sequence[Path],
    extra_roots: Sequence[Path] = (),
) -> Path:
    p0 = Path(entry).expanduser()
    candidates: List[Path] = []
    if p0.is_absolute():
        candidates.append(p0)
    else:
        candidates.append(p0)
        candidates.append(Path.cwd() / p0)
        candidates.append(data_root / p0)
        for sd in search_dirs:
            candidates.append(sd / p0.name)
            if p0.parent != Path("."):
                candidates.append(sd / p0)
        for er in extra_roots:
            candidates.append(er / p0)
    tried: List[str] = []
    for c in candidates:
        rp = c.resolve() if c.exists() else c
        key = str(rp)
        if key in tried:
            continue
        tried.append(key)
        if c.is_file():
            suf = c.suffix.lower()
            if suf not in (".h5", ".hdf5"):
                raise ValueError(f"Expected .h5 or .hdf5 replay buffer file, got: {c}")
            return c.resolve()
    raise FileNotFoundError(
        f"Replay buffer file not found: {entry!r} (searched under data_root={data_root}, "
        f"{len(search_dirs)} hint dir(s), {len(extra_roots)} extra root(s))"
    )


def _normalize_path_inputs(
    paths: Union[str, Path, Sequence[Union[str, Path]]],
) -> List[Path]:
    if isinstance(paths, (str, Path)):
        return [Path(paths).expanduser().resolve()]
    seq = [Path(p).expanduser().resolve() for p in paths]  # type: ignore[arg-type]
    if not seq:
        raise ValueError("At least one replay buffer path is required")
    return seq


def load_ic_replay_buffer_files(
    paths: Sequence[Union[str, Path]],
    *,
    min_episode_length: Optional[int] = None,
    log_summary: bool = False,
) -> List[Dict[str, Any]]:
    """
    Load several replay buffer bundles and concatenate episodes (file order).
    """
    plist = [Path(p).expanduser().resolve() for p in paths]
    for p in plist:
        if not p.is_file():
            raise FileNotFoundError(f"Not a file: {p}")
        suf = p.suffix.lower()
        if suf not in (".h5", ".hdf5"):
            raise ValueError(f"Unsupported replay buffer file type: {p}")
    merged: List[Dict[str, Any]] = []
    for p in plist:
        merged.extend(load_trajectories_hdf5(p))
    hint = ", ".join(x.name for x in plist[:8]) + (" …" if len(plist) > 8 else "")
    if log_summary and merged:
        log.info(
            "{}\n{}",
            f"ICL replay buffer bundle (raw, {len(plist)} file(s)): {hint}",
            format_ic_replay_buffer_summary(merged, source_hint=hint),
        )
    if merged:
        _assert_flat_bundle_compatible(merged)
    if min_episode_length is not None and int(min_episode_length) > 0:
        ml = int(min_episode_length)
        merged, n_before, n_after = filter_trajectories_min_episode_length(merged, ml)
        log.info(
            "ICL replay buffer length filter: kept {}/{} episodes with T >= {}",
            n_after,
            n_before,
            ml,
        )
        if log_summary and merged:
            log.info(
                "{}\n{}",
                f"ICL replay buffer bundle (after T>={ml}, {len(plist)} file(s))",
                format_ic_replay_buffer_summary(merged, source_hint=hint),
            )
    return merged


def _assert_flat_bundle_compatible(trajs: List[Dict[str, Any]]) -> None:
    """Require consistent state/action dims and matching image view layouts."""
    obs_d: Optional[int] = None
    act_d: Optional[int] = None
    img_layout: Optional[Tuple[int, Tuple[Tuple[int, int, int], ...]]] = None
    for i, t in enumerate(trajs):
        o = t.get("observations")
        a = t.get("actions")
        if o is None or a is None:
            raise ValueError(f"Episode {i}: missing observations/actions")
        oa = o if isinstance(o, np.ndarray) else np.asarray(o)
        aa = a if isinstance(a, np.ndarray) else np.asarray(a)
        od = int(oa.shape[-1]) if oa.ndim >= 2 else int(oa.size)
        ad = int(aa.shape[-1]) if aa.ndim >= 2 else int(aa.size)
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
        hwc: List[Tuple[int, int, int]] = []
        for v, arr in enumerate(imgs):
            im = arr if isinstance(arr, np.ndarray) else np.asarray(arr)
            if im.ndim != 4 or int(im.shape[-1]) != 3:
                raise ValueError(f"Episode {i} view {v}: expected (T,H,W,3) uint8, got {im.shape}")
            hwc.append((int(im.shape[1]), int(im.shape[2]), int(im.shape[3])))
        layout = (n_view, tuple(hwc))
        if img_layout is None:
            img_layout = layout
        elif layout != img_layout:
            raise ValueError(f"Episode {i}: image layout {layout} != {img_layout}")


def load_ic_replay_buffer_bundle(
    paths: Union[str, Path, Sequence[Union[str, Path]]],
    *,
    min_episode_length: Optional[int] = None,
    log_summary: bool = True,
) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    """
    Load one or many replay buffer files and build the ``(trajectories, prompt_per_task)``
    pair expected by :func:`src.data.dataset.get_icl_trajectory_dataset`.
    """
    plist = _normalize_path_inputs(paths)
    if len(plist) == 1:
        trajectories = load_trajectories_file(
            plist[0],
            min_episode_length=min_episode_length,
            log_summary=log_summary,
        )
    else:
        trajectories = load_ic_replay_buffer_files(
            plist,
            min_episode_length=min_episode_length,
            log_summary=log_summary,
        )
    if not trajectories:
        raise ValueError(f"No trajectories loaded from {plist!r}")
    pool = sort_trajectories_by_return(list(trajectories), ascending=False)
    prompt_per_task: List[List[Dict[str, Any]]] = [pool]
    return trajectories, prompt_per_task
