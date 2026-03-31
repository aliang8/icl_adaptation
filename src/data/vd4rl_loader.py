"""
Load V-D4RL pixel trajectories: Offline DreamerV2 `*.npz` (64px) or DrQ-v2 `*.hdf5` (84px).

Dataset layout (see https://github.com/conglu1997/v-d4rl):
  <data_root>/<suite>/<task>/<split>/<pixel_size>/*.npz   # e.g. main/.../64px
  <data_root>/<suite>/<task>/<split>/<pixel_size>/*.hdf5  # e.g. distracting/.../84px (shard_*.hdf5)

Upstream: 64px uses npz (DV2); 84px uses hdf5 (DrQ+BC). `load_vd4rl_npz_trajectories` tries npz
first, then hdf5 if the directory has no npz files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger as log

from src.data.trajectories import sort_trajectories_by_return

# V-D4RL drqbc/utils.py: step_type 0=FIRST, 1=MID, 2=LAST
_STEP_LAST = 2
_STEP_FIRST = 0


def _resize_flatten_observation(image_hwc: np.ndarray, out_size: int, cv2: Any) -> np.ndarray:
    """uint8 (H,W,3) -> float32 (out_size*out_size*3,) in [0,1]."""
    if cv2 is not None:
        small = cv2.resize(image_hwc, (out_size, out_size), interpolation=cv2.INTER_AREA)
    else:
        h, w = image_hwc.shape[:2]
        if h >= out_size and w >= out_size:
            y0, x0 = (h - out_size) // 2, (w - out_size) // 2
            small = image_hwc[y0 : y0 + out_size, x0 : x0 + out_size]
        else:
            small = np.asarray(image_hwc, dtype=np.float32)
            return small.reshape(-1) / 255.0
    return small.reshape(-1).astype(np.float32) / 255.0


def _traj_from_rgb_hwc(
    image_hwc: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    terminals: np.ndarray,
    obs_downsample: int,
    store_images: bool,
    cv2: Any,
) -> Dict[str, Any]:
    """Shared builder: image (T,H,W,3) uint8 -> trajectory dict."""
    if image_hwc.ndim != 4 or image_hwc.shape[-1] != 3:
        raise ValueError(f"Expected image (T,H,W,3), got shape {image_hwc.shape}")
    action = np.asarray(action, dtype=np.float32)
    reward = np.asarray(reward, dtype=np.float32).reshape(-1)
    terminals = np.asarray(terminals, dtype=np.float32).reshape(-1)
    n = min(
        int(action.shape[0]),
        int(reward.shape[0]),
        int(image_hwc.shape[0]),
        int(terminals.shape[0]),
    )
    if n < 2:
        raise ValueError(f"Episode too short after align (n={n})")
    action = action[:n]
    reward = reward[:n]
    image_hwc = np.asarray(image_hwc[:n], dtype=np.uint8)
    terminals = terminals[:n]

    observations = np.stack(
        [_resize_flatten_observation(image_hwc[t], obs_downsample, cv2) for t in range(n)],
        axis=0,
    )
    out: Dict[str, Any] = {
        "observations": observations.astype(np.float32),
        "actions": action.astype(np.float32),
        "rewards": reward.astype(np.float32),
        "terminals": terminals.astype(np.float32),
    }
    if store_images:
        out["images"] = [image_hwc]
    return out


def _npz_to_trajectory(
    episode: Dict[str, np.ndarray],
    obs_downsample: int,
    store_images: bool,
    cv2: Any,
) -> Dict[str, np.ndarray]:
    """Build one trajectory dict compatible with ``get_icl_trajectory_dataset`` / ICL dataset classes."""
    image = np.asarray(episode["image"])
    action = np.asarray(episode["action"], dtype=np.float32)
    reward = np.asarray(episode["reward"], dtype=np.float32).reshape(-1)
    n = min(int(action.shape[0]), int(reward.shape[0]), int(image.shape[0]))
    discount = episode.get("discount")
    if discount is not None:
        discount = np.asarray(discount, dtype=np.float32).reshape(-1)[:n]
    is_term = episode.get("is_terminal")
    if is_term is not None:
        terminals = np.asarray(is_term, dtype=np.float32).reshape(-1)[:n]
    elif discount is not None:
        terminals = (discount == 0.0).astype(np.float32)
    else:
        terminals = np.zeros(n, dtype=np.float32)
        if n > 0:
            terminals[-1] = 1.0
    return _traj_from_rgb_hwc(
        image[:n],
        action[:n],
        reward[:n],
        terminals,
        obs_downsample,
        store_images,
        cv2,
    )


def _obs_thwc_from_drq_chw(obs: np.ndarray) -> np.ndarray:
    """DrQ offline (T,C,H,W) uint8 -> (T,H,W,3) using the latest RGB stack (last 3 channels)."""
    if obs.ndim != 4:
        raise ValueError(f"Expected observation (T,C,H,W), got {obs.shape}")
    _t, c, _h, _w = obs.shape
    if c < 3 or c % 3 != 0:
        raise ValueError(f"Expected C divisible by 3 (stacked frames), got C={c}")
    rgb = obs[:, -3:, :, :]
    return np.transpose(rgb, (0, 2, 3, 1)).astype(np.uint8, copy=False)


def _hdf5_episode_ranges(step_type: np.ndarray) -> List[Tuple[int, int]]:
    """Split flat buffer on FIRST (0); same convention as v-d4rl drqbc/utils.py."""
    st = np.asarray(step_type, dtype=np.int64).reshape(-1)
    n = st.shape[0]
    starts = [i for i in range(n) if int(st[i]) == _STEP_FIRST]
    if not starts:
        starts = [0]
    ranges: List[Tuple[int, int]] = []
    for j, s in enumerate(starts):
        e = starts[j + 1] if j + 1 < len(starts) else n
        if e - s >= 2:
            ranges.append((s, e))
    return ranges


def _load_vd4rl_hdf5_trajectories(
    root: Path,
    *,
    max_episodes: Optional[int],
    obs_downsample: int,
    store_images: bool,
    shuffle: bool,
    seed: int,
    cv2: Any,
) -> Tuple[List[Dict[str, np.ndarray]], List[List[Dict[str, np.ndarray]]]]:
    try:
        import h5py
    except ImportError:
        log.error(
            "V-D4RL *.hdf5 requires h5py (84px DrQ format). "
            "Install: uv pip install h5py  (or: uv sync --extra icrt)"
        )
        return [], []

    paths = sorted(root.glob("*.hdf5"))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(paths)

    trajectories: List[Dict[str, np.ndarray]] = []
    for path in paths:
        if max_episodes is not None and len(trajectories) >= max_episodes:
            break
        try:
            with h5py.File(path, "r") as f:
                keys = set(f.keys())
                need = {"observation", "action", "reward", "discount", "step_type"}
                if not need.issubset(keys):
                    log.warning(
                        "Skip hdf5 {}: missing keys (have {}, need {})",
                        path.name,
                        sorted(keys),
                        sorted(need),
                    )
                    continue
                data = {k: np.asarray(f[k][:]) for k in need}
        except Exception as e:
            log.warning("Skip hdf5 {}: {}", path.name, e)
            continue

        n = int(data["reward"].shape[0])
        if any(int(arr.shape[0]) != n for arr in data.values()):
            log.warning("Skip hdf5 {}: length mismatch across keys", path.name)
            continue
        try:
            image_hwc_full = _obs_thwc_from_drq_chw(data["observation"])
        except ValueError as e:
            log.warning("Skip hdf5 {}: {}", path.name, e)
            continue

        for s, e in _hdf5_episode_ranges(data["step_type"]):
            if max_episodes is not None and len(trajectories) >= max_episodes:
                break
            try:
                st_seg = data["step_type"][s:e]
                terminals = (np.asarray(st_seg, dtype=np.int64).reshape(-1) == _STEP_LAST).astype(
                    np.float32
                )
                if float(terminals.sum()) < 1.0:
                    dis_seg = np.asarray(data["discount"][s:e], dtype=np.float32).reshape(-1)
                    terminals = (dis_seg == 0.0).astype(np.float32)
                if float(terminals.sum()) < 1.0 and (e - s) > 0:
                    terminals = np.zeros(e - s, dtype=np.float32)
                    terminals[-1] = 1.0
                traj = _traj_from_rgb_hwc(
                    image_hwc_full[s:e],
                    data["action"][s:e],
                    data["reward"][s:e],
                    terminals,
                    obs_downsample,
                    store_images,
                    cv2,
                )
                trajectories.append(traj)
            except Exception as ex:
                log.warning("Skip episode slice [{}:{}] in {}: {}", s, e, path.name, ex)

    if not trajectories:
        return [], []
    sorted_pool = sort_trajectories_by_return(trajectories, ascending=False)
    prompt_per_task = [sorted_pool]
    log.info(
        "Loaded {} V-D4RL trajectories from {} hdf5 files under {} (obs_downsample={}, images={})",
        len(trajectories),
        len(paths),
        root,
        obs_downsample,
        store_images,
    )
    return trajectories, prompt_per_task


def load_vd4rl_npz_trajectories(
    npz_dir: str,
    *,
    max_episodes: Optional[int] = None,
    obs_downsample: int = 16,
    store_images: bool = True,
    shuffle: bool = False,
    seed: int = 0,
) -> Tuple[List[Dict[str, np.ndarray]], List[List[Dict[str, np.ndarray]]]]:
    """
    Load V-D4RL episodes from a leaf directory (``suite/task/split/pixel_size``).

    Prefers ``*.npz`` (64px / DreamerV2). If there are no npz files, loads ``*.hdf5``
    shards (84px / DrQ-v2), splitting each file on ``step_type == FIRST`` like upstream
    ``drqbc/utils.py``.

    Returns (trajectories, prompt_per_task): one task, context pool sorted by return (desc).
    """
    root = Path(npz_dir).expanduser().resolve()
    if not root.is_dir():
        log.error("V-D4RL data directory does not exist: {}", root)
        return [], []

    try:
        import cv2
    except ImportError:
        cv2 = None
        log.warning(
            "opencv-python not installed; V-D4RL observations use coarse center-crop "
            "fallback instead of resize. Install opencv for better quality."
        )

    npz_paths = sorted(root.glob("*.npz"))
    h5_paths = sorted(root.glob("*.hdf5"))
    if npz_paths and h5_paths:
        log.warning(
            "V-D4RL directory has both *.npz and *.hdf5; loading npz only. "
            "Use a directory with a single format, or remove one set."
        )

    if npz_paths:
        paths: List[Path] = list(npz_paths)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(paths)
        trajectories: List[Dict[str, np.ndarray]] = []
        for i, p in enumerate(paths):
            if max_episodes is not None and i >= max_episodes:
                break
            try:
                with p.open("rb") as f:
                    raw = np.load(f, allow_pickle=False)
                    ep = {k: raw[k] for k in raw.files}
                traj = _npz_to_trajectory(ep, obs_downsample, store_images, cv2)
                trajectories.append(traj)
            except Exception as e:
                log.warning("Skip bad npz {}: {}", p.name, e)
        if not trajectories:
            return [], []
        sorted_pool = sort_trajectories_by_return(trajectories, ascending=False)
        prompt_per_task = [sorted_pool]
        log.info(
            "Loaded {} V-D4RL trajectories from {} (npz, obs_downsample={}, images={})",
            len(trajectories),
            root,
            obs_downsample,
            store_images,
        )
        return trajectories, prompt_per_task

    if h5_paths:
        return _load_vd4rl_hdf5_trajectories(
            root,
            max_episodes=max_episodes,
            obs_downsample=obs_downsample,
            store_images=store_images,
            shuffle=shuffle,
            seed=seed,
            cv2=cv2,
        )

    log.error(
        "V-D4RL directory has no *.npz or *.hdf5 files: {} "
        "(expected upstream layout under suite/task/split/{{64px|84px}})",
        root,
    )
    return [], []
