"""
Load V-D4RL Offline DreamerV2-style *.npz episodes (64px pixel datasets).

Dataset layout (see https://github.com/conglu1997/v-d4rl):
  <root>/main/<task>/<split>/64px/*.npz

Each npz typically contains: image, action, reward, discount; optional is_terminal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger as log

from src.data.trajectories import sort_trajectories_by_return


def _resize_flatten_observation(
    image_hwc: np.ndarray, out_size: int, cv2: Any
) -> np.ndarray:
    """uint8 (H,W,3) -> float32 (out_size*out_size*3,) in [0,1]."""
    if cv2 is not None:
        small = cv2.resize(
            image_hwc, (out_size, out_size), interpolation=cv2.INTER_AREA
        )
    else:
        h, w = image_hwc.shape[:2]
        if h >= out_size and w >= out_size:
            y0, x0 = (h - out_size) // 2, (w - out_size) // 2
            small = image_hwc[y0 : y0 + out_size, x0 : x0 + out_size]
        else:
            small = np.asarray(image_hwc, dtype=np.float32)
            return small.reshape(-1) / 255.0
    return small.reshape(-1).astype(np.float32) / 255.0


def _npz_to_trajectory(
    episode: Dict[str, np.ndarray],
    obs_downsample: int,
    store_images: bool,
    cv2: Any,
) -> Dict[str, np.ndarray]:
    """Build one trajectory dict compatible with ICLTrajectoryDataset."""
    action = np.asarray(episode["action"], dtype=np.float32)
    reward = np.asarray(episode["reward"], dtype=np.float32).reshape(-1)
    image = np.asarray(episode["image"])
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError(f"Expected image (T,H,W,3), got shape {image.shape}")

    n = min(int(action.shape[0]), int(reward.shape[0]), int(image.shape[0]))
    if n < 2:
        raise ValueError(f"Episode too short after align (n={n})")

    action = action[:n]
    reward = reward[:n]
    image = image[:n]

    observations = np.stack(
        [_resize_flatten_observation(image[t], obs_downsample, cv2) for t in range(n)],
        axis=0,
    )

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

    out: Dict[str, Any] = {
        "observations": observations.astype(np.float32),
        "actions": action.astype(np.float32),
        "rewards": reward.astype(np.float32),
        "terminals": terminals.astype(np.float32),
    }
    if store_images:
        out["images"] = [np.asarray(image, dtype=np.uint8)]
    return out


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
    Load all *.npz episodes from a V-D4RL 64px (or compatible) directory.

    Returns (trajectories, prompt_per_task) with the same convention as HalfCheetah:
    one task, context pool = all trajectories sorted by return (descending).
    """
    root = Path(npz_dir).expanduser().resolve()
    if not root.is_dir():
        log.error("V-D4RL npz directory does not exist: {}", root)
        return [], []

    try:
        import cv2
    except ImportError:
        cv2 = None
        log.warning(
            "opencv-python not installed; V-D4RL observations use coarse center-crop "
            "fallback instead of resize. Install opencv for better quality."
        )

    paths = sorted(root.glob("*.npz"))
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
        "Loaded {} V-D4RL trajectories from {} (obs_downsample={}, images={})",
        len(trajectories),
        root,
        obs_downsample,
        store_images,
    )
    return trajectories, prompt_per_task
