"""Eval rollout helpers: seeds, metrics, ragged arrays, video grids."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def resolve_per_trial_eval_target_returns(
    eval_num_trials: int,
    eval_target_return: Optional[float],
    eval_target_returns: Optional[List[float]],
) -> List[Optional[float]]:
    K = max(0, int(eval_num_trials))
    if K == 0:
        return []
    if eval_target_returns is not None:
        lst = [float(x) for x in list(eval_target_returns)]
        if len(lst) != K:
            raise ValueError(
                "eval_target_returns length must equal eval_num_trials: "
                f"got len={len(lst)}, eval_num_trials={K}"
            )
        return lst
    if eval_target_return is not None:
        G = float(eval_target_return)
        if K > 1:
            return [G * (i + 1) / K for i in range(K)]
        return [G]
    return [None] * K


def eval_episode_reset_seed(
    *,
    step: int,
    session_rep: int,
    trial: int,
    n_trials_in_session: int,
    eval_scene_seeds: Optional[List[int]],
    randomize_scene_between_trials: bool,
) -> int:
    seeds: Optional[List[int]] = None
    if eval_scene_seeds:
        seeds = [int(x) for x in eval_scene_seeds]
        if len(seeds) == 0:
            seeds = None
    nt = max(1, int(n_trials_in_session))
    rep = int(session_rep)
    tr = int(trial)

    if seeds is not None:
        L = len(seeds)
        if nt > 1 and randomize_scene_between_trials:
            idx = (rep * nt + tr) % L
        else:
            idx = rep % L
        return seeds[idx]

    if nt > 1:
        if randomize_scene_between_trials:
            return int(step + rep * 100_000 + tr)
        return int(step + rep * 100_000)
    return int(step + rep)


def default_eval_scene_seeds(
    *,
    eval_context_mode: str,
    num_rollouts: int,
    eval_num_trials: int,
    randomize_scene_between_trials: bool,
    seed_base: int,
) -> List[int]:
    mode = str(eval_context_mode).strip().lower()
    n_sessions = max(0, int(num_rollouts))
    n_trials = max(1, int(eval_num_trials)) if mode == "zero_shot_adaptation" else 1
    if n_sessions <= 0:
        b = 9_000_000 + (abs(int(seed_base)) % 1_000_000)
        return [b]
    if randomize_scene_between_trials and mode == "zero_shot_adaptation":
        n = n_sessions * n_trials
    else:
        n = n_sessions
    b = 9_000_000 + (abs(int(seed_base)) % 1_000_000)
    return [b + i for i in range(max(1, n))]


def action_prediction_stats_from_rollouts(rollouts_actions: List[np.ndarray]) -> Dict[str, float]:
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


def pad_ragged_1d(arrays: List[np.ndarray], fill: float = float("nan")) -> np.ndarray:
    if not arrays:
        return np.zeros((0, 0), dtype=np.float64)
    T = max(int(np.asarray(a).shape[0]) for a in arrays)
    out = np.full((len(arrays), T), fill, dtype=np.float64)
    for i, a in enumerate(arrays):
        v = np.asarray(a, dtype=np.float64).reshape(-1)
        n = int(v.shape[0])
        out[i, :n] = v
    return out


def grid_layout_dims(num_rollouts: int, n_trials: int) -> Tuple[int, int]:
    if n_trials > 1:
        return max(0, int(num_rollouts)), max(0, int(n_trials))
    n = max(0, int(num_rollouts))
    if n <= 0:
        return 0, 0
    r = int(math.ceil(math.sqrt(n)))
    c = int(math.ceil(n / r))
    return r, c


def pack_flat_clips_to_grid(
    clips: List[Optional[List[np.ndarray]]],
    n_rows: int,
    n_cols: int,
) -> List[List[Optional[List[np.ndarray]]]]:
    grid: List[List[Optional[List[np.ndarray]]]] = [[None] * n_cols for _ in range(n_rows)]
    for i, cl in enumerate(clips):
        if i >= n_rows * n_cols:
            break
        r, c = divmod(i, n_cols)
        grid[r][c] = cl
    return grid


def _resize_frame_u8(fr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    fr = np.asarray(fr)
    if fr.dtype != np.uint8:
        fr = np.clip(fr, 0, 255).astype(np.uint8)
    if fr.ndim == 2:
        fr = np.stack([fr, fr, fr], axis=-1)
    if fr.shape[0] == target_h and fr.shape[1] == target_w:
        return fr
    try:
        import cv2

        return cv2.resize(fr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    except Exception:
        from PIL import Image

        pil = Image.fromarray(fr)
        pil = pil.resize((target_w, target_h), Image.Resampling.BILINEAR)
        return np.asarray(pil).astype(np.uint8)


def compose_grid_frames_sequence(
    grid: List[List[Optional[List[np.ndarray]]]],
    *,
    n_rows: int,
    n_cols: int,
) -> List[np.ndarray]:
    target_h, target_w = 0, 0
    max_t = 0
    for r in range(n_rows):
        for c in range(n_cols):
            clip = grid[r][c] if r < len(grid) and c < len(grid[r]) else None
            if not clip:
                continue
            max_t = max(max_t, len(clip))
            f0 = np.asarray(clip[0])
            target_h = max(target_h, int(f0.shape[0]))
            target_w = max(target_w, int(f0.shape[1]))
    if max_t == 0 or target_h == 0:
        return []

    def cell(t: int, clip: Optional[List[np.ndarray]]) -> np.ndarray:
        if not clip:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)
        ti = min(t, len(clip) - 1)
        return _resize_frame_u8(clip[ti], target_h, target_w)

    out: List[np.ndarray] = []
    for t in range(max_t):
        rows = []
        for r in range(n_rows):
            cols = [
                cell(t, grid[r][c] if r < len(grid) and c < len(grid[r]) else None)
                for c in range(n_cols)
            ]
            rows.append(np.hstack(cols))
        out.append(np.vstack(rows))
    return out


def write_frames_video(
    video_folder: Path, filename: str, frames: List[np.ndarray], fps: int = 20
) -> None:
    if not frames:
        return
    import imageio

    path = video_folder / filename
    writer = imageio.get_writer(str(path), fps=fps)
    for f in frames:
        writer.append_data(np.asarray(f))
    writer.close()
