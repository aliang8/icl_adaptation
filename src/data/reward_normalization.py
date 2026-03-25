"""
Offline reward normalization for trajectory dicts (keys: rewards, ...).

Modes:
  - none: no change
  - constant: rewards /= reward_norm_constant (scale to smaller magnitude)
  - standardize: global (r - mean) / (std + eps) over all steps in the dataset
  - minmax: global (r - r_min) / (r_max - r_min + eps) to [0, 1]

Returns a stats dict suitable for JSON (reproducibility / eval alignment).
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

StatsDict = Dict[str, Any]


def _all_step_rewards(trajectories: List[Dict[str, np.ndarray]]) -> np.ndarray:
    parts = [np.asarray(t["rewards"], dtype=np.float64).ravel() for t in trajectories]
    return np.concatenate(parts)


def compute_standardize_stats(
    trajectories: List[Dict[str, np.ndarray]], epsilon: float
) -> StatsDict:
    x = _all_step_rewards(trajectories)
    mean = float(x.mean())
    std = float(x.std())
    return {
        "mode": "standardize",
        "mean": mean,
        "std": std,
        "epsilon": float(epsilon),
        "n_steps": int(x.size),
    }


def compute_minmax_stats(trajectories: List[Dict[str, np.ndarray]], epsilon: float) -> StatsDict:
    x = _all_step_rewards(trajectories)
    r_min = float(x.min())
    r_max = float(x.max())
    return {
        "mode": "minmax",
        "r_min": r_min,
        "r_max": r_max,
        "epsilon": float(epsilon),
        "n_steps": int(x.size),
    }


def apply_constant(
    trajectories: List[Dict[str, np.ndarray]],
    divisor: float,
    *,
    in_place: bool = True,
) -> Tuple[List[Dict[str, np.ndarray]], StatsDict]:
    if divisor == 0:
        raise ValueError("reward_norm_constant (divisor) must be non-zero")
    stats: StatsDict = {"mode": "constant", "divisor": float(divisor)}
    out = trajectories if in_place else deepcopy(trajectories)
    for t in out:
        r = np.asarray(t["rewards"], dtype=np.float32)
        if not in_place:
            r = r.copy()
        t["rewards"] = (r / float(divisor)).astype(np.float32)
    return out, stats


def apply_standardize(
    trajectories: List[Dict[str, np.ndarray]],
    epsilon: float,
    stats: Optional[StatsDict] = None,
    *,
    in_place: bool = True,
) -> Tuple[List[Dict[str, np.ndarray]], StatsDict]:
    out = trajectories if in_place else deepcopy(trajectories)
    if stats is None:
        stats = compute_standardize_stats(out, epsilon)
    mean = float(stats["mean"])
    std = float(stats["std"])
    denom = std + float(epsilon)
    for t in out:
        r = np.asarray(t["rewards"], dtype=np.float32)
        if not in_place:
            r = r.copy()
        t["rewards"] = ((r - mean) / denom).astype(np.float32)
    return out, stats


def apply_minmax(
    trajectories: List[Dict[str, np.ndarray]],
    epsilon: float,
    stats: Optional[StatsDict] = None,
    *,
    in_place: bool = True,
) -> Tuple[List[Dict[str, np.ndarray]], StatsDict]:
    out = trajectories if in_place else deepcopy(trajectories)
    if stats is None:
        stats = compute_minmax_stats(out, epsilon)
    r_min = float(stats["r_min"])
    r_max = float(stats["r_max"])
    denom = r_max - r_min + float(epsilon)
    for t in out:
        r = np.asarray(t["rewards"], dtype=np.float32)
        if not in_place:
            r = r.copy()
        t["rewards"] = ((r - r_min) / denom).astype(np.float32)
    return out, stats


def initial_rtg_token(
    return_scale: float,
    *,
    target_future_normalized_return_sum: Optional[float] = None,
) -> float:
    """
    Starting RTG for online rollouts, same space as training targets:
    ``discount_cumsum(trajectory[\"rewards\"]) / return_scale`` with **already-normalized**
    per-step rewards.

    Conditioning on total future **normalized** return ``G`` uses token ``G / return_scale``.
    Default ``G = return_scale`` gives ``1.0`` (standard DT-style eval). ``return_scale`` itself
    is not passed through ``normalize_reward_scalar`` — only per-step env rewards are.
    """
    s = float(return_scale)
    if s == 0.0:
        raise ValueError("return_scale must be non-zero")
    g = s if target_future_normalized_return_sum is None else float(target_future_normalized_return_sum)
    return g / s


def normalize_reward_scalar(
    reward: float,
    mode: str,
    *,
    reward_norm_constant: float = 1.0,
    epsilon: float = 1e-8,
    stats: Optional[StatsDict] = None,
) -> float:
    """Normalize one scalar reward using the same rules as dataset normalization."""
    m = (mode or "none").strip().lower()
    r = float(reward)
    if m in ("none", "", "off", "false"):
        return r
    if m == "constant":
        if reward_norm_constant == 0:
            raise ValueError("reward_norm_constant (divisor) must be non-zero")
        return r / float(reward_norm_constant)
    if m in ("standardize", "dataset_std", "zscore"):
        if stats is None:
            raise ValueError("standardize mode needs stats dict (mean/std)")
        mean = float(stats["mean"])
        std = float(stats["std"])
        return (r - mean) / (std + float(epsilon))
    if m in ("minmax", "dataset_minmax"):
        if stats is None:
            raise ValueError("minmax mode needs stats dict (r_min/r_max)")
        r_min = float(stats["r_min"])
        r_max = float(stats["r_max"])
        return (r - r_min) / (r_max - r_min + float(epsilon))
    raise ValueError(
        f"Unknown reward_normalization mode {mode!r}. Use: none, constant, standardize, minmax."
    )


def apply_reward_normalization(
    trajectories: List[Dict[str, np.ndarray]],
    mode: str,
    *,
    reward_norm_constant: float = 1.0,
    epsilon: float = 1e-8,
    stats: Optional[StatsDict] = None,
    in_place: bool = True,
) -> Tuple[List[Dict[str, np.ndarray]], StatsDict]:
    """
    Apply normalization. `stats` is only used for standardize/minmax to apply fixed offline stats.
    """
    m = (mode or "none").strip().lower()
    if m in ("none", "", "off", "false"):
        return trajectories, {"mode": "none"}
    if m == "constant":
        return apply_constant(trajectories, reward_norm_constant, in_place=in_place)
    if m in ("standardize", "dataset_std", "zscore"):
        return apply_standardize(trajectories, epsilon, stats=stats, in_place=in_place)
    if m in ("minmax", "dataset_minmax"):
        return apply_minmax(trajectories, epsilon, stats=stats, in_place=in_place)
    raise ValueError(
        f"Unknown reward_normalization mode {mode!r}. Use: none, constant, standardize, minmax."
    )


def save_stats(path: Path, stats: StatsDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def load_stats(path: Path) -> StatsDict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def default_stats_path_for_pkl(pkl_path: Path) -> Path:
    """Sidecar JSON next to trajectories.pkl."""
    return pkl_path.parent / "reward_normalization_stats.json"
