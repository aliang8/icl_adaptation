"""
Eval context: build prompt tuple from K trajectories (sorted by return) for prompt mode
or zero_shot_adaptation. Same format as training (state, action, rtg, timestep, mask).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.data.trajectories import discount_cumsum, sort_trajectories_by_return


def _subsample_indices(length: int, cap: Optional[int], strategy: str) -> np.ndarray:
    """Select ordered indices from [0, length) according to strategy."""
    if length <= 0:
        return np.zeros(0, dtype=np.int64)
    if strategy == "none":
        return np.arange(length, dtype=np.int64)
    if cap is None or cap <= 0 or length <= cap:
        return np.arange(length, dtype=np.int64)
    if strategy == "last":
        return np.arange(length - cap, length, dtype=np.int64)
    if strategy == "uniform":
        return np.linspace(0, length - 1, num=cap, dtype=np.int64)
    if strategy == "random":
        idx = np.sort(np.random.choice(length, size=cap, replace=False))
        return idx.astype(np.int64)
    raise ValueError(
        f"Unsupported context_subsample_strategy='{strategy}'. "
        "Use one of: none, last, uniform, random."
    )


def _prompt_segment_from_traj(
    traj: Dict[str, np.ndarray],
    state_mean: np.ndarray,
    state_std: np.ndarray,
    rtg_scale: float,
    max_prompt_trajectory_length: Optional[int],
    context_subsample_strategy: str,
    state_dim: int,
    act_dim: int,
    trial_idx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One trajectory as prompt segment with optional per-trajectory subsampling."""
    T = traj["rewards"].shape[0]
    if T == 0:
        raise ValueError("Empty trajectory")
    obs = np.asarray(traj["observations"], dtype=np.float32)
    act = np.asarray(traj["actions"], dtype=np.float32)
    rew = np.asarray(traj["rewards"], dtype=np.float32)
    idx = _subsample_indices(T, max_prompt_trajectory_length, context_subsample_strategy)
    full_rtg = discount_cumsum(rew, gamma=1.0).reshape(-1, 1)
    ps = (obs[idx] - state_mean) / state_std
    pa = act[idx]
    pr = rew[idx].reshape(-1, 1)
    prtg = full_rtg[idx] / rtg_scale
    pts = idx.astype(np.float32)
    T = idx.shape[0]
    pm = np.ones(T, dtype=np.float32)
    ptrial = np.full(T, trial_idx, dtype=np.float32)
    return ps, pa, pr, prtg, pts, pm, ptrial


def _pad_or_trim_prompt(
    ps: np.ndarray,
    pa: np.ndarray,
    pr: np.ndarray,
    prtg: np.ndarray,
    pts: np.ndarray,
    pm: np.ndarray,
    ptrial: np.ndarray,
    total_plen: int,
    state_dim: int,
    act_dim: int,
    take_last: bool,
) -> Tuple[np.ndarray, ...]:
    """
    If shorter than total_plen: **left-pad** so valid prompt timesteps are **right-aligned**
    (abutting the query segment), matching training collate. If longer: trim from start (take_last)
    or from end (not take_last).
    """
    if ps.shape[0] >= total_plen:
        if take_last:
            ps, pa, pr, prtg, pts, pm, ptrial = (
                ps[-total_plen:],
                pa[-total_plen:],
                pr[-total_plen:],
                prtg[-total_plen:],
                pts[-total_plen:],
                pm[-total_plen:],
                ptrial[-total_plen:],
            )
        else:
            ps, pa, pr, prtg, pts, pm, ptrial = (
                ps[:total_plen],
                pa[:total_plen],
                pr[:total_plen],
                prtg[:total_plen],
                pts[:total_plen],
                pm[:total_plen],
                ptrial[:total_plen],
            )
    else:
        pad_len = total_plen - ps.shape[0]
        ps = np.concatenate([np.zeros((pad_len, state_dim)), ps], axis=0)
        pa = np.concatenate([np.ones((pad_len, act_dim)) * -10.0, pa], axis=0)
        pr = np.concatenate([np.zeros((pad_len, 1)), pr], axis=0)
        prtg = np.concatenate([np.zeros((pad_len, 1)), prtg], axis=0)
        pts = np.concatenate([np.zeros(pad_len), pts], axis=0)
        pm = np.concatenate([np.zeros(pad_len), pm], axis=0)
        ptrial = np.concatenate([np.zeros(pad_len, dtype=np.float32), ptrial], axis=0)
    return ps, pa, pr, prtg, pts, pm, ptrial


def build_prompt_tuple(
    trajectories: List[Dict[str, np.ndarray]],
    state_mean: np.ndarray,
    state_std: np.ndarray,
    total_prompt_len: int,
    max_prompt_trajectory_length: Optional[int],
    state_dim: int,
    act_dim: int,
    rtg_scale: float,
    device: torch.device,
    sort_ascending: bool = True,
    trajectory_returns: Optional[List[float]] = None,
    context_subsample_strategy: str = "none",
) -> Optional[Tuple[torch.Tensor, ...]]:
    """
    Build the 7-tuple (prompt_states, prompt_actions, prompt_rewards, prompt_rtg, prompt_timesteps,
    prompt_mask, prompt_trial_idx) from K trajectories, sorted by return (ascending = worst first, like training).
    If trajectory_returns is provided (same length as trajectories), sort by those instead of sum(rewards).
    Returns None if trajectories is empty.

    Uses take_last=True when trimming/padding: short prompts are **left-padded** so real tokens sit at the
    end next to the query (same as training _pad_or_trim_prompt with take_last=True).
    """
    if not trajectories:
        return None
    if trajectory_returns is not None and len(trajectory_returns) == len(trajectories):
        order = np.argsort(trajectory_returns)
        if not sort_ascending:
            order = order[::-1]
        sorted_trajs = [trajectories[i] for i in order]
    else:
        sorted_trajs = sort_trajectories_by_return(trajectories, ascending=sort_ascending)
    segs_ps, segs_pa, segs_pr, segs_prtg, segs_pts, segs_pm, segs_ptrial = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for trial_idx, traj in enumerate(sorted_trajs):
        ps, pa, pr, prtg, pts, pm, pt = _prompt_segment_from_traj(
            traj,
            state_mean,
            state_std,
            rtg_scale,
            max_prompt_trajectory_length,
            context_subsample_strategy,
            state_dim,
            act_dim,
            trial_idx,
        )
        segs_ps.append(ps)
        segs_pa.append(pa)
        segs_pr.append(pr)
        segs_prtg.append(prtg)
        segs_pts.append(pts)
        segs_pm.append(pm)
        segs_ptrial.append(pt)
    ps = np.concatenate(segs_ps, axis=0)
    pa = np.concatenate(segs_pa, axis=0)
    pr = np.concatenate(segs_pr, axis=0)
    prtg = np.concatenate(segs_prtg, axis=0)
    pts = np.concatenate(segs_pts, axis=0)
    pm = np.concatenate(segs_pm, axis=0)
    ptrial = np.concatenate(segs_ptrial, axis=0)
    ps, pa, pr, prtg, pts, pm, ptrial = _pad_or_trim_prompt(
        ps, pa, pr, prtg, pts, pm, ptrial, total_prompt_len, state_dim, act_dim, take_last=True
    )
    return (
        torch.from_numpy(ps).float().unsqueeze(0).to(device),
        torch.from_numpy(pa).float().unsqueeze(0).to(device),
        torch.from_numpy(pr).float().unsqueeze(0).to(device),
        torch.from_numpy(prtg).float().unsqueeze(0).to(device),
        torch.from_numpy(pts).float().unsqueeze(0).to(device),
        torch.from_numpy(pm).float().unsqueeze(0).to(device),
        torch.from_numpy(ptrial).float().unsqueeze(0).to(device),
    )
