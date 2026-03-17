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


def _prompt_segment_from_traj(
    traj: Dict[str, np.ndarray],
    state_mean: np.ndarray,
    state_std: np.ndarray,
    scale: float,
    max_prompt_trajectory_length: Optional[int],
    state_dim: int,
    act_dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One trajectory as prompt segment (last max_prompt_trajectory_length steps)."""
    T = traj["rewards"].shape[0]
    if T == 0:
        raise ValueError("Empty trajectory")
    cap = max_prompt_trajectory_length
    if cap is not None and T > cap:
        start = T - cap
        T = cap
    else:
        start = 0
    obs = np.asarray(traj["observations"], dtype=np.float32)
    act = np.asarray(traj["actions"], dtype=np.float32)
    rew = np.asarray(traj["rewards"], dtype=np.float32)
    ps = (obs[start : start + T] - state_mean) / state_std
    pa = act[start : start + T]
    pr = rew[start : start + T].reshape(-1, 1)
    prtg = discount_cumsum(rew[start:], gamma=1.0)[:T].reshape(-1, 1) / scale
    pts = np.arange(start, start + T, dtype=np.float32)
    pm = np.ones(T, dtype=np.float32)
    return ps, pa, pr, prtg, pts, pm


def _pad_or_trim_prompt(
    ps: np.ndarray,
    pa: np.ndarray,
    pr: np.ndarray,
    prtg: np.ndarray,
    pts: np.ndarray,
    pm: np.ndarray,
    total_plen: int,
    state_dim: int,
    act_dim: int,
    take_last: bool,
) -> Tuple[np.ndarray, ...]:
    if ps.shape[0] >= total_plen:
        if take_last:
            ps, pa, pr, prtg, pts, pm = (
                ps[-total_plen:],
                pa[-total_plen:],
                pr[-total_plen:],
                prtg[-total_plen:],
                pts[-total_plen:],
                pm[-total_plen:],
            )
        else:
            ps, pa, pr, prtg, pts, pm = (
                ps[:total_plen],
                pa[:total_plen],
                pr[:total_plen],
                prtg[:total_plen],
                pts[:total_plen],
                pm[:total_plen],
            )
    else:
        pad_len = total_plen - ps.shape[0]
        ps = np.concatenate([np.zeros((pad_len, state_dim)), ps], axis=0)
        pa = np.concatenate([np.ones((pad_len, act_dim)) * -10.0, pa], axis=0)
        pr = np.concatenate([np.zeros((pad_len, 1)), pr], axis=0)
        prtg = np.concatenate([np.zeros((pad_len, 1)), prtg], axis=0)
        pts = np.concatenate([np.zeros(pad_len), pts], axis=0)
        pm = np.concatenate([np.zeros(pad_len), pm], axis=0)
    return ps, pa, pr, prtg, pts, pm


def build_prompt_tuple(
    trajectories: List[Dict[str, np.ndarray]],
    state_mean: np.ndarray,
    state_std: np.ndarray,
    total_prompt_len: int,
    max_prompt_trajectory_length: Optional[int],
    state_dim: int,
    act_dim: int,
    scale: float,
    device: torch.device,
    sort_ascending: bool = True,
    trajectory_returns: Optional[List[float]] = None,
) -> Optional[Tuple[torch.Tensor, ...]]:
    """
    Build the 6-tuple (prompt_states, prompt_actions, prompt_rewards, prompt_rtg, prompt_timesteps, prompt_mask)
    from K trajectories, sorted by return (ascending = worst first, like training context).
    If trajectory_returns is provided (same length as trajectories), sort by those instead of sum(rewards).
    Returns None if trajectories is empty.
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
    segs_ps, segs_pa, segs_pr, segs_prtg, segs_pts, segs_pm = [], [], [], [], [], []
    for traj in sorted_trajs:
        ps, pa, pr, prtg, pts, pm = _prompt_segment_from_traj(
            traj,
            state_mean,
            state_std,
            scale,
            max_prompt_trajectory_length,
            state_dim,
            act_dim,
        )
        segs_ps.append(ps)
        segs_pa.append(pa)
        segs_pr.append(pr)
        segs_prtg.append(prtg)
        segs_pts.append(pts)
        segs_pm.append(pm)
    ps = np.concatenate(segs_ps, axis=0)
    pa = np.concatenate(segs_pa, axis=0)
    pr = np.concatenate(segs_pr, axis=0)
    prtg = np.concatenate(segs_prtg, axis=0)
    pts = np.concatenate(segs_pts, axis=0)
    pm = np.concatenate(segs_pm, axis=0)
    ps, pa, pr, prtg, pts, pm = _pad_or_trim_prompt(
        ps, pa, pr, prtg, pts, pm, total_prompt_len, state_dim, act_dim, take_last=True
    )
    return (
        torch.from_numpy(ps).float().unsqueeze(0).to(device),
        torch.from_numpy(pa).float().unsqueeze(0).to(device),
        torch.from_numpy(pr).float().unsqueeze(0).to(device),
        torch.from_numpy(prtg).float().unsqueeze(0).to(device),
        torch.from_numpy(pts).float().unsqueeze(0).to(device),
        torch.from_numpy(pm).float().unsqueeze(0).to(device),
    )
