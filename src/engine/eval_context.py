"""
Eval context: build prompt tuples for prompt mode or zero_shot_adaptation.

Uses the same NumPy prompt layout as ``dataset.py`` (state, action, reward, RTG, timestep,
mask, trial id). Padding / trimming matches training:

- ``context_style=subsampled``: same as ``SubsampledICLTrajectoryDataset._build_prompt``
  (``_pad_or_trim_prompt`` with ``take_last=False``).
- ``context_style=full_trajectory``: per-demo pad/trim to ``max_episode_steps``, then global cap
  (``_pad_or_trim_prompt`` with ``take_last=True``), same as ``FullTrajectoryICLTrajectoryDataset``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.data.dataset import (
    _pad_or_trim_prompt,
    _subsample_indices,
    icl_prompt_segment_full_trajectory,
)
from src.data.trajectories import discount_cumsum, sort_trajectories_by_return


def _prompt_segment_subsampled_eval(
    traj: Dict[str, np.ndarray],
    state_mean: np.ndarray,
    state_std: np.ndarray,
    rtg_scale: float,
    max_prompt_trajectory_length: Optional[int],
    context_subsample_strategy: str,
    trial_idx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One trajectory as a prompt segment (subsampled path; matches eval's prior behavior)."""
    tlen = int(traj["rewards"].shape[0])
    if tlen == 0:
        raise ValueError("Empty trajectory")
    obs = np.asarray(traj["observations"], dtype=np.float32)
    act = np.asarray(traj["actions"], dtype=np.float32)
    rew = np.asarray(traj["rewards"], dtype=np.float32)
    idx = _subsample_indices(tlen, max_prompt_trajectory_length, context_subsample_strategy)
    full_rtg = discount_cumsum(rew, gamma=1.0).reshape(-1, 1)
    ps = (obs[idx] - state_mean) / state_std
    pa = act[idx]
    pr = rew[idx].reshape(-1, 1)
    prtg = full_rtg[idx] / rtg_scale
    pts = idx.astype(np.float32)
    n = int(idx.shape[0])
    pm = np.ones(n, dtype=np.float32)
    ptrial = np.full(n, float(trial_idx + 1), dtype=np.float32)
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
    *,
    context_style: str = "subsampled",
    max_episode_steps: int = 0,
) -> Optional[Tuple[torch.Tensor, ...]]:
    """
    Build the 7-tuple (prompt_states, prompt_actions, prompt_rewards, prompt_rtg, prompt_timesteps,
    prompt_mask, prompt_trial_idx) from K trajectories, sorted by return (ascending = worst first,
    like training). Trial indices are **1-based** per demo (**1..K**); **0** is padding.
    Returns None if ``trajectories`` is empty.

    ``context_style`` and ``max_episode_steps`` must match ``data.context_style`` and
    ``data.max_episode_steps`` used for training. For ``full_trajectory``, ``max_episode_steps``
    must be positive.
    """
    if not trajectories:
        return None

    use_full = str(context_style).strip().lower() == "full_trajectory"
    if use_full and max_episode_steps <= 0:
        raise ValueError(
            "build_prompt_tuple: context_style='full_trajectory' requires max_episode_steps > 0 "
            "(same as data.max_episode_steps during training)."
        )

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
        if use_full:
            s, a, r, rtg_, ts, m, pt = icl_prompt_segment_full_trajectory(
                traj,
                state_mean,
                state_std,
                rtg_scale,
                trial_idx,
                max_episode_steps,
                act_dim,
                max_prompt_trajectory_length,
                context_subsample_strategy,
            )
        else:
            s, a, r, rtg_, ts, m, pt = _prompt_segment_subsampled_eval(
                traj,
                state_mean,
                state_std,
                rtg_scale,
                max_prompt_trajectory_length,
                context_subsample_strategy,
                trial_idx,
            )
        segs_ps.append(s)
        segs_pa.append(a)
        segs_pr.append(r)
        segs_prtg.append(rtg_)
        segs_pts.append(ts)
        segs_pm.append(m)
        segs_ptrial.append(pt)

    ps = np.concatenate(segs_ps, axis=0)
    pa = np.concatenate(segs_pa, axis=0)
    pr = np.concatenate(segs_pr, axis=0)
    prtg = np.concatenate(segs_prtg, axis=0)
    pts = np.concatenate(segs_pts, axis=0)
    pm = np.concatenate(segs_pm, axis=0)
    ptrial = np.concatenate(segs_ptrial, axis=0)

    take_last = bool(use_full)
    ps, pa, pr, prtg, pts, pm, ptrial = _pad_or_trim_prompt(
        ps,
        pa,
        pr,
        prtg,
        pts,
        pm,
        ptrial,
        total_prompt_len,
        state_dim,
        act_dim,
        take_last=take_last,
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
