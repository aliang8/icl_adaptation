"""Trajectory utilities: convert data to trajectories, sort by return (for context)."""

import random
import numpy as np
from collections import OrderedDict
from typing import List, Dict, Any, Optional


def discount_cumsum(x: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    out = np.zeros_like(x)
    out[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        out[t] = x[t] + gamma * out[t + 1]
    return out


def trajectory_return(traj: Dict[str, np.ndarray]) -> float:
    return float(np.sum(traj["rewards"]))


def sort_trajectories_by_return(
    trajectories: List[Dict[str, np.ndarray]],
    ascending: bool = False,
) -> List[Dict[str, np.ndarray]]:
    """
    Sort trajectories by episode return.
    Training: ascending (low→high) so model sees increasing progress in context.
    Inference/zero-shot: ascending (worst first) for adaptation from previous rollouts.
    """
    returns = [trajectory_return(t) for t in trajectories]
    order = np.argsort(returns)
    if not ascending:
        order = order[::-1]
    return [trajectories[i] for i in order]


def sample_context_trajectories(
    pool: List[Dict[str, np.ndarray]],
    n: int,
    ascending: bool = True,
    sampling: str = "random",
) -> List[Dict[str, np.ndarray]]:
    """
    Sample n trajectories from the same-task pool for in-context conditioning.
    Returns them sorted by return (by default ascending: low→high for increasing progress).

    sampling:
      - "random": random sample then sort by return.
      - "stratified": split pool by return into buckets, sample from each for diversity,
        then sort chosen by return (so one sample can show low/med/high in order).
    """
    if not pool or n <= 0:
        return []
    n = min(n, len(pool))
    if sampling == "stratified" and len(pool) >= n:
        returns = np.array([trajectory_return(t) for t in pool])
        order = np.argsort(returns)
        n_buckets = min(n, 3)
        bucket_size = max(1, len(order) // n_buckets)
        chosen_idx = []
        per_bucket = (n + n_buckets - 1) // n_buckets
        for b in range(n_buckets):
            start, end = b * bucket_size, min((b + 1) * bucket_size, len(order))
            if start >= end:
                continue
            idx = order[start:end]
            k = min(per_bucket, len(idx), n - len(chosen_idx))
            if k <= 0:
                break
            chosen_idx.extend(np.random.choice(idx, size=k, replace=False).tolist())
        chosen_idx = chosen_idx[:n]
        chosen = [pool[i] for i in chosen_idx]
        return sort_trajectories_by_return(chosen, ascending=ascending)
    # random
    chosen = random.sample(pool, n)
    return sort_trajectories_by_return(chosen, ascending=ascending)


def convert_data_to_trajectories(
    data: Dict[str, np.ndarray],
    max_episode_steps: int,
    max_trajectories: Optional[int] = None,
) -> List[Dict[str, np.ndarray]]:
    """
    Convert flat data (observations, actions, rewards, terminals) into list of trajectories.
    Splits on terminals or every max_episode_steps.
    """
    trajectories = []
    start = 0
    n = len(data["terminals"]) if "terminals" in data else len(data.get("dones", data["rewards"]))
    term_key = "terminals" if "terminals" in data else "dones"
    for i in range(n):
        if (i + 1) % max_episode_steps == 0 or (i + 1 < n and data[term_key][i]):
            traj = OrderedDict()
            for key in [
                "observations",
                "actions",
                "rewards",
                "next_observations",
                "terminals",
                "masks",
            ]:
                if key in data:
                    traj[key] = data[key][start : i + 1]
            if "terminals" not in traj and "dones" in data:
                traj["terminals"] = data["dones"][start : i + 1]
            trajectories.append(traj)
            start = i + 1
            if max_trajectories and len(trajectories) >= max_trajectories:
                break
    return trajectories
