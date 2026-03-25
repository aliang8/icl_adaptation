"""Load D4RL-style or pre-downloaded HalfCheetah trajectories (mixed returns)."""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger as log

from src.data.reward_normalization import apply_reward_normalization, load_stats
from src.data.trajectories import sort_trajectories_by_return


def load_halfcheetah_trajectories(
    data_dir: str,
    env_name: str = "HalfCheetah-v2",
    data_quality: str = "medium_expert",
    max_trajectories: Optional[int] = None,
    reward_normalization: str = "none",
    reward_norm_constant: float = 1.0,
    reward_norm_epsilon: float = 1e-8,
    reward_normalization_stats_path: Optional[str] = None,
) -> Tuple[List[Dict[str, np.ndarray]], List[List[Dict[str, np.ndarray]]]]:
    """
    Load trajectories from datasets/HalfCheetah-v2/<data_quality>/trajectories.pkl.
    Returns (trajectories, prompt_per_task). For single-task HalfCheetah, prompt_per_task
    is [all_trajectories_sorted_by_return] so context = same mix sorted by return.
    """
    path = Path(data_dir) / env_name / data_quality / "trajectories.pkl"
    if not path.exists():
        return [], []
    log.info("Loading trajectories from {}...", path)
    with open(path, "rb") as f:
        trajectories = pickle.load(f)
    if max_trajectories:
        trajectories = trajectories[:max_trajectories]

    mode = (reward_normalization or "none").strip().lower()
    if mode not in ("none", "", "off"):
        stats_pre: Optional[Dict[str, Any]] = None
        sp_str = reward_normalization_stats_path
        if isinstance(sp_str, str) and sp_str.strip().lower() in ("", "null", "none", "~"):
            sp_str = None
        if sp_str:
            sp = Path(sp_str)
            if sp.is_file():
                stats_pre = load_stats(sp)
                log.info("Loaded reward norm stats from {}", sp)
            else:
                log.warning(
                    "reward_normalization_stats_path={} not found; computing stats from data",
                    sp,
                )
        trajectories, st = apply_reward_normalization(
            trajectories,
            reward_normalization,
            reward_norm_constant=reward_norm_constant,
            epsilon=reward_norm_epsilon,
            stats=stats_pre,
            in_place=True,
        )
        log.info(
            "Applied data.reward_normalization={} (constant={}) stats_keys={}",
            reward_normalization,
            reward_norm_constant,
            list(st.keys()),
        )

    # Single task: use all trajectories as context pool, sorted by return (best first)
    sorted_pool = sort_trajectories_by_return(trajectories, ascending=False)
    prompt_per_task = [sorted_pool]
    return trajectories, prompt_per_task
