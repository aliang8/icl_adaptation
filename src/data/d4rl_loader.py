"""Load D4RL-style or pre-downloaded HalfCheetah trajectories (mixed returns)."""
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
from loguru import logger as log

from src.data.trajectories import sort_trajectories_by_return


def load_halfcheetah_trajectories(
    data_dir: str,
    env_name: str = "HalfCheetah-v2",
    data_quality: str = "medium_expert",
    max_trajectories: Optional[int] = None,
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
        trajectories = trajectories[: max_trajectories]
    # Single task: use all trajectories as context pool, sorted by return (best first)
    sorted_pool = sort_trajectories_by_return(trajectories, ascending=False)
    prompt_per_task = [sorted_pool]
    return trajectories, prompt_per_task
