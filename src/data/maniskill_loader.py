"""Load ManiSkill PPO-exported ``trajectories.h5`` for ICL training."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from src.data.maniskill_io import load_trajectories_file
from src.data.trajectories import sort_trajectories_by_return


def load_maniskill_trajectories(trajectory_path: str) -> Tuple[List[Dict], List[List[Dict]]]:
    """
    Args:
        trajectory_path: Path to ``trajectories.h5``.

    Returns:
        (trajectories, prompt_per_task) compatible with ``get_icl_trajectory_dataset``.
        ``prompt_per_task`` is a one-element list (shared pool); ``train.py`` replicates it across
        synthetic ``task_id`` buckets so every trajectory sees the same pool.
    """
    path = Path(trajectory_path)
    if not path.is_file():
        raise FileNotFoundError(f"ManiSkill trajectories not found: {path}")
    trajectories = load_trajectories_file(path)
    if not trajectories:
        raise ValueError(f"No trajectories in {path}")
    pool = sort_trajectories_by_return(list(trajectories), ascending=False)
    prompt_per_task = [pool]
    return trajectories, prompt_per_task
