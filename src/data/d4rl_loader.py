"""Load D4RL-style or pre-downloaded HalfCheetah trajectories (mixed returns)."""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger as log

from src.data.trajectories import sort_trajectories_by_return


def parse_halfcheetah_data_qualities(data_quality: Any) -> List[str]:
    """Resolve quality tags from a string, comma-separated string, or Hydra list.

    Use CLI ``data.data_quality=[medium,medium_expert]`` to avoid Hydra comma ambiguity;
    or a single string ``medium_expert``; or ``\"random,medium_expert\"`` as one string.
    """
    if data_quality is None:
        return ["medium_expert"]
    # Hydra: data.data_quality=[medium,medium_expert] → list / ListConfig
    if isinstance(data_quality, (list, tuple)):
        parts = [str(x).strip() for x in data_quality if str(x).strip()]
        return parts if parts else ["medium_expert"]
    try:
        from omegaconf import ListConfig

        if isinstance(data_quality, ListConfig):
            parts = [str(x).strip() for x in data_quality if str(x).strip()]
            return parts if parts else ["medium_expert"]
    except ImportError:
        pass
    s = str(data_quality).strip()
    if not s:
        return ["medium_expert"]
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts if parts else ["medium_expert"]


def format_data_quality_for_log(data_quality: Any) -> str:
    """Human-readable label for logging (e.g. dataset stats title)."""
    return ", ".join(parse_halfcheetah_data_qualities(data_quality))


def load_halfcheetah_trajectories(
    data_dir: str,
    env_name: str = "HalfCheetah-v2",
    data_quality: Any = "medium_expert",
    max_trajectories: Optional[int] = None,
) -> Tuple[List[Dict[str, np.ndarray]], List[List[Dict[str, np.ndarray]]]]:
    """
    Load trajectories from ``datasets/<env_name>/<quality>/trajectories.pkl``.

    ``data_quality`` may be a single tag, comma-separated tags, or a list (Hydra
    ``[a,b]``); in the multi-pool case,
    trajectories from each subdirectory are concatenated, then sorted by return for
    ``prompt_per_task`` (same as single-pool behavior).

    Returns (trajectories, prompt_per_task). For single-task HalfCheetah, prompt_per_task
    is [all_trajectories_sorted_by_return] so context = same mix sorted by return.
    """
    qualities = parse_halfcheetah_data_qualities(data_quality)
    trajectories: List[Dict[str, np.ndarray]] = []
    multi = len(qualities) > 1

    for q in qualities:
        path = Path(data_dir) / env_name / q / "trajectories.pkl"
        if not path.exists():
            if multi:
                raise FileNotFoundError(
                    f"HalfCheetah multi-quality load: missing {path} (need all of: {qualities})"
                )
            return [], []
        log.info("Loading trajectories from {}...", path)
        with open(path, "rb") as f:
            chunk = pickle.load(f)
        trajectories.extend(chunk)

    if max_trajectories:
        trajectories = trajectories[:max_trajectories]

    sorted_pool = sort_trajectories_by_return(trajectories, ascending=False)
    prompt_per_task = [sorted_pool]
    return trajectories, prompt_per_task
