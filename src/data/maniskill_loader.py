"""Load ManiSkill-exported ICL replay buffers (layout in :mod:`src.data.ic_replay_buffer_hdf5`)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from src.data.ic_replay_buffer_hdf5 import load_ic_replay_buffer_bundle


def load_maniskill_trajectories(
    trajectory_path: Union[str, Path, Sequence[Union[str, Path]]],
    *,
    min_episode_length: Optional[int] = None,
    log_summary: bool = True,
) -> Tuple[List[Dict], List[List[Dict]]]:
    """
    Args:
        trajectory_path: One path or a list of ``.h5`` / ``.hdf5`` flat v2 bundles (concatenated).
        min_episode_length: If set and positive, keep only episodes with ``T >= min_episode_length``
            (e.g. ``data.min_trajectory_length`` for algorithm distillation).
        log_summary: Log formatted dataset stats (lengths, returns, ``episode_meta`` rates).

    Returns:
        (trajectories, prompt_per_task) compatible with ``get_icl_trajectory_dataset``.
        ``prompt_per_task`` is a one-element list (shared pool); ``train.py`` replicates it across
        synthetic ``task_id`` buckets so every trajectory sees the same pool.
    """
    return load_ic_replay_buffer_bundle(
        trajectory_path,
        min_episode_length=min_episode_length,
        log_summary=log_summary,
    )
