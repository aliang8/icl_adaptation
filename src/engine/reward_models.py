"""
Modular reward for eval: env return (sum of rewards) or a vision-language reward model
(e.g. RoboReward-8B, Robometer-4B) applied to task + rollout.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def get_return_from_env(trajectory: Dict[str, np.ndarray]) -> float:
    """Return sum of rewards (env return)."""
    return float(np.sum(trajectory["rewards"]))


def get_return_from_reward_model(
    task_description: str,
    trajectory: Dict[str, np.ndarray],
    model_name: str,
    video_path: Optional[str] = None,
    frames: Optional[List[np.ndarray]] = None,
    **kwargs: Any,
) -> float:
    """
    Get scalar return from a reward model given task and rollout (video or frames).
    Implementations: roboreward_8b, robometer_4b. Use eval_reward_source=env for env return.
    """
    if model_name in ("roboreward_8b", "teetone/RoboReward-8B"):
        return _roboreward_score(task_description, video_path, frames, **kwargs)
    if model_name in ("robometer_4b", "robometer/Robometer-4B"):
        return _robometer_score(task_description, video_path, frames, **kwargs)
    raise NotImplementedError(
        f"Reward model '{model_name}' not implemented. Use eval_reward_source=env for env return."
    )


def _roboreward_score(
    task: str,
    video_path: Optional[str],
    frames: Optional[List[np.ndarray]],
    **kwargs: Any,
) -> float:
    """RoboReward-8B: discrete progress 1–5. Map to [0,1] or use as-is for sorting."""
    raise NotImplementedError(
        "RoboReward-8B integration not yet implemented. "
        "See https://huggingface.co/teetone/RoboReward-8B. Use eval_reward_source=env for now."
    )


def _robometer_score(
    task: str,
    video_path: Optional[str],
    frames: Optional[List[np.ndarray]],
    **kwargs: Any,
) -> float:
    """Robometer-4B: per-frame progress / preference. Aggregate for trajectory score."""
    raise NotImplementedError(
        "Robometer-4B integration not yet implemented. "
        "See https://huggingface.co/robometer/Robometer-4B. Use eval_reward_source=env for now."
    )
