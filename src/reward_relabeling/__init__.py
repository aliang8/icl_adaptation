"""
Reward relabeling: offline reward computation for trajectories.

Dataset-agnostic models (RoboDopamine-8B, Robometer-4B) live here.
Dataset-specific scripts (e.g. LIBERO) import from this package and handle
loading/saving (manifest, episodes/, lowdim.npz, etc.).
"""

from src.reward_relabeling.reward_model import (
    RewardModel,
    Robometer4BRewardModel,
    RoboDopamine8BRewardModel,
    build_reward_model,
    pad_or_trunc_1d,
)

__all__ = [
    "RewardModel",
    "Robometer4BRewardModel",
    "RoboDopamine8BRewardModel",
    "build_reward_model",
    "pad_or_trunc_1d",
]
