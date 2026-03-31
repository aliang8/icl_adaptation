"""
D4RL-style normalized episode return for reporting (Decision Transformer / offline RL papers).

Formula (same as ``d4rl.get_normalized_score``): 100 * (R - R_min) / (R_max - R_min), where
R_min / R_max are reference random / expert returns for the task. Values are not clipped.

MuJoCo HalfCheetah (Gym / Gymnasium / Minari ``mujoco/halfcheetah/*``): REF_MIN and REF_MAX are
shared across all halfcheetah dataset variants in upstream D4RL — see
``REF_MIN_SCORE`` / ``REF_MAX_SCORE`` in
https://github.com/rail-berkeley/d4rl/blob/master/d4rl/infos.py
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

# halfcheetah-*-v0 / v1 / v2 all use the same refs in d4rl/infos.py
MUJOCO_HALFCHEETAH_D4RL_REF_MIN: float = -280.178953
MUJOCO_HALFCHEETAH_D4RL_REF_MAX: float = 12135.0

MUJOCO_HALFCHEETAH_D4RL_REF: Tuple[float, float] = (
    MUJOCO_HALFCHEETAH_D4RL_REF_MIN,
    MUJOCO_HALFCHEETAH_D4RL_REF_MAX,
)


def d4rl_normalize_returns(returns: Sequence[float], ref_min: float, ref_max: float) -> np.ndarray:
    """Per-episode normalized scores in D4RL percent units (typically ~0 at random, ~100 at expert)."""
    span = float(ref_max) - float(ref_min)
    if span <= 0:
        raise ValueError(
            f"d4rl_normalize_returns: need ref_max > ref_min, got {ref_min}, {ref_max}"
        )
    r = np.asarray(returns, dtype=np.float64)
    return 100.0 * (r - float(ref_min)) / span


def d4rl_normalize_halfcheetah_returns(returns: Sequence[float]) -> np.ndarray:
    """Normalize raw HalfCheetah env return sums using D4RL MuJoCo reference scores."""
    return d4rl_normalize_returns(returns, *MUJOCO_HALFCHEETAH_D4RL_REF)
