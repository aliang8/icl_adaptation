"""RTG token helper. Step rewards in trajectories and at eval are **raw** env rewards; only RTG uses ``rtg_scale``."""

from __future__ import annotations

from typing import Optional


def initial_rtg_token(
    rtg_scale: float,
    *,
    eval_target_return: Optional[float] = None,
) -> float:
    """
    First RTG **token** for a rollout, matching training: ``cumsum(r_env) / rtg_scale``.

    If ``eval_target_return`` is set, it is the target **future cumulative return** in the same
    units as env step rewards; token = ``eval_target_return / rtg_scale``.

    If omitted, uses classic DT-style default: assume a target future return of ``rtg_scale`` so
    token = ``1.0``.
    """
    s = float(rtg_scale)
    if s == 0.0:
        raise ValueError("rtg_scale must be non-zero")
    if eval_target_return is not None:
        return float(eval_target_return) / s
    return 1.0
