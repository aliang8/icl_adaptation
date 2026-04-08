"""
ManiSkill flat ``obs_mode="state"`` vectors mix robot proprioception with task/object cues.

When ``data.use_vision`` is true, training trims to proprio + EEF only for known tasks
(camera carries goal/object cues).

**PickCube-v1** (D = 42, ManiSkill 3.x layout):

- ``[0:9)``   agent qpos
- ``[9:18)``  agent qvel
- ``[18:19)`` ``extra.is_grasped``
- ``[19:26)`` ``extra.tcp_pose`` (7)
- ``[26:29)`` ``extra.goal_pos`` — dropped under ``use_vision`` for supported tasks
- ``[29:36)`` ``extra.obj_pose`` — dropped
- ``[36:39)`` ``extra.tcp_to_obj_pos`` — dropped
- ``[39:42)`` ``extra.obj_to_goal_pos`` — dropped

Vision proprio slice: ``0:26`` → dim 26.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PICKCUBE_V1_FULL_STATE_DIM = 42
PICKCUBE_V1_VISION_PROPRIO_SLICE = slice(0, 26)
PICKCUBE_V1_VISION_PROPRIO_DIM = 26

# ManiSkill/env id -> slice into flat state (``None`` = no built-in layout).
_VISION_PROPRIO_SLICES: Dict[str, slice] = {
    "PickCube-v1": PICKCUBE_V1_VISION_PROPRIO_SLICE,
}


def maniskill_task_from_env_name(env_name: str) -> str:
    if env_name.startswith("ManiSkill/"):
        return env_name.split("/", 1)[1]
    return env_name


def vision_proprio_slice_for_task(task_id: str) -> Optional[slice]:
    return _VISION_PROPRIO_SLICES.get(task_id)


def expected_full_state_dim(task_id: str) -> Optional[int]:
    if task_id == "PickCube-v1":
        return PICKCUBE_V1_FULL_STATE_DIM
    return None


def filter_trajectory_observations(traj: Dict[str, Any], env_name: str) -> Dict[str, Any]:
    """Return a copy of ``traj`` with ``observations`` sliced to vision-proprio indices."""
    task = maniskill_task_from_env_name(env_name)
    sl = vision_proprio_slice_for_task(task)
    if sl is None:
        raise ValueError(
            f"No vision-proprio state layout for ManiSkill task {task!r}. "
            f"Supported: {sorted(_VISION_PROPRIO_SLICES)}."
        )
    o = np.asarray(traj["observations"], dtype=np.float32)
    exp = expected_full_state_dim(task)
    if exp is not None and o.shape[-1] != exp:
        raise ValueError(
            f"ManiSkill {task}: expected full state dim {exp} for proprio filter, "
            f"got observations.shape[-1]={o.shape[-1]}"
        )
    if o.shape[-1] < sl.stop:
        raise ValueError(
            f"ManiSkill {task}: observations last dim {o.shape[-1]} < slice end {sl.stop}"
        )
    out = dict(traj)
    out["observations"] = o[..., sl].copy()
    return out


def apply_maniskill_vision_proprio_to_bundle(
    trajectories: List[Dict[str, Any]],
    prompt_per_task: List[List[Dict[str, Any]]],
    env_name: str,
) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    """Slice observations on every trajectory (main pool + prompt pools)."""
    main = [filter_trajectory_observations(t, env_name) for t in trajectories]
    prompt = [
        [filter_trajectory_observations(t, env_name) for t in pool] for pool in prompt_per_task
    ]
    return main, prompt
