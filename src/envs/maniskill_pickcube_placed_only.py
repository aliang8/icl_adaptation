"""Optional PickCube task: ``success`` requires goal placement only (no ``is_robot_static`` gate).

Stock ``PickCube-v1`` uses ``success = is_obj_placed & is_robot_static`` so the arm must settle.
Use ``PickCube-v1-PlaceOnly`` via ``--env-id`` when you want success (and the +5 / normalized +1
success reward overwrite) as soon as the cube is in the goal volume, even if the robot is still
moving.

Requires ``import mani_skill.envs`` before this module registers the env (see ``ppo_train_icldata``).
"""

from __future__ import annotations

import torch

from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.registration import register_env


@register_env("PickCube-v1-PlaceOnly", max_episode_steps=50)
class PickCubePlaceOnlyEnv(PickCubeEnv):
    """``PickCube-v1`` with ``success = is_obj_placed`` only."""

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1) <= self.goal_thresh
        )
        is_grasped = self.agent.is_grasping(self.cube)
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_obj_placed,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }
