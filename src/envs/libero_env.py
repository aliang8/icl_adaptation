"""
LIBERO (Lifelong-Robot-Learning/LIBERO) env wrapper for eval rollouts.
Uses LIBERO benchmark suite names: libero_10, libero_spatial, libero_object, libero_goal, libero_90.

See: https://github.com/Lifelong-Robot-Learning/LIBERO
     https://github.com/NVlabs/cosmos-policy (LIBERO.md)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Benchmark suite names (lowercase, as in get_benchmark_dict()). No libero import here so callers can use this without installing libero.
LIBERO_SUITES = ("libero_10", "libero_spatial", "libero_object", "libero_goal", "libero_90")

# Add project's LIBERO clone to path when this module is loaded, so "import libero" works when running from project root.
_project_root = Path(__file__).resolve().parent.parent.parent
_libero_dir = _project_root / "LIBERO"
if _libero_dir.is_dir() and str(_libero_dir) not in sys.path:
    sys.path.insert(0, str(_libero_dir))


def _obs_to_proprio(obs: Any, state_dim: int = 9) -> np.ndarray:
    """Extract 9-dim proprio from LIBERO/robosuite obs dict or return array as-is."""
    if isinstance(obs, np.ndarray):
        if obs.size == state_dim:
            return np.asarray(obs, dtype=np.float32).flatten()
        return np.asarray(obs, dtype=np.float32).flatten()[:state_dim]
    if isinstance(obs, dict):
        if "proprio" in obs:
            return np.asarray(obs["proprio"], dtype=np.float32).flatten()[:state_dim]
        parts = []
        for key in ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]:
            if key in obs:
                parts.append(np.asarray(obs[key]).flatten())
        if parts:
            out = np.concatenate(parts, axis=0).astype(np.float32)[:state_dim]
            if len(out) < state_dim:
                out = np.pad(out, (0, state_dim - len(out)))
            return out
    return np.zeros(state_dim, dtype=np.float32)


def make_libero_env(
    suite_name: str,
    task_id: int = 0,
    camera_heights: int = 128,
    camera_widths: int = 128,
    state_dim: int = 9,
    action_dim: int = 7,
) -> Any:
    """
    Create a gym-like LIBERO env for the given benchmark suite and task.

    Args:
        suite_name: One of libero_10, libero_spatial, libero_object, libero_goal, libero_90.
        task_id: Task index within the suite (0 to n_tasks-1).
        camera_heights, camera_widths: OffScreenRenderEnv camera size.
        state_dim, action_dim: Observation and action size (9 and 7 for LIBERO-Cosmos).

    Returns:
        Env with observation_space (state_dim,), action_space (action_dim,);
        reset() -> (obs, info), step(a) -> (obs, reward, terminated, truncated, info).
    """
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    try:
        import gymnasium as gym

        Box = gym.spaces.Box
    except ImportError:
        import gym

        Box = gym.spaces.Box

    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        raise ValueError(
            f"Unknown LIBERO suite '{suite_name}'. Must be one of: {list(benchmark_dict.keys())}"
        )
    task_suite = benchmark_dict[suite_name]()
    n_tasks = task_suite.get_num_tasks()
    if task_id < 0 or task_id >= n_tasks:
        task_id = 0
    bddl_file_name = task_suite.get_task_bddl_file_path(task_id)
    if not os.path.isfile(bddl_file_name):
        raise FileNotFoundError(
            f"LIBERO BDDL file not found: {bddl_file_name}. Install LIBERO and ensure bddl_files are present."
        )
    base_env = OffScreenRenderEnv(
        bddl_file_name=bddl_file_name,
        camera_heights=camera_heights,
        camera_widths=camera_widths,
    )

    observation_space = Box(
        low=-np.inf,
        high=np.inf,
        shape=(state_dim,),
        dtype=np.float32,
    )
    action_space = Box(
        low=-1.0,
        high=1.0,
        shape=(action_dim,),
        dtype=np.float32,
    )

    class LIBEROWrapper:
        def __init__(self, env, state_dim: int, action_dim: int):
            self._env = env
            self._state_dim = state_dim
            self._action_dim = action_dim
            self.observation_space = observation_space
            self.action_space = action_space

        def reset(self, seed=None, options=None):
            if seed is not None:
                self._env.seed(seed)
            obs = self._env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            proprio = _obs_to_proprio(obs, self._state_dim)
            return proprio, {}

        def step(self, action: np.ndarray):
            action = np.asarray(action, dtype=np.float64).flatten()[: self._action_dim]
            if action.size < self._action_dim:
                action = np.pad(
                    action,
                    (0, self._action_dim - action.size),
                    mode="constant",
                    constant_values=0.0,
                )
            step_out = self._env.step(action)
            if len(step_out) == 4:
                obs, reward, done, info = step_out
                truncated = False
            else:
                obs, reward, done, truncated, info = step_out[:5]
            if isinstance(obs, tuple):
                obs = obs[0]
            proprio = _obs_to_proprio(obs, self._state_dim)
            return proprio, float(reward), bool(done), bool(truncated), info or {}

        def close(self):
            self._env.close()

    return LIBEROWrapper(base_env, state_dim, action_dim)
