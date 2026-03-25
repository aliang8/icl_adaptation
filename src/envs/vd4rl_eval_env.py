"""
V-D4RL-style eval: dm_control suite env with pixel observations processed like `vd4rl_loader`
(flattened resized RGB), so observation dim matches offline training.

See upstream task naming: https://github.com/conglu1997/v-d4rl (walker_walk, cheetah_run, …).
This uses `dm_control.suite` + `pixels.Wrapper`, not Gymnasium MuJoCo (e.g. Walker2d-v5), which
is a different domain than dm_control Walker-walk.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

# dm_control task id (V-D4RL / DrQ naming) -> (suite domain, suite task)
VD4RL_DMC_SUITE_TASK: Dict[str, Tuple[str, str]] = {
    "walker_walk": ("walker", "walk"),
    "cheetah_run": ("cheetah", "run"),
    "humanoid_walk": ("humanoid", "walk"),
}


def _dm_control_task_or_raise(task_slug: str) -> Tuple[str, str]:
    if task_slug not in VD4RL_DMC_SUITE_TASK:
        raise ValueError(
            f"Unknown V-D4RL dm_control task {task_slug!r}. "
            f"Supported: {sorted(VD4RL_DMC_SUITE_TASK.keys())}. "
            "Set data.eval_env_name to VD4RL/dmc/<slug> or add a mapping in vd4rl_eval_env.py."
        )
    return VD4RL_DMC_SUITE_TASK[task_slug]


class VD4RLDmControlPixelGymEnv:
    """
    Gymnasium-like API: observations are float32 vectors (obs_downsample**2 * 3), matching
    `src.data.vd4rl_loader._resize_flatten_observation` on rendered pixels.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        task_slug: str,
        *,
        pixel_hw: int,
        obs_downsample: int,
        seed: int = 0,
    ):
        try:
            from dm_control import suite
            from dm_control.suite.wrappers import action_scale, pixels
        except ImportError as e:
            raise ImportError(
                "V-D4RL dm_control eval requires `dm_control`. "
                "Install: uv sync --extra d4rl (includes dm_control) or pip install dm_control"
            ) from e

        domain, task = _dm_control_task_or_raise(task_slug)
        self._task_slug = task_slug
        self._pixel_hw = int(pixel_hw)
        self._obs_downsample = int(obs_downsample)
        self._seed = int(seed)

        env = suite.load(domain, task, task_kwargs={"random": self._seed}, visualize_reward=False)
        env = action_scale.Wrapper(env, minimum=-1.0, maximum=1.0)
        # Match v-d4rl drqbc/dmc.py: zoomed camera for quadrupeds
        camera_id = 2 if domain == "cheetah" else 0
        env = pixels.Wrapper(
            env,
            pixels_only=True,
            render_kwargs=dict(height=self._pixel_hw, width=self._pixel_hw, camera_id=camera_id),
        )
        self._env = env
        self._last_pixels_hwc: Optional[np.ndarray] = None

        try:
            import cv2

            self._cv2 = cv2
        except ImportError:
            self._cv2 = None

        from src.data.vd4rl_loader import _resize_flatten_observation

        self._resize_flatten = _resize_flatten_observation

        act_spec = env.action_spec()
        adim = int(np.prod(act_spec.shape, dtype=int))
        obs_dim = self._obs_downsample * self._obs_downsample * 3
        lo = np.asarray(act_spec.minimum, dtype=np.float32).reshape(-1)
        hi = np.asarray(act_spec.maximum, dtype=np.float32).reshape(-1)
        if lo.size == 1 and adim > 1:
            lo = np.full((adim,), float(lo[0]), dtype=np.float32)
        if hi.size == 1 and adim > 1:
            hi = np.full((adim,), float(hi[0]), dtype=np.float32)

        try:
            import gymnasium as gym

            self._Box = gym.spaces.Box
        except ImportError:
            import gym

            self._Box = gym.spaces.Box

        self.observation_space = self._Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = self._Box(low=lo, high=hi, shape=(adim,), dtype=np.float32)

    def _vec_from_pixels(self, img_hwc: np.ndarray) -> np.ndarray:
        v = self._resize_flatten(img_hwc, self._obs_downsample, self._cv2)
        return np.asarray(v, dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._seed = int(seed)
        ts = self._env.reset()
        img = np.asarray(ts.observation["pixels"], dtype=np.uint8)
        self._last_pixels_hwc = img
        return self._vec_from_pixels(img), {}

    def step(self, action: np.ndarray):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        ts = self._env.step(a)
        img = np.asarray(ts.observation["pixels"], dtype=np.uint8)
        self._last_pixels_hwc = img
        obs = self._vec_from_pixels(img)
        reward = float(ts.reward) if ts.reward is not None else 0.0
        terminated = ts.last()
        truncated = False
        return obs, reward, terminated, truncated, {}

    def render(self):
        if self._last_pixels_hwc is None:
            return None
        return np.ascontiguousarray(self._last_pixels_hwc)

    def close(self):
        pass


def parse_vd4rl_dmc_env_name(env_name: str) -> Optional[str]:
    """If env_name is ``VD4RL/dmc/<task_slug>``, return task_slug; else None."""
    prefix = "VD4RL/dmc/"
    if env_name.startswith(prefix):
        return env_name[len(prefix) :].strip() or None
    return None


def make_vd4rl_dm_control_pixel_env(
    env_name: str,
    *,
    pixel_hw: int,
    obs_downsample: int,
    seed: int = 0,
) -> Any:
    """Create env from ``VD4RL/dmc/walker_walk``-style id."""
    slug = parse_vd4rl_dmc_env_name(env_name)
    if slug is None:
        raise ValueError(f"Expected env_name like VD4RL/dmc/walker_walk, got {env_name!r}")
    return VD4RLDmControlPixelGymEnv(
        slug, pixel_hw=pixel_hw, obs_downsample=obs_downsample, seed=seed
    )
