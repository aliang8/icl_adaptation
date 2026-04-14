"""Gymnasium / suite env construction and RGB rendering for eval rollouts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np


class ManiSkillEvalEnvCache:
    """Reuse one ManiSkill eval env across calls when the construction key is unchanged.

    Periodic training eval otherwise builds and tears down PhysX/SAPIEN each step, which is slow
    and can leak GPU/sim resources. :meth:`get_or_create` keeps a single env until the key changes
    or :meth:`close` is called (e.g. after training).
    """

    def __init__(self) -> None:
        self._env: Optional[Any] = None
        self._key: Optional[tuple[Any, ...]] = None

    def get_or_create(self, key: tuple[Any, ...], factory: Callable[[], Any]) -> Any:
        if self._key != key:
            self.close()
            self._env = factory()
            self._key = key
        assert self._env is not None
        return self._env

    def close(self) -> None:
        if self._env is None:
            self._key = None
            return
        env = self._env
        self._env = None
        self._key = None
        try:
            env.close()
        except BaseException as ex:
            print(f"[ManiSkillEvalEnvCache] env.close() failed (ignored): {ex!r}", flush=True)

# D4RL-era ids (e.g. HalfCheetah-v2) → Gymnasium MuJoCo v5 where shapes match.
GYMNASIUM_EVAL_ENV_ALIASES = {
    "HalfCheetah-v2": "HalfCheetah-v5",
    "halfcheetah-v2": "HalfCheetah-v5",
}
_EVAL_ENV_ALIAS_LOGGED: set[str] = set()


def render_rgb_frame(env: Any) -> Optional[np.ndarray]:
    frame = env.render()
    if frame is None:
        return None

    try:
        import torch

        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
    except ImportError:
        pass

    if not isinstance(frame, np.ndarray):
        return None

    arr = np.asarray(frame)
    while arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim != 3 or arr.shape[-1] != 3:
        return None

    if arr.dtype != np.uint8:
        mx = float(np.max(arr)) if arr.size else 0.0
        if mx <= 1.0 + 1e-6:
            arr = np.clip(arr * 255.0, 0, 255)
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


def try_make_eval_env(
    env_name: str,
    render_both_views: bool = True,
    render_mode: Optional[str] = None,
    *,
    vd4rl_eval_pixel_hw: Optional[int] = None,
    vd4rl_eval_obs_downsample: Optional[int] = None,
    vd4rl_eval_seed: int = 0,
    minari_halfcheetah_dataset_id: Optional[str] = None,
    maniskill_sim_backend: Optional[str] = None,
    maniskill_reward_mode: Optional[str] = None,
    maniskill_control_mode: Optional[str] = None,
    maniskill_state_obs_slice: Optional[slice] = None,
):
    from src.envs.libero_env import LIBERO_SUITES, make_libero_env

    if env_name.startswith("VD4RL/dmc/"):
        from src.envs.vd4rl_eval_env import make_vd4rl_dm_control_pixel_env

        if vd4rl_eval_pixel_hw is None or vd4rl_eval_obs_downsample is None:
            raise ValueError(
                f"Env {env_name!r} requires data.vd4rl_pixel_size and data.vd4rl_obs_downsample "
                "(passed as vd4rl_eval_pixel_hw / vd4rl_eval_obs_downsample from train)."
            )
        return make_vd4rl_dm_control_pixel_env(
            env_name,
            pixel_hw=int(vd4rl_eval_pixel_hw),
            obs_downsample=int(vd4rl_eval_obs_downsample),
            seed=int(vd4rl_eval_seed),
        )

    if env_name in LIBERO_SUITES:
        return make_libero_env(
            suite_name=env_name,
            task_id=0,
            state_dim=9,
            action_dim=7,
            render_both_views=render_both_views,
        )

    if env_name.startswith("ManiSkill/"):
        from src.envs.maniskill_eval_env import make_maniskill_eval_env

        return make_maniskill_eval_env(
            env_name,
            render_mode=render_mode,
            sim_backend=maniskill_sim_backend,
            reward_mode=maniskill_reward_mode,
            control_mode=maniskill_control_mode,
            reconfiguration_freq=1,
            state_obs_slice=maniskill_state_obs_slice,
        )

    import gymnasium as gym

    gym_id = GYMNASIUM_EVAL_ENV_ALIASES.get(env_name, env_name)
    if gym_id != env_name and env_name not in _EVAL_ENV_ALIAS_LOGGED:
        _EVAL_ENV_ALIAS_LOGGED.add(env_name)
        print(
            f"Eval env alias: {env_name!r} -> {gym_id!r} "
            f"(core Gymnasium MuJoCo; same obs/act dims as D4RL/Minari HalfCheetah)"
        )

    if minari_halfcheetah_dataset_id and gym_id == "HalfCheetah-v5":
        from src.envs.minari_halfcheetah_eval import make_halfcheetah_env_via_minari

        return make_halfcheetah_env_via_minari(
            minari_halfcheetah_dataset_id, render_mode=render_mode
        )

    try:
        if render_mode is not None:
            return gym.make(gym_id, render_mode=render_mode)
        return gym.make(gym_id)
    except Exception as e:
        msg = str(e).lower()
        if "gymnasium-robotics" in msg or "mujoco v2" in msg or "mujoco v3" in msg:
            raise RuntimeError(
                f"Could not create Gymnasium env {gym_id!r} (from config {env_name!r}). "
                f"For HalfCheetah, use MuJoCo-backed Gymnasium (HalfCheetah-v5) and "
                f"`uv sync --extra d4rl` so `mujoco` is installed. Original error: {e}"
            ) from e
        raise


class GymnasiumToGymStepAdapter:
    """Wraps a Gymnasium env so step() returns (obs, reward, done, info) for gym RecordVideo."""

    def __init__(self, env: Any):
        self._env = env

    def __getattr__(self, name: str) -> Any:
        env = object.__getattribute__(self, "_env")
        return object.__getattribute__(env, name)

    def step(self, action: Any):
        out = self._env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            return (obs, reward, bool(terminated or truncated), info)
        return out

    def reset(self, *args, **kwargs):
        return self._env.reset(*args, **kwargs)


def wrap_record_video(env: Any, video_folder: Path) -> Tuple[Any, bool]:
    import gymnasium as gym

    return (
        gym.wrappers.RecordVideo(env, str(video_folder), episode_trigger=lambda ep: True),
        True,
    )
