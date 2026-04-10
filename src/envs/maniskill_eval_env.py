"""
ManiSkill environments for ``run_rollouts_and_save_viz`` (``src.train`` periodic eval).

Training data uses ``data.env_name=ManiSkill/<Task-v1>``; Gymnasium registers plain ``<Task-v1>``
after ``import mani_skill.envs``. Vector envs return batched torch tensors — this module adapts
to numpy vectors expected by ``eval_viz._run_one_rollout``.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any, Optional, Tuple

import numpy as np


def _vec_render_to_hwc_u8(vec_env: Any) -> Optional[np.ndarray]:
    """ManiSkill ``render()`` -> (H, W, 3) uint8 RGB (aligned with ``src.envs.eval_gym.render_rgb_frame``)."""
    frame = vec_env.render()
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


def ensure_maniskill_imported() -> None:
    if importlib.util.find_spec("mani_skill") is None:
        raise RuntimeError(
            "ManiSkill eval needs `mani_skill` importable in the **same** Python that runs "
            "`src.train` (not only `.venv-maniskill` on your shell prompt).\n"
            "  `uv run python -m src.train` uses the project uv env — install there, or run training as:\n"
            "    .venv-maniskill/bin/python -m src.train ...   (with PYTHONPATH=repo and ICL deps)\n"
            "  pip install -r scripts/maniskill/requirements.txt\n"
            "  pip install -r scripts/maniskill/requirements_icl_train.txt\n"
            "See docs/MANISKILL.md."
        )
    import mani_skill.envs  # noqa: F401 — register envs with Gymnasium


def maniskill_task_id(env_name: str) -> str:
    if env_name.startswith("ManiSkill/"):
        return env_name.split("/", 1)[1]
    return env_name


def _obs_to_numpy_vector(obs: Any) -> np.ndarray:
    import torch

    if torch.is_tensor(obs):
        x = obs.detach().cpu().float().numpy().reshape(-1)
        return np.asarray(x, dtype=np.float32)
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def _first_bool(x: Any) -> bool:
    import torch

    if torch.is_tensor(x):
        return bool(x.reshape(-1)[0].item())
    return bool(np.asarray(x).reshape(-1)[0])


def _first_float(x: Any) -> float:
    import torch

    if torch.is_tensor(x):
        return float(x.reshape(-1)[0].item())
    return float(np.asarray(x).reshape(-1)[0])


def _box_slice_obs_space(base: Any, sl: slice) -> Any:
    """Narrow a 1D Box to ``base.low[sl], base.high[sl]``."""
    try:
        import gymnasium as gym
    except ImportError:
        return base
    if not isinstance(base, gym.spaces.Box):
        return base
    lo = np.asarray(base.low, dtype=np.float32).reshape(-1)
    hi = np.asarray(base.high, dtype=np.float32).reshape(-1)
    if lo.shape[0] < sl.stop or hi.shape[0] < sl.stop:
        return base
    return gym.spaces.Box(
        low=lo[sl],
        high=hi[sl],
        shape=(sl.stop - sl.start,),
        dtype=base.dtype,
    )


class ManiSkillEvalAdapter:
    """Wraps ManiSkill vector env (``num_envs=1``) for eval_viz single-env numpy API."""

    def __init__(self, vec_env: Any, *, state_obs_slice: Optional[slice] = None) -> None:
        self._env = vec_env
        self._device = getattr(vec_env, "device", None)
        self._state_obs_slice = state_obs_slice
        if not hasattr(vec_env, "single_observation_space"):
            raise TypeError(
                f"Expected ManiSkill vector env with single_observation_space; got {type(vec_env)}"
            )
        base_obs = vec_env.single_observation_space
        if state_obs_slice is not None:
            self.observation_space = _box_slice_obs_space(base_obs, state_obs_slice)
        else:
            self.observation_space = base_obs
        self.action_space = vec_env.single_action_space

    def _project_obs(self, o_np: np.ndarray) -> np.ndarray:
        if self._state_obs_slice is not None:
            return np.asarray(o_np[self._state_obs_slice], dtype=np.float32)
        return o_np

    def reset(self, *, seed: Optional[int] = None, options: Any = None) -> Tuple[np.ndarray, dict]:
        obs, info = self._env.reset(seed=seed, options=options)
        return self._project_obs(_obs_to_numpy_vector(obs)), info

    def step(self, action_np: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        import torch

        dev = self._device or torch.device("cpu")
        a = torch.as_tensor(np.asarray(action_np, dtype=np.float32), device=dev).reshape(1, -1)
        o, r, term, trunc, info = self._env.step(a)
        o_np = self._project_obs(_obs_to_numpy_vector(o))
        r_f = _first_float(r)
        done = _first_bool(term) or _first_bool(trunc)
        return o_np, r_f, done, False, info

    def render(self) -> Any:
        return self._env.render()

    def close(self) -> None:
        self._env.close()

    def get_current_images(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Primary camera RGB for vision eval; wrist unused (single-camera ManiSkill tasks)."""
        arr = _vec_render_to_hwc_u8(self._env)
        if arr is None:
            return None, None
        return arr, None


def make_maniskill_eval_env(
    env_name: str,
    *,
    render_mode: Optional[str] = None,
    sim_backend: Optional[str] = None,
    reward_mode: Optional[str] = None,
    control_mode: Optional[str] = None,
    reconfiguration_freq: int = 1,
    state_obs_slice: Optional[slice] = None,
) -> Any:
    """Match ``ppo_train_icldata.py`` / ``_mani_skill_env_kwargs`` + ``FlattenActionSpaceWrapper``."""
    ensure_maniskill_imported()
    import gymnasium as gym

    from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper

    task = maniskill_task_id(env_name)
    sim = sim_backend
    if sim is None or str(sim).strip() == "" or str(sim).strip() == "auto":
        sim = os.environ.get(
            "MANISKILL_EVAL_SIM", os.environ.get("MANISKILL_SMOKE_SIM", "physx_cuda")
        )
    if sim in ("", "auto"):
        sim = "physx_cuda"

    kw: dict[str, Any] = {
        "num_envs": 1,
        "obs_mode": "state",
        "sim_backend": sim,
        "reconfiguration_freq": int(reconfiguration_freq),
    }
    if render_mode is not None:
        kw["render_mode"] = render_mode
    if reward_mode is not None and str(reward_mode).strip():
        kw["reward_mode"] = reward_mode
    if control_mode is not None and str(control_mode).strip():
        kw["control_mode"] = control_mode

    last_err: Optional[BaseException] = None
    backends = [sim]
    if sim == "physx_cuda":
        backends.append("physx_cpu")
    for backend in backends:
        try:
            kw_try = {**kw, "sim_backend": backend}
            vec = gym.make(task, **kw_try)
            if isinstance(vec.action_space, gym.spaces.Dict):
                vec = FlattenActionSpaceWrapper(vec)
            return ManiSkillEvalAdapter(vec, state_obs_slice=state_obs_slice)
        except BaseException as e:
            last_err = e
    assert last_err is not None
    raise RuntimeError(
        f"Failed to create ManiSkill env {task!r} for eval (tried sim backends {backends!r}). "
        f"Set MANISKILL_EVAL_SIM=physx_cpu if CUDA/driver fails. Original: {last_err}"
    ) from last_err
