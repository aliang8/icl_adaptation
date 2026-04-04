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


def ensure_maniskill_imported() -> None:
    if importlib.util.find_spec("mani_skill") is None:
        raise RuntimeError(
            "ManiSkill eval needs `mani-skill` in the current Python env.\n"
            "  pip install -r scripts/maniskill/requirements.txt\n"
            "Or train from `.venv-maniskill` with ICL deps: "
            "scripts/maniskill/requirements_icl_train.txt (see docs/MANISKILL.md)."
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


class ManiSkillEvalAdapter:
    """Wraps ManiSkill vector env (``num_envs=1``) for eval_viz single-env numpy API."""

    def __init__(self, vec_env: Any) -> None:
        self._env = vec_env
        self._device = getattr(vec_env, "device", None)
        if not hasattr(vec_env, "single_observation_space"):
            raise TypeError(
                f"Expected ManiSkill vector env with single_observation_space; got {type(vec_env)}"
            )
        self.observation_space = vec_env.single_observation_space
        self.action_space = vec_env.single_action_space

    def reset(self, *, seed: Optional[int] = None, options: Any = None) -> Tuple[np.ndarray, dict]:
        obs, info = self._env.reset(seed=seed, options=options)
        return _obs_to_numpy_vector(obs), info

    def step(self, action_np: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        import torch

        dev = self._device or torch.device("cpu")
        a = torch.as_tensor(np.asarray(action_np, dtype=np.float32), device=dev).reshape(1, -1)
        o, r, term, trunc, info = self._env.step(a)
        o_np = _obs_to_numpy_vector(o)
        r_f = _first_float(r)
        done = _first_bool(term) or _first_bool(trunc)
        return o_np, r_f, done, False, info

    def render(self) -> Any:
        return self._env.render()

    def close(self) -> None:
        self._env.close()

    def get_current_images(self) -> Tuple[None, None]:
        return None, None


def make_maniskill_eval_env(
    env_name: str,
    *,
    render_mode: Optional[str] = None,
) -> Any:
    ensure_maniskill_imported()
    import gymnasium as gym

    task = maniskill_task_id(env_name)
    sim = os.environ.get("MANISKILL_EVAL_SIM", os.environ.get("MANISKILL_SMOKE_SIM", "physx_cuda"))
    if sim in ("", "auto"):
        sim = "physx_cuda"

    kw: dict[str, Any] = {
        "num_envs": 1,
        "obs_mode": "state",
        "sim_backend": sim,
        "reconfiguration_freq": 1,
    }
    if render_mode is not None:
        kw["render_mode"] = render_mode

    last_err: Optional[BaseException] = None
    backends = [sim]
    if sim == "physx_cuda":
        backends.append("physx_cpu")
    for backend in backends:
        try:
            kw_try = {**kw, "sim_backend": backend}
            vec = gym.make(task, **kw_try)
            return ManiSkillEvalAdapter(vec)
        except BaseException as e:
            last_err = e
    assert last_err is not None
    raise RuntimeError(
        f"Failed to create ManiSkill env {task!r} for eval (tried sim backends {backends!r}). "
        f"Set MANISKILL_EVAL_SIM=physx_cpu if CUDA/driver fails. Original: {last_err}"
    ) from last_err
