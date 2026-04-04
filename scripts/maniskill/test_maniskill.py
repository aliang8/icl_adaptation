#!/usr/bin/env python3
"""Quick check: ManiSkill import, env creation, one short rollout (state + render).

By default tries GPU sim (``physx_cuda``), then falls back to ``physx_cpu`` if CUDA/driver
init fails (e.g. PyTorch built for newer CUDA than the host driver).

Override: ``MANISKILL_SMOKE_SIM=cpu`` or ``MANISKILL_SMOKE_SIM=cuda`` (no fallback).
"""

from __future__ import annotations

import os
import sys
from typing import Optional


def main() -> int:
    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
        from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
        from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    except ImportError as e:
        print(
            "ManiSkill not installed. Use a dedicated venv: "
            "pip install -r scripts/maniskill/requirements.txt",
            file=sys.stderr,
        )
        print(e, file=sys.stderr)
        return 1

    env_id = "PickCube-v1"
    mode = (os.environ.get("MANISKILL_SMOKE_SIM") or "auto").strip().lower()
    if mode not in ("auto", "cpu", "cuda"):
        print(
            f"MANISKILL_SMOKE_SIM must be auto, cpu, or cuda; got {mode!r}",
            file=sys.stderr,
        )
        return 1

    backends: list[str]
    if mode == "cpu":
        backends = ["physx_cpu"]
    elif mode == "cuda":
        backends = ["physx_cuda"]
    else:
        backends = ["physx_cuda", "physx_cpu"]

    env = None
    last_err: Optional[BaseException] = None
    for sim_backend in backends:
        try:
            env = gym.make(
                env_id,
                num_envs=1,
                reconfiguration_freq=1,
                obs_mode="state",
                render_mode="rgb_array",
                sim_backend=sim_backend,
            )
            if sim_backend == "physx_cpu" and "physx_cuda" in backends:
                print(
                    "[maniskill test] Using physx_cpu "
                    "(GPU sim unavailable or driver/CUDA mismatch).",
                    file=sys.stderr,
                )
            break
        except RuntimeError as e:
            last_err = e
            msg = str(e).lower()
            if sim_backend == "physx_cuda" and "physx_cpu" in backends and (
                "nvidia" in msg or "cuda" in msg or "driver" in msg
            ):
                print(f"[maniskill test] physx_cuda failed: {e}", file=sys.stderr)
                continue
            raise
    if env is None:
        assert last_err is not None
        raise last_err
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, num_envs=1, ignore_terminations=False, record_metrics=True)

    obs, _ = env.reset(seed=0)
    import numpy as np
    import torch

    low = torch.from_numpy(env.single_action_space.low).float()
    high = torch.from_numpy(env.single_action_space.high).float()
    for t in range(5):
        a = torch.zeros((1, int(np.prod(env.single_action_space.shape))), dtype=torch.float32)
        a = torch.clamp(a, low, high)
        obs, rew, term, trunc, _ = env.step(a)
        rgb = env.render()
        r0 = float(rew.view(-1)[0])
        sh = getattr(rgb, "shape", type(rgb))
        print(f"step {t} reward={r0:.4f} obs={tuple(obs.shape)} rgb={sh}")
        if bool(term.view(-1)[0]) or bool(trunc.view(-1)[0]):
            break
    env.close()
    print("ManiSkill test OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
