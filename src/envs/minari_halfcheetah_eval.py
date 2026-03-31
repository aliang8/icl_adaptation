"""
HalfCheetah eval env aligned with Minari: ``load_dataset`` + ``recover_environment()``.

See https://minari.farama.org/datasets/mujoco/halfcheetah/medium-v0/ (HalfCheetah-v5, same
specs for train and eval). Falls back to ``gym.make("HalfCheetah-v5", ...)`` if Minari or the
dataset is unavailable.

Minari exposes a single Gymnasium env from ``recover_environment()`` (no vectorized / batched
recover API in the dataset docs). Eval stays **sequential**: one env, ``num_rollouts`` independent
episodes in ``run_rollouts_and_save_viz`` (or ``eval_num_trials`` sequential zero-shot trials).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# Matches scripts/download_d4rl_halfcheetah.py MINARI_HALFCHEETAH (single-quality only).
HALFCHEETAH_QUALITY_TO_MINARI_ID: Dict[str, str] = {
    "medium": "mujoco/halfcheetah/medium-v0",
    "expert": "mujoco/halfcheetah/expert-v0",
    "simple": "mujoco/halfcheetah/simple-v0",
    "medium_replay": "mujoco/halfcheetah/medium-replay-v0",
}


def resolve_minari_halfcheetah_eval_id(data_quality: str) -> Optional[str]:
    """Map ``data_quality`` to Minari id, or None -> plain ``gym.make`` (e.g. medium_expert).

    Comma-separated qualities: return the first segment that has a Minari mapping (e.g.
    ``random,medium`` -> medium).
    """
    raw = (data_quality or "").strip()
    if not raw:
        return None
    for seg in raw.split(","):
        q = seg.strip().lower().replace("-", "_")
        if not q:
            continue
        hit = HALFCHEETAH_QUALITY_TO_MINARI_ID.get(q)
        if hit is not None:
            return hit
    return None


def make_halfcheetah_env_via_minari(
    dataset_id: str,
    render_mode: Optional[str] = None,
    *,
    download_if_missing: bool = False,
) -> Any:
    """Build env via Minari ``recover_environment``; fall back to Gymnasium v5."""
    import gymnasium as gym

    def _gym_v5() -> Any:
        if render_mode is not None:
            return gym.make("HalfCheetah-v5", render_mode=render_mode)
        return gym.make("HalfCheetah-v5")

    try:
        import minari
    except ImportError:
        print(
            "[eval] minari not installed; using gym.make('HalfCheetah-v5'). "
            "Install minari to match offline HalfCheetah recover_environment().",
            flush=True,
        )
        return _gym_v5()

    try:
        ds = minari.load_dataset(dataset_id, download=download_if_missing)
    except Exception as e:
        print(
            f"[eval] Minari load_dataset({dataset_id!r}) failed ({e}); "
            f"using gym.make('HalfCheetah-v5').",
            flush=True,
        )
        return _gym_v5()

    env = ds.recover_environment()
    if render_mode is None:
        print(
            f"[eval] HalfCheetah env from Minari recover_environment({dataset_id!r})",
            flush=True,
        )
        return env

    spec = env.spec
    if spec is None:
        eid = None
        kwargs: Dict[str, Any] = {}
    else:
        eid = spec.id
        kwargs = dict(spec.kwargs or {})
    try:
        env.close()
    except Exception:
        pass
    if eid:
        print(
            f"[eval] HalfCheetah: Minari spec {eid!r} + render_mode={render_mode!r} "
            f"(dataset {dataset_id!r})",
            flush=True,
        )
        return gym.make(eid, render_mode=render_mode, **kwargs)
    return _gym_v5()
