"""
Save short debug videos from training data (query window + RTG + mask + prompt stats).
Used when the dataset carries images (V-D4RL, LIBERO-style lazy ICL, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import numpy as np
from loguru import logger as log

from src.engine.eval_viz import _annotate_eval_frame


def _chw01_to_uint8_hwc(arr: np.ndarray) -> np.ndarray:
    """(3,H,W) float [0,1] or [0,255] or uint8 -> (H,W,3) uint8."""
    x = np.asarray(arr)
    if x.ndim != 3:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    if x.shape[0] == 3:
        x = np.transpose(x, (1, 2, 0))
    if x.dtype == np.uint8:
        return x
    xf = x.astype(np.float32)
    if xf.max() > 1.5:
        xf = xf / 255.0
    xf = np.clip(xf, 0.0, 1.0)
    return (xf * 255.0).astype(np.uint8)


def _write_mp4(frames: List[np.ndarray], path: Path, fps: int = 8) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError:
        log.warning("imageio not installed; cannot write {}", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=fps) as w:
        for f in frames:
            w.append_data(np.asarray(f))


def save_training_sample_videos(
    run_dir: Path,
    dataset: Any,
    *,
    rtg_scale: float,
    num_clips: int = 3,
    fps: int = 8,
) -> None:
    """
    For ICL lazy datasets with `_get_one_sample`, save videos of the query window
    (first view) with RTG and mask overlays, plus prompt valid-step counts.

    Writes under ``run_dir / viz / training_sample_debug /``.
    """
    if num_clips <= 0:
        return
    trajs = dataset.trajectories
    if not trajs:
        log.info("Training sample videos: no trajectories on dataset; skipping.")
        return

    out_dir = run_dir / "viz" / "training_sample_debug"
    clip_id = 0
    for traj_idx, traj in enumerate(trajs):
        if clip_id >= num_clips:
            break
        if not isinstance(traj, dict) or "images" not in traj:
            continue
        if not isinstance(traj["images"], list) or not traj["images"]:
            continue
        T = int(traj["rewards"].shape[0])
        if T < 2:
            continue
        si = max(1, min(T - 2, T // 2))
        try:
            sample = dataset._get_one_sample(traj_idx, si)
        except Exception as e:
            log.warning("Training sample videos: _get_one_sample failed traj={}: {}", traj_idx, e)
            continue

        (
            _s,
            _c,
            _a,
            _r,
            _d,
            rtg_seg,
            ts_seg,
            mask_seg,
            _qtrial,
            _ps,
            _pa,
            _pr,
            _prtg,
            _pts,
            pm,
            _pptrial,
            _instr,
            images_out,
        ) = sample
        if images_out is None or not images_out:
            continue

        view0 = np.asarray(images_out[0])
        if view0.ndim != 4:
            continue
        K = view0.shape[0]
        prompt_valid = float(np.sum(np.asarray(pm) > 0))
        prompt_total = float(np.asarray(pm).shape[0])
        frames: List[np.ndarray] = []
        rarr = np.asarray(rtg_seg)
        tarr = np.asarray(ts_seg)
        marr = np.asarray(mask_seg).reshape(-1)
        for t in range(K):
            img = _chw01_to_uint8_hwc(view0[t])
            ti = min(t, marr.shape[0] - 1)
            m = float(marr[ti])
            ri = min(t, rarr.shape[0] - 1)
            rtg = float(rarr[ri, 0]) if rarr.ndim == 2 else float(rarr[ri])
            ts = float(tarr[ri, 0]) if tarr.ndim == 2 else float(tarr[ri])
            lines = [
                f"train sample | traj={traj_idx} si={si} q_t={t}/{K - 1}",
                f"env_t={ts:.0f}  mask={m:.0f}  RTG(/rtg_scale)={rtg:.4f}",
                f"context valid steps={prompt_valid:.0f}/{prompt_total:.0f}  rtg_scale={rtg_scale}",
            ]
            frames.append(_annotate_eval_frame(img, lines))
        path = out_dir / f"clip_{clip_id:02d}_traj{traj_idx:05d}_si{si:04d}.mp4"
        _write_mp4(frames, path, fps=fps)
        log.info("Wrote training debug video: {}", path)
        clip_id += 1

    if clip_id == 0:
        log.info(
            "Training sample videos: no clips written (need lazy ICL + trajectories with 'images')."
        )
