#!/usr/bin/env python3
"""Print summary statistics for ManiSkill ICL ``trajectories.h5`` (flat HDF5 v2).

Loads via :func:`src.data.ic_replay_buffer_hdf5.load_trajectories_file`, which expands the flat file into
the usual list of per-episode dicts, so the summaries below are unchanged from the old layout.

Optional: ``--sample-videos N`` writes up to ``N`` sample MP4s per file from trajectories that
contain non-zero RGB (skips state-only episodes stored as zero-filled image frames in mixed HDF5).
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# ManiSkill episode metrics; often bool but sometimes 0/1 after JSON/HDF5 round-trip.
_BOOLISH_META_KEYS = frozenset({"success_once", "success_at_end", "fail_once", "fail_at_end"})


def _meta_collect_scalar(
    em: dict[str, Any],
    meta_bool: dict[str, list[float]],
    meta_num: dict[str, list[float]],
) -> None:
    import numpy as np

    for k, v in em.items():
        if isinstance(v, (bool, np.bool_)):
            meta_bool[k].append(float(bool(v)))
            continue
        if (
            k in _BOOLISH_META_KEYS
            and isinstance(v, (int, np.integer, float, np.floating))
            and not isinstance(v, bool)
        ):
            fv = float(v)
            if fv in (0.0, 1.0):
                meta_bool[k].append(fv)
                continue
        if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
            meta_num[k].append(float(v))
            continue
        if isinstance(v, (float, np.floating)):
            meta_num[k].append(float(v))
            continue


def _summarize_trajectories(trajs: list, label: str) -> None:
    import numpy as np

    n = len(trajs)
    if n == 0:
        print(f"{label}: empty list")
        return

    lens: list[int] = []
    rets: list[float] = []
    n_img = 0
    img_shape = None
    obs_dim = None
    act_dim = None
    n_meta = 0
    meta_bool: dict[str, list[float]] = defaultdict(list)
    meta_num: dict[str, list[float]] = defaultdict(list)

    for t in trajs:
        o = t.get("observations")
        a = t.get("actions")
        r = t.get("rewards")
        if o is None or a is None or r is None:
            print(f"{label}: skip malformed traj (missing keys)", file=sys.stderr)
            continue
        lens.append(int(o.shape[0]))
        rets.append(float(np.asarray(r, dtype=np.float64).sum()))
        if obs_dim is None:
            obs_dim = int(o.shape[-1]) if o.ndim >= 2 else int(np.prod(o.shape))
        if act_dim is None:
            act_dim = int(a.shape[-1]) if a.ndim >= 2 else int(np.prod(a.shape))
        imgs = t.get("images")
        if imgs is not None and len(imgs) > 0:
            n_img += 1
            if img_shape is None:
                arr = np.asarray(imgs[0])
                img_shape = tuple(arr.shape)
        em = t.get("episode_meta")
        if isinstance(em, dict) and em:
            n_meta += 1
            _meta_collect_scalar(em, meta_bool, meta_num)

    if not lens:
        print(f"{label}: no valid trajectories")
        return

    la = np.asarray(lens, dtype=np.float64)
    ra = np.asarray(rets, dtype=np.float64)
    print(f"\n=== {label} ===")
    print(f"  trajectories:     {len(lens)}")
    print(
        f"  episode length:    mean={la.mean():.1f} std={la.std():.1f} "
        f"min={int(la.min())} max={int(la.max())}"
    )
    print(
        f"  return (sum r):    mean={ra.mean():.4f} std={ra.std():.4f} "
        f"min={ra.min():.4f} max={ra.max():.4f}"
    )
    if obs_dim is not None:
        print(f"  state_dim (first): {obs_dim}")
    if act_dim is not None:
        print(f"  action_dim (first): {act_dim}")
    print(f"  with 'images' key:  {n_img}/{len(lens)}")
    if img_shape is not None:
        print(f"  image array shape:  {img_shape} (first traj with images)")
    print(f"  with 'episode_meta': {n_meta}/{len(lens)}")
    if n_meta < len(lens):
        print(f"  episode_meta missing: {len(lens) - n_meta}/{len(lens)}")

    if meta_bool:
        print("  episode_meta (boolean-ish):")
        for k in sorted(meta_bool.keys()):
            vals = np.asarray(meta_bool[k], dtype=np.float64)
            c = len(vals)
            print(f"    {k}:  True rate={vals.mean():.4f}  (n={c} trajs with key)")

    if meta_num:
        print("  episode_meta (numeric):")
        for k in sorted(meta_num.keys()):
            vals = np.asarray(meta_num[k], dtype=np.float64)
            c = len(vals)
            print(
                f"    {k}:  mean={vals.mean():.4f} std={vals.std():.4f} "
                f"min={vals.min():.4f} max={vals.max():.4f}  (n={c})"
            )


def _traj_has_nonzero_rgb(t: dict[str, Any]) -> bool:
    """True if any image view has non-zero pixels (real RGB, not HDF5 zero padding)."""
    import numpy as np

    imgs = t.get("images")
    if not isinstance(imgs, list) or not imgs:
        return False
    for im in imgs:
        arr = np.asarray(im, dtype=np.uint8)
        if arr.ndim == 4 and int(arr.shape[-1]) == 3 and arr.size > 0 and arr.max() > 0:
            return True
    return False


def _safe_stem(path: Path) -> str:
    s = path.stem
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")
    return s or "trajectories"


def write_sample_trajectory_videos(
    trajs: list,
    *,
    h5_path: Path,
    out_dir: Path,
    num_episodes: int,
    fps: float,
) -> list[Path]:
    """Export up to ``num_episodes`` trajectories with non-zero RGB; one MP4 per camera view."""
    import numpy as np

    try:
        import imageio.v2 as imageio
    except ImportError as e:
        raise RuntimeError(
            "Writing sample videos requires imageio (and imageio-ffmpeg for MP4). "
            "Install with: pip install 'imageio[ffmpeg]'"
        ) from e

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = _safe_stem(h5_path)
    written: list[Path] = []
    exported_eps = 0
    max_eps = max(0, int(num_episodes))

    for ep_i, t in enumerate(trajs):
        if exported_eps >= max_eps:
            break
        if not _traj_has_nonzero_rgb(t):
            continue
        imgs = t.get("images")
        if not isinstance(imgs, list):
            continue
        any_view = False
        for v, im in enumerate(imgs):
            arr = np.asarray(im, dtype=np.uint8)
            if arr.ndim != 4 or int(arr.shape[-1]) != 3:
                continue
            if arr.size == 0 or arr.max() == 0:
                continue
            any_view = True
            out_path = out_dir / f"{tag}_ep{ep_i:04d}_view{v}.mp4"
            with imageio.get_writer(str(out_path), fps=float(fps)) as w:
                for ti in range(int(arr.shape[0])):
                    w.append_data(np.asarray(arr[ti]))
            written.append(out_path.resolve())
        if any_view:
            exported_eps += 1

    return written


def main() -> int:
    p = argparse.ArgumentParser(
        description="Print stats for ManiSkill trajectory .h5 files (flat v2).",
    )
    p.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="One or more trajectories.h5 paths (shell-expand globs yourself).",
    )
    p.add_argument(
        "--sample-videos",
        type=int,
        default=0,
        metavar="N",
        help=(
            "If N>0, export up to N episodes with non-zero RGB "
            "(one MP4 per camera view per episode)."
        ),
    )
    p.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help=(
            "Video output base dir. Default: <h5_dir>/<h5_stem>_sample_videos. "
            "If set: writes under <dir>/<h5_stem>/."
        ),
    )
    p.add_argument(
        "--video-fps",
        type=float,
        default=20.0,
        help="FPS for sample trajectory MP4s (default: 20).",
    )
    args = p.parse_args()

    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.data.ic_replay_buffer_hdf5 import load_trajectories_file

    for path in args.paths:
        path = path.resolve()
        if not path.is_file():
            print(f"skip (not a file): {path}", file=sys.stderr)
            continue
        try:
            trajs = load_trajectories_file(path)
        except Exception as e:
            print(f"{path}: failed to load ({e})", file=sys.stderr)
            continue
        _summarize_trajectories(trajs, str(path))

        if int(args.sample_videos) > 0:
            if args.video_dir is not None:
                vdir = Path(args.video_dir).resolve() / _safe_stem(path)
            else:
                vdir = path.parent / f"{path.stem}_sample_videos"
            try:
                outs = write_sample_trajectory_videos(
                    trajs,
                    h5_path=path,
                    out_dir=vdir,
                    num_episodes=int(args.sample_videos),
                    fps=float(args.video_fps),
                )
            except RuntimeError as e:
                print(f"{path}: sample videos: {e}", file=sys.stderr)
                continue
            if not outs:
                msg = (
                    f"{path}: sample videos: no non-zero RGB episodes (wrote 0 files under {vdir})"
                )
                print(msg, file=sys.stderr)
            else:
                print(f"{path}: wrote {len(outs)} sample video(s) under {vdir}:")
                for op in outs:
                    print(f"  {op}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
