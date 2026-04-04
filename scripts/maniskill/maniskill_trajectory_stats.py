#!/usr/bin/env python3
"""Load ManiSkill ICL trajectory pickles and print summary statistics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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
    success_end_vals: list[float] = []

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
            if "success_at_end" in em:
                success_end_vals.append(float(bool(em["success_at_end"])))

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
    if success_end_vals:
        sa = np.asarray(success_end_vals, dtype=np.float64)
        print(f"  success_at_end rate: {sa.mean():.4f} (over trajs with key)")


def main() -> int:
    p = argparse.ArgumentParser(description="Print stats for ManiSkill trajectory .h5 or .pkl files.")
    p.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="One or more trajectories.h5 / trajectories.pkl paths (shell-expand globs yourself).",
    )
    args = p.parse_args()

    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.data.maniskill_io import load_trajectories_file

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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
