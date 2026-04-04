#!/usr/bin/env python3
"""Load ``trajectories.h5`` or ``trajectories.pkl`` and save a histogram of episode returns (sum of rewards)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    p = argparse.ArgumentParser(description="Histogram of episode returns from trajectory file.")
    p.add_argument("trajectory_path", type=Path, help="Path to trajectories.h5 or trajectories.pkl")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG (default: <stem>_returns_hist.png beside the trajectory file).",
    )
    p.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins (default: 50).",
    )
    args = p.parse_args()

    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import numpy as np

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for the histogram.", file=sys.stderr)
        return 1

    from src.data.maniskill_io import load_trajectories_file
    from src.data.trajectories import trajectory_return

    traj_path = args.trajectory_path.resolve()
    if not traj_path.is_file():
        print(f"not a file: {traj_path}", file=sys.stderr)
        return 1

    trajs = load_trajectories_file(traj_path)
    returns: list[float] = []
    for i, t in enumerate(trajs):
        if t.get("rewards") is None:
            print(f"skip episode {i}: missing 'rewards'", file=sys.stderr)
            continue
        returns.append(float(trajectory_return(t)))

    if not returns:
        print("no valid trajectories with rewards", file=sys.stderr)
        return 1

    r = np.asarray(returns, dtype=np.float64)
    out_path = args.output
    if out_path is None:
        out_path = traj_path.with_name(f"{traj_path.stem}_returns_hist.png")
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(r, bins=int(args.bins), color="steelblue", edgecolor="white", alpha=0.9)
    ax.set_xlabel("episode return (sum of rewards)")
    ax.set_ylabel("count")
    ax.set_title(f"{traj_path.name}  (n={len(r)})")
    ax.axvline(float(r.mean()), color="darkorange", linestyle="--", linewidth=1.5, label=f"mean={r.mean():.4g}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(
        f"wrote {out_path} | n={len(r)} mean={r.mean():.6g} std={r.std():.6g} "
        f"min={r.min():.6g} max={r.max():.6g}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
