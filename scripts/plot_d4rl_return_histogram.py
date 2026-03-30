#!/usr/bin/env python3
"""
Histogram of per-episode returns for a D4RL-style split (trajectories.pkl).

Expects the same layout as training: <data_root>/<env_name>/<data_quality>/trajectories.pkl
(e.g. from scripts/download_d4rl_halfcheetah.py).

Example:
  uv run python scripts/plot_d4rl_return_histogram.py \\
    --data-root /scr2/shared/icl_adaptation/datasets \\
    --env-name HalfCheetah-v2 \\
    --data-quality expert \\
    --output returns_expert.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Project root
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.d4rl_loader import load_halfcheetah_trajectories
from src.data.trajectories import trajectory_return


def main() -> None:
    p = argparse.ArgumentParser(description="Histogram of trajectory returns (D4RL-style pkl).")
    p.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root containing <env_name>/<data_quality>/trajectories.pkl",
    )
    p.add_argument("--env-name", type=str, default="HalfCheetah-v2")
    p.add_argument(
        "--data-quality",
        type=str,
        default="expert",
        help="Subfolder name, e.g. expert, medium, medium_expert, medium_replay",
    )
    p.add_argument("--max-trajectories", type=int, default=None)
    p.add_argument("--bins", type=int, default=50)
    p.add_argument(
        "--output",
        type=str,
        default="d4rl_return_histogram.png",
        help="Path to save the figure (PNG).",
    )
    args = p.parse_args()

    trajs, _ = load_halfcheetah_trajectories(
        args.data_root,
        env_name=args.env_name,
        data_quality=args.data_quality,
        max_trajectories=args.max_trajectories,
    )
    if not trajs:
        pkl = Path(args.data_root) / args.env_name / args.data_quality / "trajectories.pkl"
        raise SystemExit(f"No trajectories loaded. Expected file: {pkl}")

    returns = np.array([trajectory_return(t) for t in trajs], dtype=np.float64)
    print(
        f"Loaded {len(returns)} trajectories | "
        f"return: min={returns.min():.2f} max={returns.max():.2f} "
        f"mean={returns.mean():.2f} std={returns.std():.2f} median={np.median(returns):.2f}"
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(returns, bins=args.bins, color="steelblue", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Episode return (sum of rewards)")
    ax.set_ylabel("Count")
    ax.set_title(f"{args.env_name} / {args.data_quality}  (n={len(returns)})")
    ax.axvline(
        returns.mean(),
        color="darkred",
        linestyle="--",
        linewidth=1.2,
        label=f"mean={returns.mean():.1f}",
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
