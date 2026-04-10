#!/usr/bin/env python3
"""Load ``trajectories.h5`` (flat v2) and plot episode returns: histogram + sorted line.

Optional ``--max-episode-steps`` keeps only episodes with ``len(rewards) == T``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Plots of episode returns from a flat v2 trajectory .h5 (histogram + sorted line)."
    )
    p.add_argument("trajectory_path", type=Path, help="Path to trajectories.h5")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Histogram PNG (default: <stem>_returns_hist.png beside the trajectory file).",
    )
    p.add_argument(
        "--sorted-output",
        type=Path,
        default=None,
        help="Sorted line-plot PNG (default: <stem>_returns_sorted.png beside the trajectory file).",
    )
    p.add_argument(
        "--no-sorted-plot",
        action="store_true",
        help="Skip the sorted-by-return line plot.",
    )
    p.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins (default: 50).",
    )
    p.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        metavar="T",
        help="If set, only episodes with len(rewards) == T (e.g. match env / AD horizon).",
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
    want_T = args.max_episode_steps
    returns: list[float] = []
    skipped_len = 0
    for i, t in enumerate(trajs):
        if t.get("rewards") is None:
            print(f"skip episode {i}: missing 'rewards'", file=sys.stderr)
            continue
        T = int(np.asarray(t["rewards"], dtype=np.float64).reshape(-1).shape[0])
        if want_T is not None and T != int(want_T):
            skipped_len += 1
            continue
        returns.append(float(trajectory_return(t)))

    if not returns:
        print("no valid trajectories with rewards", file=sys.stderr)
        if want_T is not None:
            print(
                f"(filter max_episode_steps={want_T} excluded all {len(trajs)} episodes)",
                file=sys.stderr,
            )
        return 1
    if want_T is not None and skipped_len:
        print(
            f"filter max_episode_steps={want_T}: kept {len(returns)} / {len(trajs)} episodes "
            f"({skipped_len} wrong length)",
            file=sys.stderr,
        )

    r = np.asarray(returns, dtype=np.float64)
    r_sorted = np.sort(r)

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
    ax.axvline(
        float(r.mean()),
        color="darkorange",
        linestyle="--",
        linewidth=1.5,
        label=f"mean={r.mean():.4g}",
    )
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

    if not args.no_sorted_plot:
        sorted_out = args.sorted_output
        if sorted_out is None:
            sorted_out = traj_path.with_name(f"{traj_path.stem}_returns_sorted.png")
        sorted_out = sorted_out.resolve()
        sorted_out.parent.mkdir(parents=True, exist_ok=True)
        x = np.arange(len(r_sorted), dtype=np.float64)
        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
        ax2.plot(x, r_sorted, color="darkgreen", linewidth=1.2)
        ax2.set_xlabel("episode index (sorted by return, lowest → highest)")
        ax2.set_ylabel("return (sum of rewards)")
        ax2.set_title(f"{traj_path.name}  sorted returns  (n={len(r_sorted)})")
        ax2.grid(True, alpha=0.25)
        ax2.margins(x=0.02)
        fig2.tight_layout()
        fig2.savefig(sorted_out, dpi=150)
        plt.close(fig2)
        print(f"wrote {sorted_out} (sorted line plot)", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
