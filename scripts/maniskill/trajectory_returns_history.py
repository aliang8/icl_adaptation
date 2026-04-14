#!/usr/bin/env python3
"""Load ``trajectories.h5`` (flat v2) and write **one** PNG beside it with three panels.

``--min-traj-len`` (default 10) keeps only episodes with ``len(rewards) >= min_traj_len``.

**X-axis = episode index** in **HDF5 / file order** (the order episodes appear after
``load_trajectories_file``). That matches how episodes were appended when the bundle was written
(e.g. PPO ICL stitch: completions are recorded in rollout loop order, not a single-env timeline).

Panels (top to bottom):

1. Episode index vs **return** (sum of rewards).
2. Episode index vs **rolling ``success_once`` rate** (%), over the last ``--rolling-window``
   episodes (``nan`` where the window has no ``episode_meta`` success flag).
3. **Histogram** of returns with ``--bins`` bins (default 10).

Prints ``episode_meta`` success summaries to stderr when present.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _boolish_from_meta(em: Any, *keys: str) -> Optional[float]:
    """Read first present key from ``episode_meta``; return 0/1 float or ``None``."""
    if not isinstance(em, dict):
        return None
    import numpy as np

    for key in keys:
        v = em.get(key)
        if v is None:
            continue
        if isinstance(v, (bool, np.bool_)):
            return 1.0 if v else 0.0
        if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
            if int(v) in (0, 1):
                return float(int(v))
            continue
        if isinstance(v, (float, np.floating)):
            fv = float(v)
            if abs(fv) < 1e-12:
                return 0.0
            if abs(fv - 1.0) < 1e-12:
                return 1.0
    return None


def _summarize_success_flags(
    name: str,
    values: List[Optional[float]],
    n_kept: int,
) -> Tuple[str, Optional[float], int]:
    """Return (one-line summary, mean or None, count with key)."""
    present = [v for v in values if v is not None]
    if not present:
        return (
            f"{name}: no `episode_meta` / key missing for all {n_kept} kept episodes",
            None,
            0,
        )
    import numpy as np

    a = np.asarray(present, dtype=np.float64)
    mean = float(a.mean())
    return (
        f"{name}: mean={mean:.4f}  (episodes with key {len(present)}/{n_kept})",
        mean,
        len(present),
    )


def _rolling_success_pct(flags: List[Optional[float]], window: int):
    """Per-episode rolling mean of known 0/1 flags × 100; entries with no data in window are NaN."""
    import numpy as np

    n = len(flags)
    arr = np.array([np.nan if f is None else float(f) for f in flags], dtype=np.float64)
    out = np.full(n, np.nan, dtype=np.float64)
    w = max(1, int(window))
    for i in range(n):
        lo = max(0, i - w + 1)
        chunk = arr[lo : i + 1]
        if np.all(np.isnan(chunk)):
            continue
        out[i] = 100.0 * float(np.nanmean(chunk))
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Three panels beside the .h5: episode vs return, episode vs rolling success_once %, "
            "and a return histogram (default 10 bins)."
        )
    )
    p.add_argument("trajectory_path", type=Path, help="Path to trajectories.h5")
    p.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of histogram bins (default: 10).",
    )
    p.add_argument(
        "--rolling-window",
        type=int,
        default=50,
        metavar="W",
        help="Episodes in the rolling window for success_once %% (default: 50).",
    )
    p.add_argument(
        "--min-traj-len",
        type=int,
        default=10,
        metavar="L",
        help="Only include episodes with len(rewards) >= L (default: 10).",
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
        print("matplotlib is required.", file=sys.stderr)
        return 1

    from src.data.ic_replay_buffer_hdf5 import load_trajectories_file
    from src.data.trajectories import trajectory_return

    traj_path = args.trajectory_path.resolve()
    if not traj_path.is_file():
        print(f"not a file: {traj_path}", file=sys.stderr)
        return 1

    trajs = load_trajectories_file(traj_path)

    min_len = int(args.min_traj_len)
    returns: list[float] = []
    success_once: List[Optional[float]] = []
    success_at_end: List[Optional[float]] = []
    skipped_short = 0
    for i, t in enumerate(trajs):
        if t.get("rewards") is None:
            print(f"skip episode {i}: missing 'rewards'", file=sys.stderr)
            continue
        T = int(np.asarray(t["rewards"], dtype=np.float64).reshape(-1).shape[0])
        if T < min_len:
            skipped_short += 1
            continue
        returns.append(float(trajectory_return(t)))
        em = t.get("episode_meta")
        success_once.append(_boolish_from_meta(em, "success_once"))
        success_at_end.append(_boolish_from_meta(em, "success_at_end", "success_end"))

    if not returns:
        print("no valid trajectories with rewards", file=sys.stderr)
        if min_len > 0:
            print(
                f"(filter min_traj_len={min_len} excluded all {len(trajs)} episodes)",
                file=sys.stderr,
            )
        return 1
    if min_len > 0 and skipped_short:
        print(
            f"filter min_traj_len={min_len}: kept {len(returns)} / {len(trajs)} episodes "
            f"({skipped_short} shorter than L)",
            file=sys.stderr,
        )

    n_kept = len(returns)
    print("episode_meta success (same kept episodes as plots):", file=sys.stderr)
    line_so, m_so, c_so = _summarize_success_flags("success_once", success_once, n_kept)
    print(f"  {line_so}", file=sys.stderr)
    line_se, m_se, c_se = _summarize_success_flags(
        "success_at_end (or success_end)", success_at_end, n_kept
    )
    print(f"  {line_se}", file=sys.stderr)
    title_xtra = ""
    bits: list[str] = []
    if c_so > 0 and m_so is not None:
        bits.append(f"success_once={m_so:.1%}")
    if c_se > 0 and m_se is not None:
        bits.append(f"success_at_end={m_se:.1%}")
    if bits:
        title_xtra = " · ".join(bits)

    r = np.asarray(returns, dtype=np.float64)
    n_ep = int(r.shape[0])
    x_ep = np.arange(n_ep, dtype=np.float64)
    roll_w = int(args.rolling_window)
    roll_pct = _rolling_success_pct(success_once, roll_w)

    out_path = traj_path.with_name(f"{traj_path.stem}_returns_overview.png").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)
    ax0, ax1, ax2 = axes

    ax0.plot(x_ep, r, color="#1a5276", linewidth=1.2, marker="o", markersize=2.5)
    ax0.set_ylabel("return")
    ax0.set_title("Return vs episode index (HDF5 / export order)")
    ax0.set_xlabel("episode index (kept episodes, file order)")
    ax0.grid(True, alpha=0.25)
    ax0.margins(x=0.02)

    ax1.plot(x_ep, roll_pct, color="seagreen", linewidth=1.4)
    ax1.set_ylabel("rolling success_once (%)")
    ax1.set_title(f"Rolling mean of success_once (window={roll_w} episodes; NaN if no meta in window)")
    ax1.set_xlabel("episode index (same order as above)")
    ax1.set_ylim(0.0, 105.0)
    ax1.grid(True, alpha=0.25)
    ax1.margins(x=0.02)

    nb = max(1, int(args.bins))
    ax2.hist(r, bins=nb, color="steelblue", edgecolor="white", alpha=0.9)
    ax2.set_xlabel("episode return (sum of rewards)")
    ax2.set_ylabel("count")
    ax2.set_title(f"Histogram of returns ({nb} bins)")
    ax2.axvline(
        float(r.mean()),
        color="darkorange",
        linestyle="--",
        linewidth=1.5,
        label=f"mean={r.mean():.4g}",
    )
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.25)

    st = f"{traj_path.name}  (n={len(r)})"
    if title_xtra:
        st += f"  |  {title_xtra}"
    fig.suptitle(st, fontsize=11)
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
