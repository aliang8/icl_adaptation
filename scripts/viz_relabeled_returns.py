#!/usr/bin/env python3
"""
Visualize relabeled dataset: histogram of trajectory returns and example trajectories.

After reward relabeling (e.g. compute_dense_rewards.py), run this to:
  - Plot a histogram of returns (sum of rewards per trajectory).
  - Show a few example trajectories: first frame, cumulative return curve, language instruction, task id.

Uses Palatino font for all text. Outputs to --out-dir (default: data_dir/viz_relabeled).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if _project_root not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.utils import load_npz_arrays, read_mp4_frames


def _resolve_root(data_dir: str) -> Path:
    root = Path(data_dir).resolve()
    if (root / "LIBERO-Cosmos-Policy").is_dir():
        root = root / "LIBERO-Cosmos-Policy"
    return root


def _load_returns_and_meta(root: Path, reward_key: str, max_episodes: int | None):
    """Load per-episode returns and metadata from manifest + lowdim.npz. Yields (episode_id, return, task_description, task_id)."""
    import pandas as pd

    manifest_path = root / "manifest.parquet"
    episodes_dir = root / "episodes"
    if not manifest_path.is_file() or not episodes_dir.is_dir():
        raise FileNotFoundError(f"Need {manifest_path} and {episodes_dir}")
    df = pd.read_parquet(manifest_path)
    if "episode_id" not in df.columns:
        raise ValueError("manifest.parquet must have episode_id")
    task_col = "task_description" if "task_description" in df.columns else None
    task_id_col = "task_id" if "task_id" in df.columns else None

    if max_episodes is not None:
        df = df.head(max_episodes)

    for _, row in df.iterrows():
        ep_id = int(row["episode_id"])
        ep_dir = episodes_dir / f"{ep_id:06d}"
        lowdim_path = ep_dir / "lowdim.npz"
        if not lowdim_path.is_file():
            continue
        data = load_npz_arrays(lowdim_path)
        if reward_key not in data:
            continue
        rewards = np.asarray(data[reward_key], dtype=np.float64).reshape(-1)
        ret = float(np.sum(rewards))
        task_desc = str(row.get(task_col, "") or "") if task_col else ""
        task_id = int(row.get(task_id_col, -1)) if task_id_col else -1
        yield ep_id, ret, task_desc, task_id, rewards


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize relabeled dataset: return histogram and example trajectories."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Data root (e.g. output of convert_libero_hdf5_to_dataset, contains LIBERO-Cosmos-Policy/).",
    )
    parser.add_argument(
        "--reward-key",
        type=str,
        default="rewards",
        help="Key in lowdim.npz for reward array (default: rewards). Use dense_rewards_robometer_4b etc. if not overwritten.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Cap number of episodes for histogram (default: all).",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=4,
        help="Number of example trajectories to plot (default: 4).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for figures (default: <data-dir>/viz_relabeled).",
    )
    args = parser.parse_args()

    root = _resolve_root(args.data_dir)
    out_dir = Path(args.out_dir or str(root / "viz_relabeled"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect returns and metadata
    returns_list: list[float] = []
    meta_list: list[tuple[int, str, int, np.ndarray]] = []  # ep_id, task_desc, task_id, rewards
    for ep_id, ret, task_desc, task_id, rewards in _load_returns_and_meta(
        root, args.reward_key, args.max_episodes
    ):
        returns_list.append(ret)
        meta_list.append((ep_id, task_desc, task_id, rewards))

    if not returns_list:
        print("No episodes found with reward key", args.reward_key, flush=True)
        return

    returns_arr = np.array(returns_list)
    font_serif = ["Palatino", "Palatino Linotype", "DejaVu Serif"]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- Histogram of returns ---
    with plt.rc_context({"font.family": "serif", "font.serif": font_serif}):
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(
            returns_arr,
            bins=min(50, max(10, len(returns_arr) // 5)),
            color="steelblue",
            edgecolor="white",
            alpha=0.85,
        )
        ax.axvline(
            np.mean(returns_arr),
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"mean = {np.mean(returns_arr):.3f}",
        )
        ax.set_xlabel("Return (sum of rewards)")
        ax.set_ylabel("Count")
        ax.set_title(f"Returns after relabeling (n={len(returns_arr)}, key={args.reward_key})")
        ax.legend()
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        hist_path = out_dir / "relabel_returns_histogram.png"
        fig.savefig(hist_path, dpi=150)
        plt.close(fig)
    print("Saved", hist_path, flush=True)

    # --- Example trajectories: first frame, return plot, language instruction, task id ---
    n_ex = min(args.num_examples, len(meta_list))
    if n_ex == 0:
        return

    episodes_dir = root / "episodes"
    with plt.rc_context({"font.family": "serif", "font.serif": font_serif}):
        fig, axes = plt.subplots(n_ex, 2, figsize=(9, 3.2 * n_ex), squeeze=False)
        for i in range(n_ex):
            ep_id, task_desc, task_id, rewards = meta_list[i]
            ep_dir = episodes_dir / f"{ep_id:06d}"
            primary_path = ep_dir / "primary.mp4"

            # First frame
            ax_img = axes[i, 0]
            if primary_path.is_file():
                frames = read_mp4_frames(primary_path)
                if frames:
                    img = np.asarray(frames[0])
                    ax_img.imshow(img)
                else:
                    ax_img.set_facecolor("gray")
                    ax_img.text(
                        0.5, 0.5, "No frames", ha="center", va="center", transform=ax_img.transAxes
                    )
            else:
                ax_img.set_facecolor("gray")
                ax_img.text(
                    0.5, 0.5, "No video", ha="center", va="center", transform=ax_img.transAxes
                )
            ax_img.set_axis_off()
            short_desc = (
                (task_desc[:70] + "…") if len(task_desc) > 70 else (task_desc or "(no instruction)")
            )
            ax_img.set_title(f"Episode {ep_id}  ·  task_id = {task_id}\n{short_desc}", fontsize=9)

            # Cumulative return vs timestep (0–1 on y)
            ax_ret = axes[i, 1]
            T = len(rewards)
            cum = np.cumsum(rewards)
            total = cum[-1] if T > 0 else 0.0
            y = cum / (total + 1e-9) if total > 0 else np.zeros_like(cum)
            x = np.arange(T, dtype=np.int32)
            ax_ret.plot(x, y, color="steelblue", linewidth=1.5)
            ax_ret.set_xlabel("Timestep")
            ax_ret.set_ylabel("Cumulative return (0–1)")
            ax_ret.set_ylim(0, 1)
            ax_ret.set_title(f"Return = {total:.3f}")
            ax_ret.grid(True, alpha=0.3)

        fig.tight_layout()
        ex_path = out_dir / "relabel_examples.png"
        fig.savefig(ex_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print("Saved", ex_path, flush=True)


if __name__ == "__main__":
    main()
