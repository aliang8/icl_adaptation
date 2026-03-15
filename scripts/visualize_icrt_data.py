#!/usr/bin/env python3
"""
Visualize ICRT-MT dataset: task distribution, episode lengths, and sample frames (exterior + wrist).

Usage:
  python scripts/visualize_icrt_data.py
  python scripts/visualize_icrt_data.py --config datasets/ICRT-MT/dataset_config.json --out-dir outputs/icrt_viz
  python scripts/visualize_icrt_data.py --max-episodes 3 --sample-frames 5
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:
    raise SystemExit("Install h5py: pip install h5py")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("Install matplotlib: pip install matplotlib")


# ICRT image format: 180 x 320 x 3 per frame (stored as bytes)
IMAGE_H, IMAGE_W, IMAGE_C = 180, 320, 3


def decode_image_frame(raw) -> np.ndarray:
    """Decode one frame from HDF5 (bytes or array) to (H, W, 3) uint8."""
    if hasattr(raw, "tobytes"):
        raw = raw.tobytes()
    if isinstance(raw, bytes):
        arr = np.frombuffer(raw, dtype=np.uint8)
    else:
        arr = np.asarray(raw, dtype=np.uint8).ravel()
    if arr.size != IMAGE_H * IMAGE_W * IMAGE_C:
        return np.zeros((IMAGE_H, IMAGE_W, IMAGE_C), dtype=np.uint8)
    return arr.reshape(IMAGE_H, IMAGE_W, IMAGE_C)


def task_name_from_episode_key(key: str) -> str:
    """Extract task name from key like real_episode_2024-05-31-close-drawer_0 -> close-drawer."""
    m = re.match(r"real_episode_\d{4}-\d{2}-\d{2}-(.+)_\d+", key)
    return m.group(1) if m else key


def load_keys_and_build_verb_to_episode(config_path: str):
    """Load HDF5 keys and build verb_to_episode from episode key names."""
    with open(config_path) as f:
        config = json.load(f)
    hdf5_keys = config["hdf5_keys"]
    if isinstance(hdf5_keys, str):
        hdf5_keys = [hdf5_keys]
    all_keys = []
    for p in hdf5_keys:
        with open(p) as f:
            all_keys.extend(json.load(f))
    verb_to_episode = defaultdict(list)
    for k in all_keys:
        task = task_name_from_episode_key(k)
        verb_to_episode[task].append(k)
    return config, list(all_keys), dict(verb_to_episode)


def get_episode_length(h5file, episode_key: str) -> int:
    """Return episode length (number of steps) from observation array."""
    grp = h5file[episode_key]
    obs = grp.get("observation", grp)
    for key in ("cartesian_position", "gripper_position", "exterior_image_1_left"):
        if key in obs:
            return len(obs[key])
    return 0


def main():
    parser = argparse.ArgumentParser(description="Visualize ICRT-MT dataset")
    parser.add_argument("--config", type=str, default="datasets/ICRT-MT/dataset_config.json")
    parser.add_argument("--out-dir", type=str, default="outputs/icrt_viz")
    parser.add_argument(
        "--max-episodes", type=int, default=2, help="Max episodes to load frames from (for speed)"
    )
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=3,
        help="Number of timesteps to show per episode (start, mid, end)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    config, all_keys, verb_to_episode = load_keys_and_build_verb_to_episode(str(config_path))
    dataset_paths = config["dataset_path"]
    if isinstance(dataset_paths, str):
        dataset_paths = [dataset_paths]
    image_keys = config.get(
        "image_keys", ["observation/exterior_image_1_left", "observation/wrist_image_left"]
    )

    # Use only keys that exist in first HDF5 (for simplicity we use part1 only if it has the key)
    h5path = dataset_paths[0]
    if not Path(h5path).exists():
        raise SystemExit(f"HDF5 not found: {h5path}")

    # 1) Task distribution (episodes per task)
    tasks = sorted(verb_to_episode.keys())
    counts = [len(verb_to_episode[t]) for t in tasks]

    # 2) Episode lengths (sample from first file only to avoid loading both)
    with open(
        config["hdf5_keys"][0] if isinstance(config["hdf5_keys"], list) else config["hdf5_keys"]
    ) as f:
        keys_part1 = json.load(f)
    with h5py.File(h5path, "r") as f:
        lengths = []
        for k in keys_part1[:500]:
            try:
                lengths.append(get_episode_length(f, k))
            except Exception:
                pass

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Plot 1: Tasks bar chart -----
    fig1, ax1 = plt.subplots(figsize=(14, 5))
    ax1.bar(range(len(tasks)), counts, color="steelblue", edgecolor="navy", alpha=0.8)
    ax1.set_xticks(range(len(tasks)))
    ax1.set_xticklabels(tasks, rotation=45, ha="right", fontsize=7)
    ax1.set_ylabel("Number of episodes")
    ax1.set_title("ICRT-MT: Episodes per task")
    fig1.tight_layout()
    fig1.savefig(out_dir / "tasks_distribution.png", dpi=120, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved {out_dir / 'tasks_distribution.png'}")

    # ----- Plot 2: Episode length histogram -----
    if lengths:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.hist(
            lengths, bins=min(50, len(set(lengths))), color="coral", edgecolor="darkred", alpha=0.8
        )
        ax2.set_xlabel("Episode length (steps)")
        ax2.set_ylabel("Count")
        ax2.set_title("ICRT-MT: Episode length distribution")
        fig2.tight_layout()
        fig2.savefig(out_dir / "episode_length_histogram.png", dpi=120, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved {out_dir / 'episode_length_histogram.png'}")

    # ----- Plot 3: Sample frames from one episode (exterior + wrist) -----
    # Use observation key names without "observation/" if that's how they're stored in the group
    obs_image_keys = [k.replace("observation/", "") for k in image_keys]
    with h5py.File(h5path, "r") as f:
        ep_keys = [k for k in keys_part1 if k in f][: args.max_episodes]
        for ep_idx, ep_key in enumerate(ep_keys):
            grp = f[ep_key]
            obs = grp.get("observation", grp)
            T = get_episode_length(f, ep_key)
            if T == 0:
                continue
            indices = [0, T // 2, T - 1][: args.sample_frames]
            n_views = len(obs_image_keys)
            n_cols = args.sample_frames
            fig3, axes = plt.subplots(n_views, n_cols, figsize=(3 * n_cols, 3 * n_views))
            if n_views == 1:
                axes = axes.reshape(1, -1)
            for view_idx, ok in enumerate(obs_image_keys):
                if ok not in obs:
                    continue
                img_ds = obs[ok]
                for col_idx, t in enumerate(indices):
                    raw = img_ds[t]
                    frame = decode_image_frame(raw)
                    ax = axes[view_idx, col_idx]
                    ax.imshow(frame)
                    ax.set_axis_off()
                    if col_idx == 0:
                        ax.set_ylabel(ok.replace("_", " "), fontsize=9)
                    if view_idx == 0:
                        ax.set_title(f"t={t}", fontsize=9)
            task = task_name_from_episode_key(ep_key)
            fig3.suptitle(f"Episode: {ep_key[:50]}...\nTask: {task}", fontsize=10)
            fig3.tight_layout()
            fig3.savefig(out_dir / f"sample_episode_{ep_idx}.png", dpi=100, bbox_inches="tight")
            plt.close(fig3)
            print(f"Saved {out_dir / f'sample_episode_{ep_idx}.png'}")

    print(f"Done. Visualizations in {out_dir}")


if __name__ == "__main__":
    main()
