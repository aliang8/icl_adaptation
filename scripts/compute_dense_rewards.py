#!/usr/bin/env python3
"""
Compute dense reward (per-step) signals for each episode and write them into lowdim.npz.

Works with any dataset that has manifest.parquet + episodes/ (e.g. output of
convert_libero_hdf5_to_dataset.py or convert_roboarena_to_dataset.py).

Output:
  - Adds arrays to lowdim.npz (keeps existing arrays):
      dense_rewards_robometer_4b: float32 [T]
      dense_rewards_robodopamine_8b: float32 [T]
  - Optionally overwrites lowdim.npz['rewards'] with one of the dense signals
    (--overwrite-rewards).

Notes:
  - Robometer: episode is split into fixed-size frame chunks; each chunk is one model input.
  - RoboDopamine: per-trajectory (internally batches prompts).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if _project_root not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.utils import (
    ensure_uint8_rgb_frames,
    load_npz_arrays,
    read_mp4_frames,
    save_npz_arrays,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute dense reward signals for episodes (manifest.parquet + episodes/)."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Dataset root containing manifest.parquet and episodes/ (e.g. from convert_libero_hdf5_to_dataset or convert_roboarena_to_dataset).",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=["robodopamine_8b", "robometer_4b"],
        help="Dense reward models to compute (default: robodopamine_8b robometer_4b).",
    )
    parser.add_argument(
        "--overwrite-rewards",
        type=str,
        default="robometer_4b",
        help="If set to one of the computed models, overwrite lowdim.npz['rewards'] with that dense signal. Use 'none' to keep original.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        help="Episode subset: 'all' or 'firstN' where N is provided via --max-episodes.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Limit number of episodes processed (default: all).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for Robometer model inference (default: cuda:0).",
    )
    parser.add_argument(
        "--cam",
        type=str,
        default="primary",
        choices=["primary", "wrist"],
        help="Which camera view to feed reward models with (default: primary).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on number of frames fed to reward models (default: use episode length).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Inference batch size: frames per segment (Robometer) / prompts per batch (RoboDopamine). Default 32.",
    )
    parser.add_argument(
        "--dopamine-frame-interval",
        type=int,
        default=1,
        help="RoboDopamine: frame sampling interval (default: 1). Larger values speed up but reduce temporal resolution.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=True,
        help="Print input shapes and task instructions for first episode and first Robometer batch (default: True).",
    )
    parser.add_argument(
        "--no-debug",
        dest="debug",
        action="store_false",
        help="Disable debug prints.",
    )

    args = parser.parse_args()

    root = Path(args.data_dir).resolve()
    manifest_path = root / "manifest.parquet"
    episodes_dir = root / "episodes"
    if not manifest_path.is_file() or not episodes_dir.is_dir():
        raise FileNotFoundError(
            f"Expected manifest.parquet and episodes/ under {root}. Run a convert script first (e.g. convert_libero_hdf5_to_dataset or convert_roboarena_to_dataset)."
        )

    import pandas as pd
    from tqdm import tqdm

    df = pd.read_parquet(manifest_path)
    if "episode_id" not in df.columns:
        raise ValueError("manifest.parquet must contain episode_id")
    if "task_description" not in df.columns:
        df["task_description"] = ""

    if args.max_episodes is not None:
        df = df.head(args.max_episodes)

    # Prepare reward model instances (dataset-agnostic models live in reward_relabeling)
    from src.reward_relabeling import build_reward_model, pad_or_trunc_1d

    device = args.device
    reward_models = {}
    for m in args.models:
        reward_models[m] = build_reward_model(
            model_name=m,
            device=device,
            batch_size=args.batch_size,
            dopamine_frame_interval=args.dopamine_frame_interval,
        )

    overwrite = None if args.overwrite_rewards.lower() == "none" else args.overwrite_rewards

    cam_fname = "primary.mp4" if args.cam == "primary" else "wrist.mp4"

    def _read_episode_frames(ep_dir: Path) -> np.ndarray:
        frames_path = ep_dir / cam_fname
        frames_list = read_mp4_frames(frames_path)
        if not frames_list:
            return np.zeros((0, 1, 1, 3), dtype=np.uint8)
        frames_arr = np.stack(frames_list, axis=0)
        frames_arr = ensure_uint8_rgb_frames(frames_arr)
        if args.max_frames is not None and frames_arr.shape[0] > args.max_frames:
            frames_arr = frames_arr[: args.max_frames]
        return frames_arr

    episode_rows = df.to_dict(orient="records")
    debug_printed = False

    for row in tqdm(episode_rows, desc="Computing dense rewards", unit="ep"):
        task = str(row.get("task_description", "") or "")
        if row.get("lowdim_path"):
            lowdim_path = root / row["lowdim_path"]
            ep_dir = lowdim_path.parent
        else:
            ep_dir = episodes_dir / f"{int(row['episode_id']):06d}"
            lowdim_path = ep_dir / "lowdim.npz"
        arrays = load_npz_arrays(lowdim_path)
        T = int(arrays["proprio"].shape[0])
        frames_arr = _read_episode_frames(ep_dir)

        if args.debug and not debug_printed and frames_arr.shape[0] > 0:
            debug_printed = True
            print(
                f"[debug] ep_id={row['episode_id']} frames.shape={frames_arr.shape} T={T} task={repr(task[:60])}...",
                flush=True,
            )

        for model_name, model in reward_models.items():
            key = f"dense_rewards_{model.name}"
            if frames_arr.shape[0] == 0:
                arrays[key] = np.zeros((T,), dtype=np.float32)
            else:
                dense = model.compute_rewards_one(frames=frames_arr, task=task)
                arrays[key] = pad_or_trunc_1d(dense, T).astype(np.float32)
            if overwrite == model_name:
                arrays["rewards"] = arrays[key]

        save_npz_arrays(lowdim_path, arrays)
    print("Dense reward computation complete.", flush=True)


if __name__ == "__main__":
    main()
