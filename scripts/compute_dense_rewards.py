#!/usr/bin/env python3
"""
Compute dense reward (per-step) signals for each LIBERO episode using robometer baselines/models,
and write them back into episodes/{id}/lowdim.npz.

This is an *offline preprocessing* step intended for the LIBERO pipeline:
  convert_libero_hdf5_to_dataset.py
  -> compute_dense_rewards.py   (this script)
  -> build_libero_sample_index.py
  -> precompute_libero_embeddings.py

Output:
  - Adds arrays to lowdim.npz (do not delete existing arrays):
      dense_rewards_robometer_4b: float32 [T]
      dense_rewards_robodopamine_8b: float32 [T]
  - Optionally overwrites lowdim.npz['rewards'] with one of the dense signals for training
    (controlled by --overwrite-rewards).

Notes:
  - Robometer dense rewards are computed in batches across trajectories.
  - RoboDopamine baseline is per-trajectory (but internally batches prompts).
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
        description="Compute dense reward signals for LIBERO episodes using RoboDopamine-8B and Robometer."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Output directory from convert_libero_hdf5_to_dataset.py (contains LIBERO-Cosmos-Policy/).",
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
    # Robometer batching
    parser.add_argument(
        "--robometer-batch-size",
        type=int,
        default=8,
        help="Batch size across trajectories for Robometer inference.",
    )
    # RoboDopamine controls
    parser.add_argument(
        "--dopamine-frame-interval",
        type=int,
        default=1,
        help="Frame interval for RoboDopamine sampling (default: 1). Larger values speed up but reduce temporal resolution.",
    )
    parser.add_argument(
        "--dopamine-batch-size",
        type=int,
        default=4,
        help="Batch size for RoboDopamine VLLM prompts inside each trajectory (default: 4).",
    )

    args = parser.parse_args()

    # Resolve dataset root
    root = Path(args.data_dir).resolve()
    if (root / "LIBERO-Cosmos-Policy").is_dir():
        root = root / "LIBERO-Cosmos-Policy"
    manifest_path = root / "manifest.parquet"
    episodes_dir = root / "episodes"
    if not manifest_path.is_file() or not episodes_dir.is_dir():
        raise FileNotFoundError(
            f"Expected manifest.parquet and episodes/ under {root}. Run convert_libero_hdf5_to_dataset.py first."
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
            robodopamine_frame_interval=args.dopamine_frame_interval,
            robodopamine_batch_size=args.dopamine_batch_size,
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

    # Process episodes. For Robometer we batch across trajectories inside each chunk.
    # For Dopamine we still do per-trajectory due to baseline constraints.
    episode_rows = df.to_dict(orient="records")
    n_episodes = len(episode_rows)

    # We do a single pass and compute both models as we go; for Robometer batching,
    # we chunk frames/tasks and then assign per-episode outputs.
    # Chunking is only for Robometer.
    robometer_name = "robometer_4b" if "robometer_4b" in reward_models else None

    robometer_batch_frames: List[np.ndarray] = []
    robometer_batch_tasks: List[str] = []
    robometer_batch_episode_ids: List[int] = []

    def _flush_robometer_batch() -> None:
        nonlocal robometer_batch_frames, robometer_batch_tasks, robometer_batch_episode_ids
        if robometer_name is None or not robometer_batch_episode_ids:
            robometer_batch_frames = []
            robometer_batch_tasks = []
            robometer_batch_episode_ids = []
            return
        model = reward_models[robometer_name]
        dense_list = model.compute_rewards_batch(
            frames_list=robometer_batch_frames,
            tasks=robometer_batch_tasks,
            batch_size=args.robometer_batch_size,
        )
        # Store back into per-episode buffers by updating the lowdim.npz on disk.
        for ep_id, frames, dense in zip(
            robometer_batch_episode_ids, robometer_batch_frames, dense_list
        ):
            ep_dir = episodes_dir / f"{int(ep_id):06d}"
            lowdim_path = ep_dir / "lowdim.npz"
            arrays = load_npz_arrays(lowdim_path)
            # Episode length is defined by proprio/actions length.
            T = int(arrays["proprio"].shape[0])
            dense = pad_or_trunc_1d(np.asarray(dense, dtype=np.float32), T)

            arrays[f"dense_rewards_{robometer_name}"] = dense.astype(np.float32)
            if overwrite == robometer_name:
                arrays["rewards"] = dense.astype(np.float32)
            save_npz_arrays(lowdim_path, arrays)
        robometer_batch_frames = []
        robometer_batch_tasks = []
        robometer_batch_episode_ids = []

    for row in tqdm(episode_rows, desc="Computing dense rewards", unit="ep", total=n_episodes):
        ep_id = int(row["episode_id"])
        task = str(row.get("task_description", "") or "")
        ep_dir = episodes_dir / f"{ep_id:06d}"
        lowdim_path = ep_dir / "lowdim.npz"
        arrays = load_npz_arrays(lowdim_path)
        T = int(arrays["proprio"].shape[0])
        frames_arr = _read_episode_frames(ep_dir)
        if frames_arr.shape[0] == 0:
            # Still write zeros for missing view to keep dataset consistent.
            if "robodopamine_8b" in reward_models:
                arrays["dense_rewards_robodopamine_8b"] = np.zeros((T,), dtype=np.float32)
                if overwrite == "robodopamine_8b":
                    arrays["rewards"] = np.zeros((T,), dtype=np.float32)
            if "robometer_4b" in reward_models:
                arrays["dense_rewards_robometer_4b"] = np.zeros((T,), dtype=np.float32)
                if overwrite == "robometer_4b":
                    arrays["rewards"] = np.zeros((T,), dtype=np.float32)
            save_npz_arrays(lowdim_path, arrays)
            continue

        # Robodopamine (per-trajectory)
        if "robodopamine_8b" in reward_models:
            dense = reward_models["robodopamine_8b"].compute_rewards_one(
                frames=frames_arr, task=task
            )
            dense = pad_or_trunc_1d(dense, T)
            arrays["dense_rewards_robodopamine_8b"] = dense.astype(np.float32)
            if overwrite == "robodopamine_8b":
                arrays["rewards"] = dense.astype(np.float32)

        # Robometer (batched across trajectories)
        if "robometer_4b" in reward_models:
            robometer_batch_frames.append(frames_arr)
            robometer_batch_tasks.append(task)
            robometer_batch_episode_ids.append(ep_id)
            if len(robometer_batch_episode_ids) >= args.robometer_batch_size:
                _flush_robometer_batch()

        if "robodopamine_8b" in reward_models:
            save_npz_arrays(lowdim_path, arrays)

    # Final robometer flush
    _flush_robometer_batch()
    print("Dense reward computation complete.", flush=True)


if __name__ == "__main__":
    main()
