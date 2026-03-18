#!/usr/bin/env python3
"""
Convert LIBERO-Cosmos-Policy HDF5 → episode folders + Parquet manifest (dataset only).

Output layout:
  - episodes/{episode_id:06d}/: primary.mp4, wrist.mp4, lowdim.npz (proprio, actions, dones, rewards)
  - manifest.parquet: episode_id, task_description, success, n_steps, paths

No sample index here. After converting, run scripts/build_libero_sample_index.py to build
sample_index.parquet for in-context training.

Usage:
  python scripts/convert_libero_hdf5_to_dataset.py --input-dir datasets/LIBERO-Cosmos-Policy
  python scripts/convert_libero_hdf5_to_dataset.py --input-dir ... --output-dir datasets
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if _project_root not in sys.path:
    sys.path.insert(0, str(_project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Convert LIBERO HDF5 to episode folders (MP4+NPZ) + Parquet manifest"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Root containing all_episodes/*.hdf5 (e.g. .../LIBERO-Cosmos-Policy)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write (default: same as input-dir)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="FPS for MP4 encoding (default: 10)",
    )
    args = parser.parse_args()

    try:
        import h5py
        import numpy as np
    except ImportError as e:
        sys.exit(f"Install dependencies: pip install h5py numpy. {e}")

    try:
        import pandas as pd
    except ImportError:
        sys.exit("Install pandas: pip install pandas (for Parquet)")

    try:
        import imageio
    except ImportError:
        imageio = None

    from tqdm import tqdm

    def _frame_to_rgb(frame):
        """Convert HDF5 frame (bytes or (H,W,C)) to (H,W,3) uint8."""
        frame = np.asarray(frame)
        if frame.dtype == np.uint8 and frame.ndim == 3:
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]
            return frame
        if frame.dtype == np.uint8 and frame.ndim == 1:
            import io
            from PIL import Image

            img = Image.open(io.BytesIO(frame.tobytes()))
            return np.array(img.convert("RGB"), dtype=np.uint8)
        return np.asarray(frame, dtype=np.uint8)

    input_root = Path(args.input_dir).resolve()
    output_root = (
        Path(args.output_dir).resolve() if args.output_dir else input_root
    ) / "LIBERO-Cosmos-Policy"
    episodes_dir = output_root / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    all_episodes_dir = input_root / "all_episodes"
    if not all_episodes_dir.is_dir():
        all_episodes_dir = input_root
    hdf5_files = sorted(all_episodes_dir.glob("*.hdf5"))
    if not hdf5_files:
        print(f"No .hdf5 files in {all_episodes_dir}", flush=True)
        sys.exit(1)

    # First pass: read episode metadata for manifest
    print("Reading episode metadata...", flush=True)
    meta = []
    for ep_id, path in enumerate(tqdm(hdf5_files, desc="Metadata", unit="file")):
        with h5py.File(path, "r") as f:
            T = f["proprio"].shape[0]
            task_desc = None
            if "task_description" in f.attrs:
                td = f.attrs["task_description"]
                task_desc = td.decode("utf-8") if hasattr(td, "decode") else td
            succ = bool(f.attrs.get("success", False))
            has_primary = "primary_images_jpeg" in f
            has_wrist = "wrist_images_jpeg" in f
        meta.append(
            {
                "episode_id": ep_id,
                "task_description": task_desc or "",
                "success": succ,
                "n_steps": T,
                "has_primary": has_primary,
                "has_wrist": has_wrist,
            }
        )

    # Second pass: write episode folders (MP4 + NPZ) and collect manifest rows; skip if already exist
    manifest_rows = []
    for ep_id, path in enumerate(tqdm(hdf5_files[:200], desc="Converting episodes", unit="file")):
        ep_dir = episodes_dir / f"{ep_id:06d}"
        lowdim_path = ep_dir / "lowdim.npz"
        skip = lowdim_path.is_file()

        rel_primary = f"episodes/{ep_id:06d}/primary.mp4"
        rel_wrist = f"episodes/{ep_id:06d}/wrist.mp4"
        rel_lowdim = f"episodes/{ep_id:06d}/lowdim.npz"

        if skip:
            # Use existing paths if files exist
            if not (ep_dir / "primary.mp4").is_file():
                rel_primary = None
            if not (ep_dir / "wrist.mp4").is_file():
                rel_wrist = None
        else:
            ep_dir.mkdir(parents=True, exist_ok=True)
            with h5py.File(path, "r") as f:
                proprio = np.asarray(f["proprio"], dtype=np.float32)
                actions = np.asarray(f["actions"], dtype=np.float32)
                T = proprio.shape[0]
                if actions.shape[0] != T:
                    T = min(proprio.shape[0], actions.shape[0])
                    proprio = proprio[:T]
                    actions = actions[:T]
                if "dones" in f:
                    dones = np.asarray(f["dones"], dtype=np.float32).ravel()[:T]
                else:
                    dones = np.zeros(T, dtype=np.float32)
                    if T > 0:
                        dones[-1] = 1.0
                success = bool(f.attrs.get("success", False))
                rewards = np.zeros(T, dtype=np.float32)
                if success and np.any(dones > 0):
                    rewards[np.argmax(dones > 0)] = 1.0

                np.savez_compressed(
                    lowdim_path,
                    proprio=proprio,
                    actions=actions,
                    dones=dones,
                    rewards=rewards,
                )

                primary_path = ep_dir / "primary.mp4"
                wrist_path = ep_dir / "wrist.mp4"
                if "primary_images_jpeg" in f and imageio is not None:
                    frames = []
                    raw = f["primary_images_jpeg"]
                    for t in range(T):
                        frames.append(_frame_to_rgb(raw[t]))
                    if frames:
                        writer = imageio.get_writer(str(primary_path), "mp4", fps=args.fps)
                        for fr in frames:
                            writer.append_data(fr)
                        writer.close()
                else:
                    rel_primary = None

                if "wrist_images_jpeg" in f and imageio is not None:
                    frames = []
                    raw = f["wrist_images_jpeg"]
                    for t in range(T):
                        frames.append(_frame_to_rgb(raw[t]))
                    if frames:
                        writer = imageio.get_writer(str(wrist_path), "mp4", fps=args.fps)
                        for fr in frames:
                            writer.append_data(fr)
                        writer.close()
                else:
                    rel_wrist = None

        m = meta[ep_id]
        manifest_rows.append(
            {
                "episode_id": ep_id,
                "task_description": m["task_description"],
                "success": m["success"],
                "n_steps": m["n_steps"],
                "primary_path": rel_primary,
                "wrist_path": rel_wrist,
                "lowdim_path": rel_lowdim,
            }
        )

    manifest_df = pd.DataFrame(manifest_rows)

    manifest_path = output_root / "manifest.parquet"
    manifest_df.to_parquet(manifest_path, index=False)

    print(f"Saved manifest to {manifest_path} ({len(manifest_df)} episodes)", flush=True)
    print(f"Episodes under {episodes_dir}", flush=True)
    print(
        "Next: python scripts/build_libero_sample_index.py --data-dir " + str(output_root.parent),
        flush=True,
    )
    if imageio is None:
        print(
            "Note: imageio not installed; images not written to MP4. pip install imageio imageio-ffmpeg",
            flush=True,
        )


if __name__ == "__main__":
    main()
