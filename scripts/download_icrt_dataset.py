#!/usr/bin/env python3
"""
Download ICRT-MT multi-task robot dataset from HuggingFace.

Dataset: https://huggingface.co/datasets/Ravenh97/ICRT-MT
Structure (after download):
  <output_dir>/ICRT-MT/
    merged_data_part1.hdf5
    merged_data_part1_keys.json  (or similar)
    epi_len_mapping.json, verb_to_episode.json (if provided by repo)

Usage:
  python scripts/download_icrt_dataset.py --output-dir datasets
  python scripts/download_icrt_dataset.py --output-dir datasets --repo-id Ravenh97/ICRT-MT
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download ICRT-MT dataset from HuggingFace")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets",
        help="Directory to save the dataset (default: datasets)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="Ravenh97/ICRT-MT",
        help="HuggingFace dataset repo (default: Ravenh97/ICRT-MT)",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Repo revision/branch (default: main)",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit(
            "Install huggingface_hub: pip install huggingface_hub\n"
            "For Git LFS (large files): sudo apt install git-lfs && git lfs install"
        )

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    local_dir = out / "ICRT-MT"
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.repo_id} to {local_dir} ...")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        revision=args.revision,
        local_dir_use_symlinks=False,
    )

    # Write a minimal dataset_config.json for our loader (paths relative to dataset root)
    config_path = local_dir / "dataset_config.json"
    hdf5_keys_path = local_dir / "merged_data_part1_keys.json"
    if not hdf5_keys_path.exists():
        # Try alternate names
        for name in ("merged_data_part1_hdf5_keys.json", "hdf5_keys.json"):
            p = local_dir / name
            if p.exists():
                hdf5_keys_path = p
                break

    hdf5_path = local_dir / "merged_data_part1.hdf5"
    config = {
        "dataset_path": [str(hdf5_path)],
        "hdf5_keys": [str(hdf5_keys_path)],
        "epi_len_mapping_json": str(local_dir / "epi_len_mapping.json"),
        "verb_to_episode": str(local_dir / "verb_to_episode.json"),
        "train_split": 0.95,
        "action_keys": ["action/cartesian_position", "action/gripper_position"],
        "image_keys": [
            "observation/exterior_image_1_left",
            "observation/wrist_image_left",
        ],
        "proprio_keys": ["observation/cartesian_position", "observation/gripper_position"],
    }

    if not hdf5_path.exists():
        print(f"Warning: {hdf5_path} not found; dataset may use different file names.")
    if not hdf5_keys_path.exists():
        print(f"Warning: hdf5_keys file not found; you may need to create it or use repo defaults.")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote {config_path}")
    print(f"Done. Dataset at {local_dir}")


if __name__ == "__main__":
    main()
