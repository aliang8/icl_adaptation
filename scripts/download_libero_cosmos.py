#!/usr/bin/env python3
"""
Download LIBERO-Cosmos-Policy from HuggingFace and prepare a local cache for training/eval.

Dataset: https://huggingface.co/datasets/nvidia/LIBERO-Cosmos-Policy
- Parquet on HF: rows are timesteps; columns include actions, primary_images_jpeg,
  wrist_images_jpeg, proprio; attributes per episode: task_description, success.
- We download and save under <output_dir>/LIBERO-Cosmos-Policy/ with a manifest
  for train/val splits and episode boundaries.

Usage:
  python scripts/download_libero_cosmos.py --output-dir datasets
  python scripts/download_libero_cosmos.py --output-dir datasets --split-fraction 0.9 --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download LIBERO-Cosmos-Policy from HuggingFace")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets",
        help="Directory to save the dataset (default: datasets)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="nvidia/LIBERO-Cosmos-Policy",
        help="HuggingFace dataset repo (default: nvidia/LIBERO-Cosmos-Policy)",
    )
    parser.add_argument(
        "--split-fraction",
        type=float,
        default=0.9,
        help="Fraction of episodes per suite for train (rest for eval); default 0.9",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split (default: 42)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming (no full download); still writes manifest from first pass.",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install datasets: pip install datasets (or uv sync --extra icrt)")

    out_root = Path(args.output_dir).resolve() / "LIBERO-Cosmos-Policy"
    out_root.mkdir(parents=True, exist_ok=True)

    log = lambda msg, *a: print(msg % a if a else msg)

    log("Loading dataset %s (train split)...", args.repo_id)
    # Load train split; dataset may be large so we use cache and optionally streaming
    ds = load_dataset(
        args.repo_id,
        split="train",
        trust_remote_code=True,
        streaming=args.streaming,
    )

    if args.streaming:
        # Streaming: we only build manifest by iterating once (no local copy of full data)
        log("Streaming mode: building episode manifest (first pass)...")
        episode_boundaries = []  # (start_idx, end_idx, task_description, success, suite)
        current_start = 0
        idx = 0
        prev_task = None
        task_descriptions = []
        success_flags = []
        suite_names = []

        def infer_suite(row):
            td = (
                (row.get("task_description") or [None])[0]
                if hasattr(row.get("task_description"), "__getitem__")
                else row.get("task_description")
            )
            if td is None:
                return "unknown"
            td = str(td).lower()
            if "spatial" in td or "libero_spatial" in td:
                return "libero_spatial"
            if "object" in td or "libero_object" in td:
                return "libero_object"
            if "goal" in td or "libero_goal" in td:
                return "libero_goal"
            if "long" in td or "libero_10" in td or "10" in td:
                return "libero_10"
            return "unknown"

        for row in ds:
            task = row.get("task_description")
            if isinstance(task, (list, tuple)):
                task = task[0] if task else None
            else:
                task = str(task) if task is not None else None
            # Assume episode boundary when task_description changes (or use episode_id if present)
            if prev_task is not None and task != prev_task:
                episode_boundaries.append((current_start, idx, prev_task, None, infer_suite(row)))
                current_start = idx
            prev_task = task
            idx += 1
            if idx % 50000 == 0:
                log("  Processed %d rows...", idx)
        if prev_task is not None:
            episode_boundaries.append(
                (current_start, idx, prev_task, None, infer_suite({"task_description": prev_task}))
            )
        log("Inferred %d episodes from %d rows (streaming).", len(episode_boundaries), idx)
    else:
        # Non-streaming: full dataset in memory / cache
        log("Dataset size: %d rows", len(ds))
        # Check for episode column
        if "episode_index" in ds.column_names or "episode_id" in ds.column_names:
            ep_col = "episode_index" if "episode_index" in ds.column_names else "episode_id"
            episode_boundaries = []
            eps = ds[ep_col]
            task_col = (
                ds["task_description"]
                if "task_description" in ds.column_names
                else [None] * len(ds)
            )
            success_col = ds["success"] if "success" in ds.column_names else [None] * len(ds)
            start = 0
            for i in range(1, len(eps)):
                if eps[i] != eps[i - 1] or i == len(eps) - 1:
                    end = i if eps[i] != eps[i - 1] else len(eps)
                    task = (
                        task_col[start]
                        if hasattr(task_col, "__getitem__")
                        else (task_col[start] if start < len(task_col) else None)
                    )
                    succ = success_col[start] if start < len(success_col) else success_col[start]
                    episode_boundaries.append((start, end, task, succ, "unknown"))
                    start = i

            def _infer_suite(td):
                if td is None:
                    return "unknown"
                td = str(td).lower()
                if "spatial" in td:
                    return "libero_spatial"
                if "object" in td:
                    return "libero_object"
                if "goal" in td:
                    return "libero_goal"
                if "long" in td or "10" in td:
                    return "libero_10"
                return "unknown"

            if start < len(eps):
                t = task_col[start] if start < len(task_col) else None
                episode_boundaries.append(
                    (
                        start,
                        len(eps),
                        t,
                        success_col[start] if start < len(success_col) else None,
                        _infer_suite(t),
                    )
                )
        else:
            # Infer boundaries by task_description change (same task = same episode; may merge episodes)
            task_col = (
                ds["task_description"]
                if "task_description" in ds.column_names
                else [None] * len(ds)
            )
            success_col = ds["success"] if "success" in ds.column_names else [None] * len(ds)
            episode_boundaries = []

            def _infer_suite(td):
                if td is None:
                    return "unknown"
                td = str(td).lower()
                if "spatial" in td or "libero_spatial" in td:
                    return "libero_spatial"
                if "object" in td or "libero_object" in td:
                    return "libero_object"
                if "goal" in td or "libero_goal" in td:
                    return "libero_goal"
                if "long" in td or "libero_10" in td or "10" in td:
                    return "libero_10"
                return "unknown"

            start = 0
            for i in range(1, len(ds)):
                t_prev = task_col[i - 1]
                t_cur = task_col[i]
                if t_prev != t_cur or i == len(ds) - 1:
                    end = i if t_prev != t_cur else len(ds)
                    task = task_col[start]
                    succ = success_col[start] if start < len(success_col) else None
                    episode_boundaries.append((start, end, task, succ, _infer_suite(task)))
                    start = i
            if start < len(ds):
                episode_boundaries.append(
                    (
                        start,
                        len(ds),
                        task_col[start],
                        success_col[start] if start < len(success_col) else None,
                        _infer_suite(task_col[start]),
                    )
                )

    # Train/val split per suite (in-distribution held-out)
    import random

    rng = random.Random(args.seed)
    by_suite = {}
    for s, e, task, succ, suite in episode_boundaries:
        by_suite.setdefault(suite, []).append(
            {"start": s, "end": e, "task_description": task, "success": succ}
        )
    train_episodes = []
    val_episodes = []
    for suite, eps in by_suite.items():
        rng.shuffle(eps)
        n = len(eps)
        n_train = max(1, int(n * args.split_fraction))
        train_episodes.extend([(e, suite) for e in eps[:n_train]])
        val_episodes.extend([(e, suite) for e in eps[n_train:]])
    rng.shuffle(train_episodes)
    rng.shuffle(val_episodes)

    manifest = {
        "repo_id": args.repo_id,
        "train_episodes": [
            {
                "start": e["start"],
                "end": e["end"],
                "task_description": e["task_description"],
                "success": e["success"],
                "suite": s,
            }
            for e, s in train_episodes
        ],
        "val_episodes": [
            {
                "start": e["start"],
                "end": e["end"],
                "task_description": e["task_description"],
                "success": e["success"],
                "suite": s,
            }
            for e, s in val_episodes
        ],
        "split_fraction": args.split_fraction,
        "seed": args.seed,
    }
    manifest_path = out_root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log(
        "Wrote %s (%d train, %d val episodes)",
        manifest_path,
        len(manifest["train_episodes"]),
        len(manifest["val_episodes"]),
    )

    # Save dataset to disk so we don't rely on HF cache only (optional for non-streaming)
    if not args.streaming and hasattr(ds, "save_to_disk"):
        cache_path = out_root / "data"
        ds.save_to_disk(str(cache_path))
        log("Saved dataset to %s", cache_path)
    else:
        log("Using HuggingFace cache for data; manifest at %s", manifest_path)

    log("Done. Use data_dir=%s and manifest=%s for training and eval.", out_root, manifest_path)


if __name__ == "__main__":
    main()
