#!/usr/bin/env python3
"""
Download LIBERO-Cosmos-Policy from HuggingFace and prepare a local cache for training/eval.

Dataset: https://huggingface.co/datasets/nvidia/LIBERO-Cosmos-Policy
- Parquet on HF: rows are timesteps; columns include actions, primary_images_jpeg,
  wrist_images_jpeg, proprio; attributes per episode: task_description, success.
- We download and save under <output_dir>/LIBERO-Cosmos-Policy/ with a manifest
  for train/val splits and episode boundaries.

Alternative (Cosmos Policy): pre-download with
  hf download nvidia/LIBERO-Cosmos-Policy --repo-type dataset --local-dir LIBERO-Cosmos-Policy
See https://github.com/NVlabs/cosmos-policy/blob/main/LIBERO.md

Usage:
  python scripts/download_libero_cosmos.py --output-dir datasets
  python scripts/download_libero_cosmos.py --output-dir datasets --split-fraction 0.9 --seed 42
  python scripts/download_libero_cosmos.py --output-dir datasets --verify  # check columns/episodes only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow importing src when running this script directly (e.g. uv run python scripts/...)
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if _project_root not in sys.path:
    sys.path.insert(0, str(_project_root))


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
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Load dataset, print columns and episode summary, then exit (no manifest/data written).",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install datasets: pip install datasets (or uv sync --extra icrt)")

    out_root = Path(args.output_dir).resolve() / "LIBERO-Cosmos-Policy"
    out_root.mkdir(parents=True, exist_ok=True)

    log = lambda msg, *a: print(msg % a if a else msg)

    # Prefer HDF5 from "hf download" (one file per episode; has task_description/success in file)
    all_episodes_dir = out_root / "all_episodes"
    hdf5_files = sorted(all_episodes_dir.glob("*.hdf5")) if all_episodes_dir.is_dir() else []
    if hdf5_files and not args.streaming:
        log("Building manifest from %d HDF5 files (all_episodes/)", len(hdf5_files))
        from src.data.libero_dataset import _parse_all_episodes_hdf5_filename

        import random as _rnd
        rng = _rnd.Random(args.seed)
        file_episodes = []
        for path in hdf5_files:
            rel = path.relative_to(out_root).as_posix()
            parsed = _parse_all_episodes_hdf5_filename(path.name)
            file_episodes.append({
                "file": rel,
                "task_description": None,
                "success": parsed["success"],
                "suite": parsed["suite"],
            })
        by_suite = {}
        for ep in file_episodes:
            by_suite.setdefault(ep["suite"], []).append(ep)
        train_episodes = []
        val_episodes = []
        for suite, eps in by_suite.items():
            rng.shuffle(eps)
            n = len(eps)
            n_train = max(1, int(n * args.split_fraction))
            train_episodes.extend(eps[:n_train])
            val_episodes.extend(eps[n_train:])
        rng.shuffle(train_episodes)
        rng.shuffle(val_episodes)
        manifest = {
            "repo_id": args.repo_id,
            "data_source": "hdf5",
            "train_episodes": train_episodes,
            "val_episodes": val_episodes,
            "split_fraction": args.split_fraction,
            "seed": args.seed,
        }
        manifest_path = out_root / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        log(
            "Wrote %s (%d train, %d val episodes from HDF5)",
            manifest_path,
            len(train_episodes),
            len(val_episodes),
        )
        log("Done. Use data_dir=%s for training (loader will read HDF5 per episode).", out_root.parent)
        return

    # Prefer local parquet from "hf download --local-dir LIBERO-Cosmos-Policy"
    parquet_train = sorted(out_root.glob("train*.parquet")) or sorted(out_root.glob("*.parquet"))
    if parquet_train and not args.streaming:
        log("Loading from local parquet (hf download layout): %s", [p.name for p in parquet_train[:3]])
        if len(parquet_train) > 3:
            log("  ... and %d more file(s)", len(parquet_train) - 3)
        data_files = [str(p) for p in parquet_train]
        ds = load_dataset(
            "parquet",
            data_files=data_files,
            split="train",
            trust_remote_code=True,
        )
    else:
        log("Loading dataset %s (train split)...", args.repo_id)
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
        # Non-streaming: full dataset in memory / cache (Parquet ~643k rows per HF)
        log("Dataset size: %d rows (columns: %s)", len(ds), ds.column_names)
        # Check for episode column (Parquet may not have it; use task_description then)
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
                if eps[i] != eps[i - 1]:
                    end = i
                    task = (
                        task_col[start]
                        if hasattr(task_col, "__getitem__")
                        else (task_col[start] if start < len(task_col) else None)
                    )
                    succ = success_col[start] if start < len(success_col) else success_col[start]
                    episode_boundaries.append((start, end, task, succ, "unknown"))
                    start = i
            # Last episode: [start, len(eps))
            if start < len(eps):
                t = task_col[start] if start < len(task_col) else None
                succ = success_col[start] if start < len(success_col) else None

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

                episode_boundaries.append(
                    (start, len(eps), t, succ, _infer_suite(t))
                )
        else:
            # Infer boundaries by task_description change (Parquet has no episode_index; ~643k rows)
            if "task_description" not in ds.column_names:
                log(
                    "WARNING: no 'task_description' or episode column; treating full dataset as one episode. "
                    "Expected Parquet columns: actions, proprio, primary_images_jpeg, wrist_images_jpeg, task_description, success"
                )
                episode_boundaries = [(0, len(ds), None, None, "unknown")]
            else:
                log("Using task_description to infer episode boundaries (columns: %s)", ds.column_names)
                task_col = ds["task_description"]
                success_col = ds["success"] if "success" in ds.column_names else [None] * len(ds)
                episode_boundaries = []

                def _norm_task(t):
                    if t is None:
                        return None
                    if isinstance(t, (list, tuple)):
                        t = t[0] if t else None
                    return str(t) if t is not None else None

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
                    t_prev = _norm_task(task_col[i - 1] if hasattr(task_col, "__getitem__") else None)
                    t_cur = _norm_task(task_col[i] if hasattr(task_col, "__getitem__") else None)
                    if t_prev != t_cur:
                        task = task_col[start] if hasattr(task_col, "__getitem__") else None
                        succ = success_col[start] if start < len(success_col) else None
                        if isinstance(succ, (list, tuple)):
                            succ = succ[0] if succ else None
                        episode_boundaries.append((start, i, _norm_task(task), succ, _infer_suite(task)))
                        start = i
                if start < len(ds):
                    task = task_col[start] if hasattr(task_col, "__getitem__") else None
                    succ = success_col[start] if start < len(success_col) else None
                    if isinstance(succ, (list, tuple)):
                        succ = succ[0] if succ else None
                    episode_boundaries.append(
                        (start, len(ds), _norm_task(task), succ, _infer_suite(task))
                    )

    if args.verify:
        total_rows = episode_boundaries[-1][1] if episode_boundaries else 0
        log("VERIFY: %d episodes, %d total rows", len(episode_boundaries), total_rows)
        for i, (s, e, task, succ, suite) in enumerate(episode_boundaries[:5]):
            log("  episode %d: [%d, %d) len=%d task=%s success=%s suite=%s", i, s, e, e - s, repr(task)[:50], succ, suite)
        if len(episode_boundaries) > 5:
            log("  ... and %d more episodes", len(episode_boundaries) - 5)
        log("Verify done. Re-run without --verify to write manifest and save data.")
        return

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
