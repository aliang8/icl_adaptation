#!/usr/bin/env python3
"""
Build sample_index.parquet from an existing LIBERO dataset (manifest.parquet + episodes/).

Reads manifest.parquet (episode_id, task_description, success, n_steps) and writes
sample_index.parquet for in-context training: query_episode_id, query_start, query_len,
prompt_episode_ids, prompt_starts, prompt_lens, task_id, is_success, prompt_len.

Run after convert_libero_hdf5_to_dataset.py. You can re-run with different
--horizon / --num-context / --max-prompt-steps without re-converting the dataset.

Usage:
  python scripts/build_libero_sample_index.py --data-dir datasets
  python scripts/build_libero_sample_index.py --data-dir datasets --horizon 32 --num-context 3 --max-prompt-steps 85
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
        description="Build sample_index.parquet from LIBERO manifest + episodes"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Data root containing LIBERO-Cosmos-Policy/ (manifest.parquet + episodes/)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=32,
        help="Query segment length (default: 32)",
    )
    parser.add_argument(
        "--num-context",
        type=int,
        default=3,
        help="Number of prompt trajectories per sample (default: 3)",
    )
    parser.add_argument(
        "--max-prompt-steps",
        type=int,
        default=None,
        metavar="N",
        help="Max steps per prompt trajectory (last N steps). Default: use all steps in the trajectory.",
    )
    parser.add_argument(
        "--success-mix",
        action="store_true",
        help="Allow mixing success/failure in prompt set (default: same-task only)",
    )
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        sys.exit("Install pandas: pip install pandas (for Parquet)")

    root = Path(args.data_dir).resolve()
    if (root / "LIBERO-Cosmos-Policy").is_dir():
        root = root / "LIBERO-Cosmos-Policy"
    manifest_path = root / "manifest.parquet"
    if not manifest_path.is_file():
        print(f"Manifest not found: {manifest_path}", flush=True)
        print("Run convert_libero_hdf5_to_dataset.py first.", flush=True)
        sys.exit(1)

    df = pd.read_parquet(manifest_path)
    if "episode_id" not in df.columns or "n_steps" not in df.columns:
        print("Manifest must have episode_id and n_steps.", flush=True)
        sys.exit(1)
    task_col = (
        df["task_description"] if "task_description" in df.columns else pd.Series([""] * len(df))
    )
    success_col = df["success"] if "success" in df.columns else pd.Series([False] * len(df))

    meta = []
    for i in range(len(df)):
        meta.append(
            {
                "episode_id": int(df["episode_id"].iloc[i]),
                "task_description": str(task_col.iloc[i]) if pd.notna(task_col.iloc[i]) else "",
                "success": bool(success_col.iloc[i]) if pd.notna(success_col.iloc[i]) else False,
                "n_steps": int(df["n_steps"].iloc[i]),
            }
        )

    from collections import defaultdict

    ep_id_to_meta = {m["episode_id"]: m for m in meta}
    task_to_eps: dict[str, list[tuple[int, bool]]] = defaultdict(list)
    for m in meta:
        task_to_eps[m["task_description"]].append((m["episode_id"], m["success"]))
    task_order = list(dict.fromkeys(m["task_description"] for m in meta))
    task_to_id = {t: i for i, t in enumerate(task_order)}

    horizon = max(1, args.horizon)
    num_context = max(0, args.num_context)
    max_prompt_steps = args.max_prompt_steps  # None = use all steps in each prompt trajectory

    sample_rows = []
    for m in meta:
        ep_id = m["episode_id"]
        task = m["task_description"]
        n_steps = m["n_steps"]
        is_success = 1 if m["success"] else 0
        task_id = task_to_id.get(task, 0)
        if n_steps < horizon:
            continue
        candidates = [(eid, succ) for eid, succ in task_to_eps[task] if eid != ep_id]
        if not candidates and num_context > 0:
            continue
        candidates = sorted(candidates, key=lambda x: (x[1], x[0]))
        for query_start in range(0, n_steps - horizon + 1):
            chosen = candidates[:num_context] if len(candidates) >= num_context else candidates
            if not chosen:
                prompt_episode_ids = []
                prompt_starts = []
                prompt_lens = []
            else:
                prompt_episode_ids = [eid for eid, _ in chosen]
                prompt_starts = []
                prompt_lens = []
                for eid in prompt_episode_ids:
                    N = ep_id_to_meta[eid]["n_steps"]
                    cap = N if max_prompt_steps is None else min(N, max_prompt_steps)
                    start = max(0, N - cap)
                    prompt_starts.append(start)
                    prompt_lens.append(min(cap, N - start))
            prompt_len_total = sum(prompt_lens) if prompt_lens else 0
            sample_rows.append(
                {
                    "query_episode_id": ep_id,
                    "query_start": query_start,
                    "query_len": horizon,
                    "prompt_episode_ids": prompt_episode_ids,
                    "prompt_starts": prompt_starts,
                    "prompt_lens": prompt_lens,
                    "task_id": task_id,
                    "is_success": is_success,
                    "prompt_len": prompt_len_total,
                }
            )

    sample_index_df = pd.DataFrame(sample_rows)
    out_path = root / "sample_index.parquet"
    sample_index_df.to_parquet(out_path, index=False)
    print(f"Saved sample_index to {out_path} ({len(sample_index_df)} rows)", flush=True)


if __name__ == "__main__":
    main()
