"""
LIBERO-Cosmos-Policy dataset: load from HuggingFace (or local disk) and produce
trajectories in the same format as ICLTrajectoryDataset (observations=proprio, actions, rewards, terminals).

Expects:
- Either a manifest at data_dir/LIBERO-Cosmos-Policy/manifest.json with train_episodes (start, end, task_description, success, suite),
  and data from load_dataset(repo_id) or from data_dir/LIBERO-Cosmos-Policy/data (saved with save_to_disk).
- Or repo_id only: we load from HF and use full train split as training (no manifest).

Returns: (trajectories, prompt_trajectories_per_task, task_instructions)
- trajectories: list of dicts with observations (T, 9), actions (T, 7), rewards (T), terminals (T)
- prompt_trajectories_per_task: list of list of trajectory dicts (for in-context prompt)
- task_instructions: list of unique task description strings
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def load_libero_manifest(manifest_path: str) -> Dict[str, Any]:
    """Load manifest.json (train_episodes, val_episodes, repo_id, etc.)."""
    with open(manifest_path) as f:
        return json.load(f)


def _get_ds_from_cache_or_hf(
    data_dir: Optional[str],
    repo_id: str,
    split: str = "train",
):
    """Load Dataset from local data_dir/LIBERO-Cosmos-Policy/data or from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")

    root = Path(data_dir or ".").resolve() / "LIBERO-Cosmos-Policy"
    local_data = root / "data"
    if local_data.is_dir():
        from datasets import load_from_disk
        return load_from_disk(str(local_data))
    return load_dataset(repo_id, split=split, trust_remote_code=True)


def _row_to_proprio(row: Dict[str, Any]) -> np.ndarray:
    """Extract proprio (9,) from a row. Column name is 'proprio'."""
    p = row.get("proprio")
    if p is None:
        return np.zeros(9, dtype=np.float32)
    if isinstance(p, (list, tuple)):
        return np.array(p, dtype=np.float32)
    return np.asarray(p, dtype=np.float32)


def _row_to_actions(row: Dict[str, Any]) -> np.ndarray:
    """Extract actions (7,) from a row."""
    a = row.get("actions")
    if a is None:
        return np.zeros(7, dtype=np.float32)
    if isinstance(a, (list, tuple)):
        return np.array(a, dtype=np.float32)
    return np.asarray(a, dtype=np.float32)


def _episode_to_trajectory(
    ds,
    start: int,
    end: int,
    task_description: Optional[str],
    success: Optional[bool],
) -> Dict[str, np.ndarray]:
    """Build one trajectory dict from dataset rows [start, end)."""
    T = end - start
    observations = np.zeros((T, 9), dtype=np.float32)
    actions = np.zeros((T, 7), dtype=np.float32)
    rewards = np.zeros(T, dtype=np.float32)
    terminals = np.zeros(T, dtype=np.float32)
    for i in range(T):
        row = ds[start + i]
        observations[i] = _row_to_proprio(row)
        actions[i] = _row_to_actions(row)
        rewards[i] = 0.0
        terminals[i] = 0.0
    if success and T > 0:
        rewards[-1] = 1.0
        terminals[-1] = 1.0
    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "task_description": task_description,
        "success": success,
    }


def load_libero_trajectories(
    data_dir: str,
    manifest_path: Optional[str] = None,
    repo_id: str = "nvidia/LIBERO-Cosmos-Policy",
    max_episodes: Optional[int] = None,
    success_only: bool = False,
) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, Any]]], List[str]]:
    """
    Load LIBERO-Cosmos-Policy into (trajectories, prompt_trajectories_per_task, task_instructions).

    - If manifest_path is set, use train_episodes from manifest and optional local/HF dataset.
    - Otherwise load full train split and build one trajectory per episode if episode_id exists, else one big trajectory (not ideal).
    - success_only: if True, only include episodes with success=True.
    """
    if manifest_path is None:
        manifest_path = str(Path(data_dir) / "LIBERO-Cosmos-Policy" / "manifest.json")
    manifest = None
    if Path(manifest_path).is_file():
        manifest = load_libero_manifest(manifest_path)
        train_eps = manifest.get("train_episodes", [])
        repo_id = manifest.get("repo_id", repo_id)
    else:
        train_eps = []

    ds = _get_ds_from_cache_or_hf(data_dir, repo_id, split="train")
    trajectories = []
    task_to_trajs: Dict[str, List[Dict]] = {}
    task_order: List[str] = []

    if train_eps:
        for ep in train_eps:
            if success_only and ep.get("success") is False:
                continue
            if max_episodes and len(trajectories) >= max_episodes:
                break
            start, end = ep["start"], ep["end"]
            if end <= start:
                continue
            traj = _episode_to_trajectory(
                ds, start, end,
                ep.get("task_description"),
                ep.get("success"),
            )
            trajectories.append(traj)
            td = traj.get("task_description") or ""
            td = str(td).strip()
            if td and td not in task_to_trajs:
                task_order.append(td)
                task_to_trajs[td] = []
            if td:
                task_to_trajs[td].append(traj)
    else:
        # No manifest: try to use episode_index / episode_id if present
        cols = ds.column_names
        if "episode_index" in cols or "episode_id" in cols:
            ep_col = "episode_index" if "episode_index" in cols else "episode_id"
            eps = ds[ep_col]
            task_col = ds.get("task_description", [None] * len(ds))
            success_col = ds.get("success", [None] * len(ds))
            start = 0
            for i in range(1, len(ds) + 1):
                if i == len(ds) or (i < len(eps) and eps[i] != eps[i - 1]):
                    end = i
                    if max_episodes and len(trajectories) >= max_episodes:
                        break
                    task = task_col[start] if start < len(task_col) else None
                    succ = success_col[start] if start < len(success_col) else None
                    if success_only and succ is False:
                        start = i
                        continue
                    traj = _episode_to_trajectory(ds, start, end, task, succ)
                    trajectories.append(traj)
                    td = str(task or "").strip()
                    if td and td not in task_to_trajs:
                        task_order.append(td)
                        task_to_trajs[td] = []
                    if td:
                        task_to_trajs[td].append(traj)
                    start = i
        else:
            # Fallback: treat full dataset as one trajectory (not ideal)
            T = len(ds)
            if T > 0 and (max_episodes is None or max_episodes >= 1):
                traj = _episode_to_trajectory(ds, 0, T, None, None)
                trajectories.append(traj)
                task_order.append("")
                task_to_trajs[""] = [traj]

    task_instructions = task_order if task_order else list(dict.fromkeys(str(t.get("task_description") or "") for t in trajectories))
    prompt_per_task = [task_to_trajs.get(ti, [])[:5] for ti in task_instructions]
    if not any(prompt_per_task):
        prompt_per_task = [trajectories[:5]] * max(1, len(task_instructions))

    return trajectories, prompt_per_task, task_instructions
