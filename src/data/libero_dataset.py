"""
LIBERO-Cosmos-Policy dataset: load from HuggingFace (or local disk) and produce
trajectories in the same format as ICLTrajectoryDataset (observations=proprio, actions, rewards, terminals).

Data sources (see https://huggingface.co/datasets/nvidia/LIBERO-Cosmos-Policy):
- HDF5 (recommended): after "hf download ... --local-dir LIBERO-Cosmos-Policy", the repo has
  all_episodes/*.hdf5 (one file per episode). Each file has proprio (T,9), actions (T,7) and
  attributes task_description, success. Manifest lists episodes by "file" path.
- Parquet/Hub: load_dataset() or auto-converted Parquet have no per-episode metadata, so
  episode boundaries are lost unless the manifest was built from HDF5.

Expects:
- Manifest at data_dir/LIBERO-Cosmos-Policy/manifest.json. If train_episodes have "file"
  (and optionally "data_source": "hdf5"), load from HDF5. Else use start/end with a single table.
- Or no manifest: load from HF and infer episodes if possible.

Returns: (trajectories, prompt_trajectories_per_task, task_instructions)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger as log


def load_libero_manifest(manifest_path: str) -> Dict[str, Any]:
    """Load manifest.json (train_episodes, val_episodes, repo_id, etc.)."""
    with open(manifest_path) as f:
        return json.load(f)


def _parse_all_episodes_hdf5_filename(filename: str) -> Dict[str, Any]:
    """Parse episode_data--suite=X--...--task=T--ep=E--success=True|False--regen_demo.hdf5."""
    out = {"suite": "unknown", "task_id": None, "success": None}
    i = filename.find("--suite=")
    if i != -1:
        i += 7
        j = filename.find("--", i)
        if j == -1:
            j = len(filename)
        out["suite"] = filename[i:j].strip()
    i = filename.find("--success=")
    if i != -1:
        i += 10
        j = filename.find("--", i)
        if j == -1:
            j = len(filename)
        out["success"] = filename[i:j].strip().lower() == "true"
    return out


def _load_trajectory_from_hdf5(
    file_path: Path,
    task_description: Optional[str] = None,
    success: Optional[bool] = None,
) -> Dict[str, np.ndarray]:
    """Load one trajectory from an all_episodes HDF5 file (proprio, actions, rewards, terminals)."""
    import h5py
    with h5py.File(file_path, "r") as f:
        proprio = np.asarray(f["proprio"], dtype=np.float32)
        actions = np.asarray(f["actions"], dtype=np.float32)
        if task_description is None and "task_description" in f.attrs:
            task_description = f.attrs["task_description"]
            if hasattr(task_description, "decode"):
                task_description = task_description.decode("utf-8")
        if success is None and "success" in f.attrs:
            success = bool(f.attrs["success"])
    T = proprio.shape[0]
    if actions.shape[0] != T:
        T = min(proprio.shape[0], actions.shape[0])
        proprio = proprio[:T]
        actions = actions[:T]
    rewards = np.zeros(T, dtype=np.float32)
    terminals = np.zeros(T, dtype=np.float32)
    if success and T > 0:
        rewards[-1] = 1.0
        terminals[-1] = 1.0
    return {
        "observations": proprio,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "task_description": task_description,
        "success": success,
    }


def _get_ds_from_cache_or_hf(
    data_dir: Optional[str],
    repo_id: str,
    split: str = "train",
):
    """
    Load Dataset from (in order):
    1. data_dir/LIBERO-Cosmos-Policy/data (our save_to_disk format)
    2. data_dir/LIBERO-Cosmos-Policy/*.parquet (hf download --local-dir layout)
    3. HuggingFace hub (or cache)
    """
    from datasets import load_dataset

    root = Path(data_dir or ".").resolve() / "LIBERO-Cosmos-Policy"
    # 1) Our script's save_to_disk format
    local_data = root / "data"
    if local_data.is_dir():
        from datasets import load_from_disk

        log.debug("Loading LIBERO-Cosmos from local data/ (save_to_disk)")
        return load_from_disk(str(local_data))

    # 2) hf download --local-dir layout: parquet file(s) in repo root
    parquet_train = sorted(root.glob("train*.parquet")) or sorted(root.glob("*.parquet"))
    if parquet_train:
        data_files = [str(p) for p in parquet_train]
        log.info("Loading LIBERO-Cosmos from local parquet (hf download): %s", data_files[:3])
        if len(data_files) > 3:
            log.info("  ... and %d more file(s)", len(data_files) - 3)
        return load_dataset(
            "parquet",
            data_files=data_files,
            split=split,
            trust_remote_code=True,
        )

    # 3) HuggingFace hub / cache
    return load_dataset(repo_id, split=split, trust_remote_code=True)


def _decode_image_bytes(value: Any) -> np.ndarray:
    """Decode image from bytes, dict with 'bytes', or PIL Image -> (H, W, C) uint8."""
    if value is None:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    if hasattr(value, "size") and hasattr(value, "mode"):
        # PIL Image
        return np.array(value)
    if isinstance(value, dict) and "bytes" in value:
        value = value["bytes"]
    if isinstance(value, (bytes, bytearray)):
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(bytes(value)))
        return np.array(img.convert("RGB"))
    return np.asarray(value, dtype=np.uint8)


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
    image_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build one trajectory dict from dataset rows [start, end). Optionally add 'images' (list of (T,H,W,C) per view)."""
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
    out: Dict[str, Any] = {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "task_description": task_description,
        "success": success,
    }
    if image_keys and hasattr(ds, "column_names"):
        cols = ds.column_names
        images_per_view: List[List[np.ndarray]] = [[] for _ in image_keys]
        for k, key in enumerate(image_keys):
            if key not in cols:
                break
            for i in range(T):
                row = ds[start + i]
                val = row.get(key) if isinstance(row, dict) else getattr(row, key, None)
                img = _decode_image_bytes(val)
                images_per_view[k].append(img)
        if all(len(v) == T for v in images_per_view):
            out["images"] = [np.stack(v, axis=0) for v in images_per_view]
    return out


def load_libero_trajectories(
    data_dir: str,
    manifest_path: Optional[str] = None,
    repo_id: str = "nvidia/LIBERO-Cosmos-Policy",
    max_episodes: Optional[int] = None,
    success_only: bool = False,
    image_keys: Optional[List[str]] = None,
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

    root = Path(data_dir or ".").resolve() / "LIBERO-Cosmos-Policy"
    use_hdf5 = bool(train_eps) and ("file" in train_eps[0] or manifest.get("data_source") == "hdf5")
    trajectories = []
    task_to_trajs: Dict[str, List[Dict]] = {}
    task_order: List[str] = []

    if use_hdf5:
        log.info("Loading LIBERO-Cosmos from HDF5 (all_episodes/*.hdf5)")
        for ep in train_eps:
            if success_only and ep.get("success") is False:
                continue
            if max_episodes and len(trajectories) >= max_episodes:
                break
            file_path = root / ep["file"]
            if not file_path.is_file():
                log.warning("Skipping missing HDF5 file: %s", file_path)
                continue
            traj = _load_trajectory_from_hdf5(
                file_path,
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
    elif train_eps:
        ds = _get_ds_from_cache_or_hf(data_dir, repo_id, split="train")
        for ep in train_eps:
            if success_only and ep.get("success") is False:
                continue
            if max_episodes and len(trajectories) >= max_episodes:
                break
            start, end = ep["start"], ep["end"]
            if end <= start:
                continue
            traj = _episode_to_trajectory(
                ds,
                start,
                end,
                ep.get("task_description"),
                ep.get("success"),
                image_keys=image_keys,
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
        # No manifest or no train_eps: load full dataset and infer episodes
        ds = _get_ds_from_cache_or_hf(data_dir, repo_id, split="train")
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
                    traj = _episode_to_trajectory(ds, start, end, task, succ, image_keys=image_keys)
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
                traj = _episode_to_trajectory(ds, 0, T, None, None, image_keys=image_keys)
                trajectories.append(traj)
                task_order.append("")
                task_to_trajs[""] = [traj]

    task_instructions = (
        task_order
        if task_order
        else list(dict.fromkeys(str(t.get("task_description") or "") for t in trajectories))
    )
    prompt_per_task = [task_to_trajs.get(ti, [])[:5] for ti in task_instructions]
    if not any(prompt_per_task):
        prompt_per_task = [trajectories[:5]] * max(1, len(task_instructions))

    if len(trajectories) <= 1:
        total_steps = sum(t["rewards"].shape[0] for t in trajectories)
        log.warning(
            "LIBERO-Cosmos: only %d trajectory(ies) loaded (total steps=%d). "
            "Re-run: python scripts/download_libero_cosmos.py --output-dir <data_dir> without --streaming to fetch full dataset, then check manifest.json train_episodes.",
            len(trajectories),
            total_steps,
        )

    return trajectories, prompt_per_task, task_instructions
