"""
LIBERO-Cosmos-Policy: episode folders + Parquet manifest + sample index for in-context training.

Layout (from convert_libero_hdf5_to_dataset.py):
  - episodes/{id:06d}/: primary.mp4, wrist.mp4, lowdim.npz
  - manifest.parquet, sample_index.parquet

Training uses the generic in-context index: build_libero_in_context_dataset is registered
as source "libero" so train.py calls build_in_context_dataset("libero", ...).
Eval: load_libero_episodes_for_eval(data_dir, ...) loads from manifest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from src.utils import read_mp4_frames
import torch
import sys
from loguru import logger as log
from tqdm import tqdm

from src.config.schema import resolved_max_total_prompt_length


def _has_new_format(root: Path) -> bool:
    """True if manifest.parquet and episodes/ exist (MP4+NPZ layout)."""
    return (root / "manifest.parquet").is_file() and (root / "episodes").is_dir()


def _load_episode_from_folder(
    root: Path,
    episode_id: int,
    task_description: Optional[str],
    success: bool,
    image_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Load one trajectory from episodes/{id:06d}/ (lowdim.npz + optional MP4)."""
    ep_dir = root / "episodes" / f"{episode_id:06d}"
    lowdim_path = ep_dir / "lowdim.npz"
    if not lowdim_path.is_file():
        raise FileNotFoundError(f"Missing {lowdim_path}")
    data = np.load(lowdim_path, allow_pickle=False)
    proprio = data["proprio"]
    actions = data["actions"]
    dones = data["dones"] if "dones" in data else np.zeros(len(proprio), dtype=np.float32)
    rewards = data["rewards"] if "rewards" in data else np.zeros(len(proprio), dtype=np.float32)
    T = len(proprio)
    if np.any(dones > 0) and rewards.sum() == 0:
        rewards[np.argmax(dones > 0)] = 1.0

    out: Dict[str, Any] = {
        "observations": np.asarray(proprio, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "terminals": np.asarray(dones, dtype=np.float32),
        "task_description": task_description,
        "success": bool(success),
    }
    if image_keys:
        images_per_view = []
        key_to_file = {"primary_images_jpeg": "primary.mp4", "wrist_images_jpeg": "wrist.mp4"}
        for key in image_keys:
            fname = key_to_file.get(key)
            if not fname:
                continue
            vid_path = ep_dir / fname
            if not vid_path.is_file():
                break
            frames = read_mp4_frames(vid_path)
            if len(frames) != T:
                frames = (
                    frames[:T]
                    if len(frames) >= T
                    else list(frames) + [np.zeros_like(frames[0])] * (T - len(frames))
                )
            images_per_view.append(np.stack(frames, axis=0))
        if len(images_per_view) == len(image_keys):
            out["images"] = images_per_view
    return out


def get_libero_sample_index(data_dir: str):
    """
    Return the precomputed sample index (manifest + episodes/ layout).
    Columns: query_episode_id, query_start, query_len, prompt_episode_ids, prompt_starts, prompt_lens,
    task_id, is_success, prompt_len. Returns None if sample_index.parquet is missing.
    """
    root = Path(data_dir or ".").resolve() / "LIBERO-Cosmos-Policy"
    idx_path = root / "sample_index.parquet"
    if not idx_path.is_file():
        return None
    import pandas as pd

    return pd.read_parquet(idx_path)


def get_libero_task_instructions_from_manifest(data_dir: str) -> Optional[List[str]]:
    """When using new format (manifest + sample_index), return task_instructions in task_id order."""
    root = Path(data_dir or ".").resolve() / "LIBERO-Cosmos-Policy"
    manifest_path = root / "manifest.parquet"
    if not manifest_path.is_file():
        return None
    import pandas as pd

    df = pd.read_parquet(manifest_path)
    if "task_description" not in df.columns:
        return None
    return list(dict.fromkeys(df["task_description"].tolist()))


def load_libero_episodes_for_eval(
    data_dir: str,
    last_n_fraction: float = 0.1,
    max_episodes: int = 500,
    image_keys: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Load the last N% of episodes from manifest for offline eval. Requires manifest + episodes."""
    root = Path(data_dir or ".").resolve() / "LIBERO-Cosmos-Policy"
    if not _has_new_format(root):
        return []
    import pandas as pd

    manifest_df = pd.read_parquet(root / "manifest.parquet")
    n_val = max(1, int(len(manifest_df) * last_n_fraction))
    n_val = min(n_val, max_episodes)
    val_rows = manifest_df.tail(n_val)
    trajectories = []
    for _, row in val_rows.iterrows():
        ep_id = int(row["episode_id"])
        task_desc = str(row.get("task_description", "")) or None
        success = bool(row.get("success", False))
        try:
            traj = _load_episode_from_folder(root, ep_id, task_desc, success, image_keys=image_keys)
        except Exception as e:
            log.warning("Skip episode {}: {}", ep_id, e)
            continue
        trajectories.append(traj)
    return trajectories


def _load_episode_embedding_segment(
    root: Path,
    episode_id: int,
    start: int,
    length: int,
) -> Optional[np.ndarray]:
    """Load embedding segment [start:start+length] from episodes/{id}/embeddings.npz. Returns (L, D) or None."""
    ep_dir = root / "episodes" / f"{episode_id:06d}"
    emb_path = ep_dir / "embeddings.npz"
    if not emb_path.is_file():
        return None
    data = np.load(emb_path, allow_pickle=False)
    if "embeddings" not in data:
        return None
    emb = data["embeddings"]
    end = min(start + length, emb.shape[0])
    return np.asarray(emb[start:end], dtype=np.float32)


def _load_episode_segment(
    root: Path,
    episode_id: int,
    start: int,
    length: int,
    image_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Load a segment [start:start+length] from one episode (NPZ + optional MP4)."""
    full = _load_episode_from_folder(root, episode_id, None, False, image_keys=image_keys)
    end = min(start + length, full["observations"].shape[0])
    s = slice(start, end)
    L = end - start
    out = {
        "observations": np.asarray(full["observations"][s], dtype=np.float32),
        "actions": np.asarray(full["actions"][s], dtype=np.float32),
        "rewards": np.asarray(full["rewards"][s], dtype=np.float32),
        "terminals": np.asarray(full["terminals"][s], dtype=np.float32),
    }
    if "images" in full:
        out["images"] = [v[s].copy() for v in full["images"]]
    return out


def make_libero_index_loader(
    root: Path,
    task_instructions: List[str],
    state_dim: int,
    act_dim: int,
    context_dim: int,
    device: torch.device,
    rtg_scale: float,
    total_prompt_len: Optional[int],
    max_prompt_trajectory_length: Optional[int],
    use_vision: bool = False,
    image_keys: Optional[List[str]] = None,
    use_precomputed_embeddings: bool = False,
) -> Callable[[Dict[str, Any]], Tuple[Any, ...]]:
    """
    Build a loader_fn(row) for IndexBackedDataset. Each row has query_episode_id, query_start,
    query_len, prompt_episode_ids, prompt_starts, prompt_lens, task_id. Returns the same tuple
    as ICL dataset __getitem__: query (+ query_trial_idx), prompt (+ prompt_trial_idx), instruction,
    optional images, optional index row (see collate_icl_batch layout).
    """
    from src.data.dataset import _pad_or_trim_prompt
    from src.data.trajectories import discount_cumsum

    state_mean = np.zeros(state_dim, dtype=np.float32)
    state_std = np.ones(state_dim, dtype=np.float32)
    img_keys = list(image_keys) if (use_vision and image_keys) else None

    def loader_fn(row: Dict[str, Any]) -> Tuple[Any, ...]:
        q_ep = int(row["query_episode_id"])
        q_start = int(row["query_start"])
        q_len = int(row["query_len"])
        task_id = int(row.get("task_id", 0))
        instruction = ""
        if 0 <= task_id < len(task_instructions):
            instruction = task_instructions[task_id] or ""

        # Query segment: q_len steps (s,a pairs for training)
        q_seg = _load_episode_segment(root, q_ep, q_start, q_len, image_keys=img_keys)
        obs = q_seg["observations"]
        actions = q_seg["actions"]
        rewards = q_seg["rewards"]
        dones = q_seg["terminals"]
        seg_len = obs.shape[0]
        if seg_len < 1:
            seg_len = 1
            obs = np.zeros((1, state_dim), dtype=np.float32)
            actions = np.zeros((1, act_dim), dtype=np.float32)
            rewards = np.zeros(1, dtype=np.float32)
            dones = np.zeros(1, dtype=np.float32)

        obs_norm = (obs - state_mean) / state_std
        rtg = discount_cumsum(rewards, gamma=1.0).reshape(-1, 1) / rtg_scale
        timesteps = np.arange(q_start, q_start + seg_len, dtype=np.float32)
        mask = np.ones(seg_len, dtype=np.float32)
        context = np.zeros((seg_len, context_dim), dtype=np.float32)

        # Prompt segments from index (Parquet may give list or ndarray)
        def _to_list(x):
            if x is None:
                return []
            if isinstance(x, (str, bytes)):
                import json

                return json.loads(x) if isinstance(x, str) else []
            return list(x)

        prompt_episode_ids = _to_list(row.get("prompt_episode_ids"))
        prompt_starts = _to_list(row.get("prompt_starts"))
        prompt_lens = _to_list(row.get("prompt_lens"))

        segs_ps, segs_pa, segs_pr, segs_prtg, segs_pts, segs_pm, segs_ptrial = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for i, (p_ep, p_start, p_len) in enumerate(
            zip(prompt_episode_ids, prompt_starts, prompt_lens)
        ):
            p_ep, p_start, p_len = int(p_ep), int(p_start), int(p_len)
            cap = max_prompt_trajectory_length if max_prompt_trajectory_length else p_len
            if cap < p_len:
                p_start = p_start + (p_len - cap)
                p_len = cap
            p_seg = _load_episode_segment(root, p_ep, p_start, p_len, image_keys=None)
            T = p_seg["observations"].shape[0]
            if T == 0:
                continue
            ps = (p_seg["observations"] - state_mean) / state_std
            pa = p_seg["actions"]
            pr = p_seg["rewards"].reshape(-1, 1)
            prtg = discount_cumsum(p_seg["rewards"], gamma=1.0)[:T].reshape(-1, 1) / rtg_scale
            pts = np.arange(p_start, p_start + T, dtype=np.float32)
            pm = np.ones(T, dtype=np.float32)
            ptrial = np.full(T, float(i + 1), dtype=np.float32)
            segs_ps.append(ps)
            segs_pa.append(pa)
            segs_pr.append(pr)
            segs_prtg.append(prtg)
            segs_pts.append(pts)
            segs_pm.append(pm)
            segs_ptrial.append(ptrial)
        if segs_ps:
            ps = np.concatenate(segs_ps, axis=0)
            pa = np.concatenate(segs_pa, axis=0)
            pr = np.concatenate(segs_pr, axis=0)
            prtg = np.concatenate(segs_prtg, axis=0)
            pts = np.concatenate(segs_pts, axis=0)
            pm = np.concatenate(segs_pm, axis=0)
            ptrial = np.concatenate(segs_ptrial, axis=0)
            actual_len = ps.shape[0]
            plen = min(actual_len, total_prompt_len) if total_prompt_len is not None else actual_len
            ps, pa, pr, prtg, pts, pm, ptrial = _pad_or_trim_prompt(
                ps, pa, pr, prtg, pts, pm, ptrial, plen, state_dim, act_dim, take_last=True
            )
        else:
            plen = total_prompt_len if total_prompt_len is not None else 256
            ps = np.zeros((plen, state_dim), dtype=np.float32)
            pa = np.ones((plen, act_dim), dtype=np.float32) * -10.0
            pr = np.zeros((plen, 1), dtype=np.float32)
            prtg = np.zeros((plen, 1), dtype=np.float32)
            pts = np.zeros(plen, dtype=np.float32)
            pm = np.zeros(plen, dtype=np.float32)
            ptrial = np.zeros(plen, dtype=np.float32)

        pmv = np.asarray(pm, dtype=np.float32).reshape(-1)
        ptv = np.asarray(ptrial, dtype=np.float32).reshape(-1)
        query_tid = int(np.max(ptv[pmv > 0])) + 1 if np.any(pmv > 0) else 1
        query_trial = np.full(seg_len, float(query_tid), dtype=np.float32)

        def _t(x: np.ndarray, long_type: bool = False) -> torch.Tensor:
            t = torch.from_numpy(np.asarray(x))
            return t.long().to(device) if long_type else t.float().to(device)

        result: Tuple[Any, ...] = (
            _t(obs_norm),
            _t(context),
            _t(actions),
            _t(rewards),
            _t(dones, True),
            _t(rtg),
            _t(timesteps, True),
            _t(mask),
            _t(query_trial, True),
            _t(ps),
            _t(pa),
            _t(pr),
            _t(prtg),
            _t(pts, True),
            _t(pm),
            _t(ptrial, True),
            instruction,
        )
        if use_precomputed_embeddings:
            emb = _load_episode_embedding_segment(root, q_ep, q_start, q_len)
            # Expected embeddings: (T, D). We wrap as (1, T, D) for collate.
            if emb is not None:
                if not isinstance(emb, np.ndarray):
                    emb = np.asarray(emb)
                if emb.ndim != 2:
                    print(
                        f"[precomputed_embeddings] Bad embeddings.ndim for episode={q_ep} "
                        f"query_start={q_start} query_len={q_len}: emb.shape={emb.shape}; "
                        "regenerate embeddings.npz.",
                        file=sys.stderr,
                        flush=True,
                    )
                    result = result + (None,)
                elif emb.shape[0] > 0 and emb.shape[1] > 0:
                    t = torch.from_numpy(emb).float().to(device).unsqueeze(0)  # (1, T, D)
                    result = result + (t,)
                else:
                    result = result + (None,)
            else:
                result = result + (None,)
        elif use_vision and img_keys and "images" in q_seg:
            # Vision encoders (DINOv2, etc.) expect (B, T, C, H, W); frames are (T, H, W, C)
            imgs = []
            for a in q_seg["images"]:
                t = torch.from_numpy(np.asarray(a)).float().to(device)
                if t.dim() == 4 and t.shape[-1] == 3:
                    t = t.permute(0, 3, 1, 2)
                if t.dim() == 4:
                    t = t.unsqueeze(0)
                imgs.append(t)
            result = result + (imgs,)
        else:
            result = result + (None,)
        # Attach index row for debug (sample index fields: query_episode_id, query_start, prompt_episode_ids, etc.)
        index_summary = {
            "query_episode_id": q_ep,
            "query_start": q_start,
            "query_len": q_len,
            "task_id": task_id,
            "prompt_episode_ids": list(prompt_episode_ids) if prompt_episode_ids else [],
            "prompt_starts": list(prompt_starts) if prompt_starts else [],
            "prompt_lens": list(prompt_lens) if prompt_lens else [],
        }
        result = result + (index_summary,)
        return result

    return loader_fn


def build_libero_in_context_dataset(
    data_dir,
    data_cfg,
    device,
    state_dim,
    action_dim,
    collate_fn,
):
    """Build in-context dataset + loader from LIBERO sample index. Returns None if no index."""
    from src.data.sample_index import (
        GroupedBatchSampler,
        IndexBackedDataset,
        InContextDatasetResult,
        SampleIndex,
    )

    root = Path(data_dir).resolve() / "LIBERO-Cosmos-Policy"
    if not _has_new_format(root):
        return None
    index_df = get_libero_sample_index(str(data_dir))
    if index_df is None:
        return None
    task_instructions = get_libero_task_instructions_from_manifest(str(data_dir)) or []
    index = SampleIndex(index_df, length_bin_columns=["query_len", "prompt_len"])
    use_precomputed_embeddings = data_cfg.use_precomputed_embeddings
    total_plen = resolved_max_total_prompt_length(data_cfg)
    if data_cfg.max_total_prompt_length is None:
        log.info(
            "LIBERO: data.max_total_prompt_length unset -> using {} "
            "(max_episode_steps * num_context_trajectories)",
            total_plen,
        )
    loader_fn = make_libero_index_loader(
        root,
        task_instructions,
        state_dim,
        action_dim,
        data_cfg.context_dim,
        device,
        float(data_cfg.rtg_scale),
        total_plen,
        data_cfg.max_prompt_trajectory_length,
        use_vision=data_cfg.use_vision and not use_precomputed_embeddings,
        image_keys=data_cfg.image_keys or [],
        use_precomputed_embeddings=use_precomputed_embeddings,
    )
    idx_dataset = IndexBackedDataset(index, loader_fn)

    class _Wrapper:
        """Wrapper so index-backed dataset matches the interface expected by train.py (_print_dataset_stats, etc.)."""

        def __init__(self, d, ti, tpl, mpt, sd, ad, cfg):
            self._d = d
            self.task_instructions = ti
            self.total_prompt_len = tpl
            self.max_prompt_trajectory_length = mpt
            self.state_mean = np.zeros(sd, dtype=np.float32)
            self.state_std = np.ones(sd, dtype=np.float32)
            self.trajectories = []
            self.state_dim = sd
            self.act_dim = ad
            self.horizon = cfg.horizon
            qh = cfg.query_history_length
            self._query_length = cfg.horizon if qh is None else qh
            self.max_episode_steps = cfg.max_episode_steps
            self.num_context_trajectories = cfg.num_context_trajectories
            self.prompt_length = cfg.prompt_length
            self.context_subsample_strategy = cfg.context_subsample_strategy
            self.return_min = 0.0
            self.return_max = 1.0
            self.return_avg = 0.5

        def __len__(self):
            return len(self._d)

        def __getitem__(self, i):
            return self._d[i]

    dataset = _Wrapper(
        idx_dataset,
        task_instructions,
        total_plen,
        data_cfg.max_prompt_trajectory_length,
        state_dim,
        action_dim,
        data_cfg,
    )
    batch_sampler = GroupedBatchSampler(
        index, data_cfg.batch_size, shuffle=True, seed=data_cfg.seed
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=0,
        collate_fn=collate_fn,
    )
    log.info(
        "LIBERO in-context index: {} rows, grouped batching by query_len/prompt_len",
        len(index),
    )
    return InContextDatasetResult(
        dataset=dataset,
        loader=loader,
        state_mean=dataset.state_mean,
        state_std=dataset.state_std,
        task_instructions=task_instructions,
        total_prompt_len=total_plen,
        max_prompt_trajectory_length=data_cfg.max_prompt_trajectory_length,
    )


from src.data.sample_index import register_in_context_builder

register_in_context_builder("libero", build_libero_in_context_dataset)
