"""Algorithm-distillation dataset over on-disk ICL replay buffers (lazy window reads).

Reads one or more shard files in the layout written by :func:`src.data.ic_replay_buffer_hdf5.save_trajectories_hdf5`
(flat time axis, ``episode_starts`` / ``episode_lengths``, optional ``images_view_*``). Keeps only a
lightweight index and running stats in RAM; each sample reads **one contiguous row range inside a
single file** (one slice per stored array — no cross-file stitching and no walking episode boundaries
at read time).

**Sampling:** pick a file with probability proportional to how many valid length-``horizon`` (or
shorter, padded) windows it contains, then a uniform start offset in that file. This is **not**
the old global “sort all episodes by return and concatenate” AD timeline; ``context_sort_ascending``
and ``top_n_episodes`` are accepted for API compatibility with :meth:`src.train.main` but **ignored**
for window placement.

**Startup:** :meth:`_build_index` still scans episodes (for ``min_traj_len``, return stats, and
state mean/std) with one file handle open per shard.

Optional ``observation_slice`` (e.g. ManiSkill vision: proprio + tcp only) is applied to every
state read so ``state_dim`` / normalization match image-only training.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from loguru import logger as log
from torch.utils.data import Dataset

from src.data.ic_replay_buffer_hdf5 import (
    HDF5_TRAJ_FILE_ATTR,
    HDF5_TRAJ_VERSION,
    _sorted_images_view_keys,
)


@dataclass(frozen=True)
class _EpisodeRef:
    """Episode span used only for stats (returns, mean/std over eligible timesteps)."""

    file_idx: int
    start: int
    length: int
    ret: float


class ICReplayBufferDataset(Dataset):
    """Random contiguous windows for AD; each window lies entirely inside one shard file."""

    def __init__(
        self,
        hdf5_paths: Sequence[Path],
        *,
        horizon: int,
        rtg_scale: float,
        device: torch.device,
        context_dim: int,
        min_traj_len: int,
        context_sort_ascending: bool,
        use_vision: bool,
        seed: int,
        max_training_examples: int = 500_000,
        top_n_episodes: int = 50_000,
        observation_slice: Optional[slice] = None,
    ) -> None:
        self.paths = [Path(p).expanduser().resolve() for p in hdf5_paths]
        self.horizon = int(horizon)
        self._query_length = int(horizon)
        self.rtg_scale = float(rtg_scale)
        self.device = device
        self.context_dim = int(context_dim)
        self.min_traj_len = int(min_traj_len)
        self.context_sort_ascending = bool(context_sort_ascending)
        self.use_vision = bool(use_vision)
        self._obs_slice: Optional[slice] = observation_slice
        self._seed = int(seed)
        self._max_examples = int(max_training_examples)
        self.total_prompt_len = 0
        self.prompt_length = None
        self.max_prompt_trajectory_length = None
        self.trajectories: List[Dict[str, Any]] = []
        self.task_instructions: Optional[List[str]] = None
        self.num_context_trajectories = 0
        self.max_episode_steps = 0
        self.context_subsample_strategy = "none"

        self._open_pid: Optional[int] = None
        self._open_files: Optional[List[h5py.File]] = None

        self._file_total_t = np.zeros(0, dtype=np.int64)
        self._start_offsets = np.zeros(1, dtype=np.int64)
        self._total_valid_starts = 0
        self._image_view_count = 0
        self._image_hwc: Optional[Tuple[int, int, int]] = None

        self.state_dim = 0
        self.act_dim = 0
        self.state_mean = np.zeros((1,), dtype=np.float32)
        self.state_std = np.ones((1,), dtype=np.float32)
        self.return_min = 0.0
        self.return_max = 0.0
        self.return_avg = 0.0

        self._build_index(int(top_n_episodes))
        # ``_init_image_layout`` may leave HDF5 files open; workers pickle the dataset, and h5py
        # files are not picklable. Close here so each worker re-opens via ``_open_files_for_pid``.
        self._close_files()

    def __getstate__(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["_open_files"] = None
        d["_open_pid"] = None
        return d

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._open_files = None
        self._open_pid = None

    def _apply_obs_slice(self, obs: np.ndarray) -> np.ndarray:
        if self._obs_slice is None:
            return obs
        o = np.asarray(obs)
        if o.shape[-1] < self._obs_slice.stop:
            raise ValueError(
                f"observations last dim {o.shape[-1]} < observation_slice.stop {self._obs_slice.stop}"
            )
        return np.asarray(o[..., self._obs_slice], dtype=o.dtype)

    def _open_files_for_pid(self) -> List[h5py.File]:
        pid = os.getpid()
        if self._open_files is not None and self._open_pid == pid:
            return self._open_files
        self._close_files()
        self._open_files = [h5py.File(p, "r") for p in self.paths]
        self._open_pid = pid
        return self._open_files

    def _close_files(self) -> None:
        if self._open_files is not None:
            for f in self._open_files:
                try:
                    f.close()
                except Exception:
                    pass
        self._open_files = None
        self._open_pid = None

    def __del__(self) -> None:
        self._close_files()

    def _assert_replay_buffer_v2(self, f: h5py.File, path: Path) -> None:
        if f.attrs.get("format") != HDF5_TRAJ_FILE_ATTR:
            raise ValueError(f"{path}: not an ICL replay buffer file (missing/wrong format attr)")
        if int(f.attrs.get("version", 0)) != HDF5_TRAJ_VERSION:
            raise ValueError(f"{path}: expected replay buffer version {HDF5_TRAJ_VERSION}")

    def _build_index(self, top_n_episodes: int) -> None:
        del top_n_episodes  # API compat with train.py; in-file sampling does not use AD timeline cap.
        eps: List[_EpisodeRef] = []
        sum_obs: Optional[np.ndarray] = None
        sum_sq_obs: Optional[np.ndarray] = None
        n_obs = 0
        all_returns: List[float] = []
        state_dim: Optional[int] = None
        act_dim: Optional[int] = None
        file_totals: List[int] = []
        starts_per_file: List[int] = []

        K = int(self.horizon)
        for file_idx, p in enumerate(self.paths):
            with h5py.File(p, "r") as f:
                self._assert_replay_buffer_v2(f, p)
                for name in (
                    "episode_starts",
                    "episode_lengths",
                    "observations",
                    "actions",
                    "rewards",
                    "terminals",
                ):
                    if name not in f:
                        raise ValueError(f"{p}: missing dataset {name!r}")
                obs = f["observations"]
                act = f["actions"]
                raw_obs_d = int(obs.shape[1])
                probe = np.asarray(obs[0:1], dtype=np.float64)
                sliced_d = int(self._apply_obs_slice(probe).shape[-1])
                if self._obs_slice is not None and raw_obs_d < int(self._obs_slice.stop):
                    raise ValueError(
                        f"{p}: observations dim {raw_obs_d} < observation_slice.stop {self._obs_slice.stop}"
                    )
                if state_dim is None:
                    state_dim = sliced_d
                    act_dim = int(act.shape[1])
                elif sliced_d != state_dim or act_dim != int(act.shape[1]):
                    raise ValueError(
                        f"Inconsistent dims across files: expected state_dim={state_dim}, act_dim={act_dim}, "
                        f"got (sliced_state={sliced_d}, act={int(act.shape[1])}) at {p}"
                    )
                T_file = int(obs.shape[0])
                file_totals.append(T_file)
                if T_file >= K:
                    starts_per_file.append(T_file - K + 1)
                elif T_file > 0:
                    starts_per_file.append(1)
                else:
                    starts_per_file.append(0)

                starts = np.asarray(f["episode_starts"], dtype=np.int64).reshape(-1)
                lens = np.asarray(f["episode_lengths"], dtype=np.int64).reshape(-1)
                rew = np.asarray(f["rewards"], dtype=np.float64).reshape(-1)
                cum = np.empty(rew.shape[0] + 1, dtype=np.float64)
                cum[0] = 0.0
                np.cumsum(rew, out=cum[1:])

                for s, l in zip(starts, lens):
                    T = int(l)
                    if T < self.min_traj_len:
                        continue
                    ss = int(s)
                    rr = float(cum[ss + T] - cum[ss])
                    eps.append(_EpisodeRef(file_idx=file_idx, start=ss, length=T, ret=rr))
                    all_returns.append(rr)

                    ob = self._apply_obs_slice(np.asarray(obs[ss : ss + T], dtype=np.float64))
                    if sum_obs is None:
                        sum_obs = np.zeros(ob.shape[1], dtype=np.float64)
                        sum_sq_obs = np.zeros(ob.shape[1], dtype=np.float64)
                    sum_obs += ob.sum(axis=0)
                    sum_sq_obs += np.square(ob, dtype=np.float64).sum(axis=0)
                    n_obs += int(T)

        if state_dim is None or act_dim is None or not eps:
            raise ValueError("No eligible episodes found for IC replay buffer dataset (check min_traj_len)")
        assert sum_obs is not None and sum_sq_obs is not None

        self.state_dim = int(state_dim)
        self.act_dim = int(act_dim)
        mean = sum_obs / max(1, n_obs)
        var = np.maximum(sum_sq_obs / max(1, n_obs) - np.square(mean), 1e-12)
        self.state_mean = mean.astype(np.float32)
        self.state_std = np.sqrt(var, dtype=np.float64).astype(np.float32) + 1e-6
        r = np.asarray(all_returns, dtype=np.float64)
        self.return_min = float(r.min())
        self.return_max = float(r.max())
        self.return_avg = float(r.mean())
        self.max_episode_steps = max(int(e.length) for e in eps)

        self._file_total_t = np.asarray(file_totals, dtype=np.int64)
        counts = np.asarray(starts_per_file, dtype=np.int64)
        self._start_offsets = np.zeros(len(counts) + 1, dtype=np.int64)
        self._start_offsets[1:] = np.cumsum(counts)
        self._total_valid_starts = int(self._start_offsets[-1])
        if self._total_valid_starts <= 0:
            raise ValueError(
                "No valid in-file windows (every shard empty or shorter than horizon with no pad path)"
            )

        if self.use_vision:
            self._init_image_layout()

        log.info(
            "ICReplayBufferDataset (in-file windows): files={} valid_starts={} eligible_eps={} "
            "state_dim={} act_dim={} (context_sort_ascending / top_n_episodes ignored for sampling)",
            len(self.paths),
            self._total_valid_starts,
            len(eps),
            self.state_dim,
            self.act_dim,
        )

    def _init_image_layout(self) -> None:
        files = self._open_files_for_pid()
        for f in files:
            keys = _sorted_images_view_keys(f)
            if not keys:
                continue
            d = f[keys[0]]
            if d.ndim != 4 or int(d.shape[-1]) != 3:
                continue
            self._image_view_count = len(keys)
            self._image_hwc = (int(d.shape[1]), int(d.shape[2]), int(d.shape[3]))
            return
        log.warning("ICReplayBufferDataset: use_vision=true but no images_view_* found; vision disabled")
        self.use_vision = False

    def __len__(self) -> int:
        return int(self._max_examples)

    def _locate_file_and_start(self, r: int) -> Tuple[int, int]:
        """Map flat index ``r`` in ``[0, total_valid_starts)`` to ``(file_idx, start_row)``."""
        i = int(np.searchsorted(self._start_offsets, r, side="right")) - 1
        if i < 0:
            i = 0
        g = int(r - int(self._start_offsets[i]))
        return i, g

    def _read_file_contiguous(
        self, file_idx: int, g: int, tlen: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        files = self._open_files_for_pid()
        hf = files[file_idx]
        e = int(g) + int(tlen)
        obs = self._apply_obs_slice(np.asarray(hf["observations"][g:e], dtype=np.float32))
        act = np.asarray(hf["actions"][g:e], dtype=np.float32)
        rew = np.asarray(hf["rewards"][g:e], dtype=np.float32).reshape(-1, 1)
        term = np.asarray(hf["terminals"][g:e], dtype=np.float32)
        return obs, act, rew, term

    def _read_image_file_slice(
        self, file_idx: int, g: int, tlen: int
    ) -> Optional[List[torch.Tensor]]:
        if not self.use_vision or self._image_hwc is None or self._image_view_count <= 0:
            return None
        files = self._open_files_for_pid()
        hf = files[file_idx]
        h, w, c = self._image_hwc
        e = int(g) + int(tlen)
        out: List[torch.Tensor] = []
        for v in range(self._image_view_count):
            key = f"images_view_{v}"
            if key in hf:
                seg_u8 = np.asarray(hf[key][g:e], dtype=np.uint8)
            else:
                seg_u8 = np.zeros((tlen, h, w, c), dtype=np.uint8)
            seg = seg_u8.astype(np.float32)
            seg = np.transpose(seg, (0, 3, 1, 2))
            out.append(torch.from_numpy(seg).float().to(self.device).unsqueeze(0))
        return out

    def __getitem__(self, index: int) -> Tuple[Any, ...]:
        K = self._query_length
        rng = np.random.default_rng(self._seed + int(index))
        r = int(rng.integers(0, self._total_valid_starts))
        file_idx, g = self._locate_file_and_start(r)
        T = int(self._file_total_t[file_idx])
        if T >= K:
            tlen = K
            pad_len = 0
        else:
            g = 0
            tlen = T
            pad_len = K - tlen

        state_seg, action_seg, reward_seg, done_seg = self._read_file_contiguous(file_idx, g, tlen)
        context_seg = np.zeros((tlen, self.context_dim), dtype=np.float32)
        rtg_seg = np.zeros((tlen, 1), dtype=np.float32)
        ts_seg = np.arange(tlen, dtype=np.float32)
        query_tid = 1

        if pad_len > 0:
            state_seg = np.concatenate(
                [np.zeros((pad_len, self.state_dim), dtype=np.float32), state_seg], axis=0
            )
            context_seg = np.concatenate(
                [np.zeros((pad_len, self.context_dim), dtype=np.float32), context_seg], axis=0
            )
            action_seg = np.concatenate(
                [np.ones((pad_len, self.act_dim), dtype=np.float32) * -10.0, action_seg], axis=0
            )
            reward_seg = np.concatenate(
                [np.zeros((pad_len, 1), dtype=np.float32), reward_seg], axis=0
            )
            done_seg = np.concatenate([np.ones(pad_len, dtype=np.float32) * 2.0, done_seg], axis=0)
            rtg_seg = np.concatenate([np.zeros((pad_len, 1), dtype=np.float32), rtg_seg], axis=0)
            ts_seg = np.concatenate([np.zeros(pad_len, dtype=np.float32), ts_seg], axis=0)
            mask_seg = np.concatenate(
                [np.zeros(pad_len, dtype=np.float32), np.ones(tlen, dtype=np.float32)]
            )
            trial_seg = np.concatenate(
                [
                    np.zeros(pad_len, dtype=np.float32),
                    np.full(tlen, float(query_tid), dtype=np.float32),
                ]
            )
        else:
            mask_seg = np.ones(K, dtype=np.float32)
            trial_seg = np.full(K, float(query_tid), dtype=np.float32)

        state_seg = (state_seg - self.state_mean) / self.state_std

        ps = np.zeros((0, self.state_dim), dtype=np.float32)
        pa = np.ones((0, self.act_dim), dtype=np.float32) * -10.0
        pr = np.zeros((0, 1), dtype=np.float32)
        prtg = np.zeros((0, 1), dtype=np.float32)
        pts = np.zeros(0, dtype=np.float32)
        pm = np.zeros(0, dtype=np.float32)
        ppt = np.zeros(0, dtype=np.float32)
        instruction = ""

        def _to_t(x: np.ndarray, long_type: bool = False) -> torch.Tensor:
            t = torch.from_numpy(np.asarray(x))
            return t.long().to(self.device) if long_type else t.float().to(self.device)

        images_t = self._read_image_file_slice(file_idx, g, tlen)
        if images_t is not None and pad_len > 0:
            for i, t_img in enumerate(images_t):
                pad = torch.zeros((1, pad_len) + tuple(t_img.shape[2:]), device=t_img.device)
                images_t[i] = torch.cat([pad, t_img], dim=1)

        result: Tuple[Any, ...] = (
            _to_t(state_seg),
            _to_t(context_seg),
            _to_t(action_seg),
            _to_t(reward_seg),
            _to_t(done_seg, True),
            _to_t(rtg_seg),
            _to_t(ts_seg, True),
            _to_t(mask_seg),
            _to_t(trial_seg, True),
            _to_t(ps),
            _to_t(pa),
            _to_t(pr),
            _to_t(prtg),
            _to_t(pts, True),
            _to_t(pm),
            _to_t(ppt, True),
            instruction,
        )
        if self.use_vision:
            result = result + (images_t,)
        return result
