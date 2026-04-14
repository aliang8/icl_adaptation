"""
AD dataset: episodes with length ``>= min_traj_len`` (default 10); then the lowest-return subset (capped
by ``ALGORITHM_DISTILLATION_TOP_N_EPISODES``) is concatenated low→high by sum of rewards (reversed if
``context_sort_ascending`` is false). Sample ``data.horizon`` windows along that timeline.

``max_episode_steps`` still sets model / padding horizon; episode lengths in the timeline may vary.

Set ``model.max_length`` ≥ train ``horizon`` and optional eval ``query_history_length``.
``sequence_token_layout=state_action_reward``; query ``rtg`` is zeros for collate.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger as log

from src.data.dataset import ICLTrajectoryDatasetBase, PromptArrays, _empty_prompt_segment
from src.data.trajectories import trajectory_return

# Max number of lowest-return (after length filter) episodes concatenated into the AD timeline.
ALGORITHM_DISTILLATION_TOP_N_EPISODES = 50000


class AlgorithmDistillationTrajectoryDataset(ICLTrajectoryDatasetBase):
    """Builds one flat timeline from the lowest-return capped trajectories."""

    def __init__(
        self,
        trajectories: List[Dict[str, np.ndarray]],
        horizon: int,
        max_episode_steps: int,
        rtg_scale: float,
        device: torch.device,
        prompt_trajectories_per_task: Optional[List[List[Dict[str, np.ndarray]]]] = None,
        context_dim: int = 16,
        state_dim: int = 27,
        act_dim: int = 8,
        total_epi_per_task: int = 100,
        context_sort_ascending: bool = True,
        context_sampling: str = "random",
        trajectory_contexts: Optional[Dict[int, np.ndarray]] = None,
        min_traj_len: int = 10,
        **kwargs: Any,
    ):
        kwargs = dict(kwargs)
        kwargs["num_context_trajectories"] = 0
        kwargs["lazy_dataset"] = True
        kwargs.setdefault("randomize_num_context_trajectories", False)
        self.min_traj_len = int(min_traj_len)
        super().__init__(
            trajectories=trajectories,
            horizon=horizon,
            max_episode_steps=max_episode_steps,
            rtg_scale=rtg_scale,
            device=device,
            prompt_trajectories_per_task=prompt_trajectories_per_task,
            context_dim=context_dim,
            state_dim=state_dim,
            act_dim=act_dim,
            total_epi_per_task=total_epi_per_task,
            context_sort_ascending=context_sort_ascending,
            context_sampling=context_sampling,
            trajectory_contexts=trajectory_contexts,
            **kwargs,
        )
        self._query_length = int(self.horizon)
        self.total_prompt_len = 0
        self.prompt_length = None
        self.max_prompt_trajectory_length = None
        self._build_ad_timeline()
        self.horizon = int(self._query_length)
        log.info(
            "AlgorithmDistillation: timeline_len={} context_K={} sort_returns_ascending={}",
            self._timeline_len,
            self._query_length,
            context_sort_ascending,
        )

    def _build_prompt(
        self,
        chosen: List[Dict[str, np.ndarray]],
        state_mean: np.ndarray,
        state_std: np.ndarray,
        total_plen: int,
    ) -> PromptArrays:
        return _empty_prompt_segment(self.state_dim, self.act_dim)

    def _build_ad_timeline(self) -> None:
        trs_all = self.trajectories
        mmin = int(self.min_traj_len)
        kept_indices: List[int] = []
        for i, t in enumerate(trs_all):
            T = int(np.asarray(t["rewards"], dtype=np.float32).reshape(-1).shape[0])
            if T >= mmin:
                kept_indices.append(i)
        if len(kept_indices) < len(trs_all):
            log.info(
                "AlgorithmDistillation: kept {}/{} trajectories with length >= min_traj_len={}",
                len(kept_indices),
                len(trs_all),
                mmin,
            )
        trs = [trs_all[i] for i in kept_indices]
        n = len(trs)
        if n == 0:
            self._timeline_len = 0
            self._obs_cat = np.zeros((0, self.state_dim), dtype=np.float32)
            self._act_cat = np.zeros((0, self.act_dim), dtype=np.float32)
            self._rew_cat = np.zeros((0,), dtype=np.float32)
            self._done_cat = np.zeros((0,), dtype=np.float32)
            self._ctx_cat = np.zeros((0, self.context_dim), dtype=np.float32)
            self._ad_images = None
            return

        rets = np.array([trajectory_return(t) for t in trs], dtype=np.float64)
        _n_keep = min(int(ALGORITHM_DISTILLATION_TOP_N_EPISODES), n)
        rank_asc = np.argsort(rets, kind="stable")
        order = rank_asc[:_n_keep]
        if not self.context_sort_ascending:
            order = order[::-1]
        if _n_keep < n:
            log.info(
                "AlgorithmDistillation: using lowest-{} of {} filtered trajectories by return (cap={})",
                _n_keep,
                n,
                ALGORITHM_DISTILLATION_TOP_N_EPISODES,
            )
        log.info(
            "AlgorithmDistillation: concat along timeline (context_sort_ascending={}): "
            "first_episode_sum_r={:.6g} last_episode_sum_r={:.6g} (among lowest-{} by return)",
            self.context_sort_ascending,
            float(rets[int(order[0])]),
            float(rets[int(order[-1])]),
            _n_keep,
        )

        obs_parts: List[np.ndarray] = []
        act_parts: List[np.ndarray] = []
        rew_parts: List[np.ndarray] = []
        done_parts: List[np.ndarray] = []
        ctx_parts: List[np.ndarray] = []
        img_parts_per_view: Optional[List[List[np.ndarray]]] = None
        image_template: Optional[List[np.ndarray]] = None

        if self.use_vision:
            for j in range(len(order)):
                t = trs[int(order[j])]
                imgs = t.get("images")
                if isinstance(imgs, list) and len(imgs) > 0:
                    image_template = [np.asarray(v, dtype=np.uint8) for v in imgs]
                    break
            if image_template is not None:
                img_parts_per_view = [[] for _ in range(len(image_template))]

        for j in range(len(order)):
            pos_in_filtered = int(order[j])
            orig_idx = int(kept_indices[pos_in_filtered])
            t = trs[pos_in_filtered]
            rew = np.asarray(t["rewards"], dtype=np.float32).reshape(-1)
            T = int(rew.shape[0])
            if T < mmin:
                raise RuntimeError(
                    f"AD timeline: internal error, expected T>={mmin} after filter, got T={T}"
                )
            obs = np.asarray(t["observations"], dtype=np.float32)
            act = np.asarray(t["actions"], dtype=np.float32)
            if obs.shape[0] < T:
                raise ValueError(
                    f"AD timeline: observations length {obs.shape[0]} < rewards length {T}"
                )
            obs = obs[:T]
            act = act[:T]
            term = np.asarray(t["terminals"], dtype=np.float32).reshape(-1)[:T]

            obs_parts.append(obs)
            act_parts.append(act)
            rew_parts.append(rew)
            done_parts.append(term)

            tc = self.trajectory_contexts.get(orig_idx) if orig_idx >= 0 else None
            if tc is None:
                ctx_parts.append(np.zeros((T, self.context_dim), dtype=np.float32))
            else:
                tc_arr = np.asarray(tc, dtype=np.float32)
                if tc_arr.shape[0] < T:
                    raise ValueError(
                        f"trajectory_contexts[{orig_idx}] length {tc_arr.shape[0]} < T={T}"
                    )
                ctx_parts.append(tc_arr[:T])

            if img_parts_per_view is not None and image_template is not None:
                imgs = t.get("images")
                if isinstance(imgs, list) and len(imgs) == len(image_template):
                    for v, arr in enumerate(imgs):
                        a = np.asarray(arr, dtype=np.uint8)
                        if a.shape[0] != T:
                            raise ValueError(
                                f"AD timeline: images[{v}] length {a.shape[0]} != T={T} for a trajectory"
                            )
                        img_parts_per_view[v].append(a)
                else:
                    h, w, c = (
                        int(image_template[0].shape[1]),
                        int(image_template[0].shape[2]),
                        int(image_template[0].shape[3]),
                    )
                    for v in range(len(image_template)):
                        img_parts_per_view[v].append(np.zeros((T, h, w, c), dtype=np.uint8))

        if not rew_parts:
            self._timeline_len = 0
            self._obs_cat = np.zeros((0, self.state_dim), dtype=np.float32)
            self._act_cat = np.zeros((0, self.act_dim), dtype=np.float32)
            self._rew_cat = np.zeros((0,), dtype=np.float32)
            self._done_cat = np.zeros((0,), dtype=np.float32)
            self._ctx_cat = np.zeros((0, self.context_dim), dtype=np.float32)
            self._ad_images = None
            return

        self._obs_cat = np.concatenate(obs_parts, axis=0)
        self._act_cat = np.concatenate(act_parts, axis=0)
        self._rew_cat = np.concatenate(rew_parts, axis=0)
        self._done_cat = np.concatenate(done_parts, axis=0)
        self._ctx_cat = np.concatenate(ctx_parts, axis=0)
        self._timeline_len = int(self._rew_cat.shape[0])

        if img_parts_per_view is not None and all(len(parts) > 0 for parts in img_parts_per_view):
            self._ad_images = [np.concatenate(parts, axis=0) for parts in img_parts_per_view]
        else:
            self._ad_images = None
            if self.use_vision and image_template is None:
                log.warning(
                    "AlgorithmDistillation: use_vision=True but no trajectory had 'images'; vision disabled for this dataset."
                )

    def __len__(self) -> int:
        if self._max_examples is not None:
            return int(self._max_examples)
        K = self._query_length
        L = self._timeline_len
        return max(1, L - K + 1) if L >= K else 1

    def __getitem__(self, index: int) -> Tuple[Any, ...]:
        K = self._query_length
        L = self._timeline_len
        if L <= 0:
            raise RuntimeError(
                "AlgorithmDistillationTrajectoryDataset: empty timeline (no transitions)."
            )

        rng = np.random.default_rng(self._seed + int(index))
        query_tid = 1

        if L >= K:
            g = int(rng.integers(0, L - K + 1))
            tlen = K
            pad_len = 0
        else:
            g = 0
            tlen = L
            pad_len = K - tlen

        state_seg = np.asarray(self._obs_cat[g : g + tlen], dtype=np.float32)
        context_seg = np.asarray(self._ctx_cat[g : g + tlen], dtype=np.float32)
        action_seg = np.asarray(self._act_cat[g : g + tlen], dtype=np.float32)
        reward_seg = np.asarray(self._rew_cat[g : g + tlen], dtype=np.float32).reshape(-1, 1)
        done_seg = np.asarray(self._done_cat[g : g + tlen], dtype=np.float32)
        rtg_seg = np.zeros((tlen, 1), dtype=np.float32)
        ts_seg = np.arange(tlen, dtype=np.float32)

        if pad_len > 0:
            z = np.zeros((pad_len, state_seg.shape[1]), dtype=np.float32)
            state_seg = np.concatenate([z, state_seg], axis=0)
            state_seg = (state_seg - self.state_mean) / self.state_std
            context_seg = np.concatenate(
                [np.zeros((pad_len, self.context_dim), dtype=np.float32), context_seg], axis=0
            )
            action_seg = np.concatenate(
                [np.ones((pad_len, action_seg.shape[1]), dtype=np.float32) * -10.0, action_seg],
                axis=0,
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
            state_seg = (state_seg - self.state_mean) / self.state_std
            mask_seg = np.ones(K, dtype=np.float32)
            trial_seg = np.full(K, float(query_tid), dtype=np.float32)

        ps, pa, pr, prtg, pts, pm, ppt = _empty_prompt_segment(self.state_dim, self.act_dim)
        instruction = ""

        def _to_t(x: np.ndarray, long_type: bool = False) -> torch.Tensor:
            t = torch.from_numpy(np.asarray(x))
            return t.long().to(self.device) if long_type else t.float().to(self.device)

        images_t: Optional[List[torch.Tensor]] = None
        if self.use_vision and self._ad_images is not None:
            images_t = []
            for view_arr in self._ad_images:
                seg = np.asarray(view_arr[g : g + tlen], dtype=np.float32)
                if seg.ndim == 4:
                    seg = np.transpose(seg, (0, 3, 1, 2))
                if pad_len > 0:
                    pad_shape = (pad_len,) + seg.shape[1:]
                    seg = np.concatenate([np.zeros(pad_shape, dtype=np.float32), seg], axis=0)
                t_img = torch.from_numpy(seg).float().to(self.device)
                if t_img.dim() == 4:
                    t_img = t_img.unsqueeze(0)
                images_t.append(t_img)

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
