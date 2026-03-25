"""
In-context trajectory dataset: sample N context trajectories per segment (same task),
sort by increasing return, concatenate for training. At inference, previous rollouts
sorted ascending for zero-shot adaptation.

Two dataset classes:
- SubsampledICLTrajectoryDataset: prompt = fixed-length segments (prompt_length steps per traj).
- FullTrajectoryICLTrajectoryDataset: prompt = full trajectory(ies), capped by max_total_prompt_length.
"""

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger as log
from torch.utils.data import Dataset

from src.data.trajectories import discount_cumsum, sample_context_trajectories

PromptArrays = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _subsample_indices(length: int, cap: Optional[int], strategy: str) -> np.ndarray:
    """Select ordered indices from [0, length) according to strategy."""
    if length <= 0:
        return np.zeros(0, dtype=np.int64)
    if strategy == "none":
        return np.arange(length, dtype=np.int64)
    if cap is None or cap <= 0 or length <= cap:
        return np.arange(length, dtype=np.int64)
    if strategy == "last":
        return np.arange(length - cap, length, dtype=np.int64)
    if strategy == "uniform":
        return np.linspace(0, length - 1, num=cap, dtype=np.int64)
    if strategy == "random":
        idx = np.sort(np.random.choice(length, size=cap, replace=False))
        return idx.astype(np.int64)
    raise ValueError(
        f"Unsupported context_subsample_strategy='{strategy}'. "
        "Use one of: none, last, uniform, random."
    )


def _pad_or_trim_prompt(
    ps: np.ndarray,
    pa: np.ndarray,
    pr: np.ndarray,
    prtg: np.ndarray,
    pts: np.ndarray,
    pm: np.ndarray,
    total_plen: int,
    state_dim: int,
    act_dim: int,
    take_last: bool = False,
) -> PromptArrays:
    """Pad at front or trim to total_plen. If take_last, trim from the start; else trim from the end."""
    if ps.shape[0] >= total_plen:
        if take_last:
            ps, pa, pr, prtg, pts, pm = (
                ps[-total_plen:],
                pa[-total_plen:],
                pr[-total_plen:],
                prtg[-total_plen:],
                pts[-total_plen:],
                pm[-total_plen:],
            )
        else:
            ps, pa, pr, prtg, pts, pm = (
                ps[:total_plen],
                pa[:total_plen],
                pr[:total_plen],
                prtg[:total_plen],
                pts[:total_plen],
                pm[:total_plen],
            )
    else:
        pad_len = total_plen - ps.shape[0]
        ps = np.concatenate([np.zeros((pad_len, state_dim)), ps], axis=0)
        pa = np.concatenate([np.ones((pad_len, act_dim)) * -10.0, pa], axis=0)
        pr = np.concatenate([np.zeros((pad_len, 1)), pr], axis=0)
        prtg = np.concatenate([np.zeros((pad_len, 1)), prtg], axis=0)
        pts = np.concatenate([np.zeros(pad_len), pts], axis=0)
        pm = np.concatenate([np.zeros(pad_len), pm], axis=0)
    return ps, pa, pr, prtg, pts, pm


class ICLTrajectoryDatasetBase(Dataset, ABC):
    """
    Base dataset of (state, context, action, reward, done, rtg, timesteps, mask) segments
    with in-context prompt. Subclasses define how the prompt is built (subsampled vs full trajectory).
    """

    def __init__(
        self,
        trajectories: List[Dict[str, np.ndarray]],
        horizon: int,
        max_episode_steps: int,
        return_scale: float,
        device: torch.device,
        prompt_trajectories_per_task: Optional[List[List[Dict[str, np.ndarray]]]] = None,
        context_dim: int = 16,
        state_dim: int = 27,
        act_dim: int = 8,
        scale: float = 500.0,
        total_epi_per_task: int = 100,
        num_context_trajectories: int = 1,
        randomize_num_context_trajectories: bool = False,
        context_sort_ascending: bool = True,
        context_sampling: str = "random",
        trajectory_contexts: Optional[Dict[int, np.ndarray]] = None,
        task_instructions: Optional[List[str]] = None,
        lazy_dataset: bool = False,
        max_training_examples: int = 500_000,
        seed: int = 0,
        query_history_length: Optional[int] = None,
        use_vision: bool = False,
        image_keys: Optional[List[str]] = None,
        context_subsample_strategy: str = "none",
        **kwargs: Any,
    ):
        self.trajectories = trajectories
        self.use_vision = use_vision
        self.image_keys = image_keys or []
        self.horizon = horizon
        # Query = last K steps of current trajectory; K=1 = OpenVLA-style; None = use horizon
        self._query_length = horizon if query_history_length is None else query_history_length
        self.max_episode_steps = max_episode_steps
        self.return_scale = return_scale
        self.device = device
        self.prompt_trajectories_per_task = prompt_trajectories_per_task or []
        self.context_dim = context_dim
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.scale = scale
        self.total_epi_per_task = total_epi_per_task
        self.num_context_trajectories = num_context_trajectories
        self.randomize_num_context_trajectories = randomize_num_context_trajectories
        if randomize_num_context_trajectories:
            log.info(
                "Training with random prior count m ~ Uniform{{0..{}}} context trajectories per sample",
                num_context_trajectories,
            )
        self.context_sort_ascending = context_sort_ascending
        self.context_sampling = context_sampling
        self.trajectory_contexts = trajectory_contexts or {}
        self.task_instructions = task_instructions
        self.prompt_length = None
        self.max_prompt_trajectory_length = None
        self.context_subsample_strategy = context_subsample_strategy
        self._lazy = lazy_dataset
        self._max_examples = max_training_examples if lazy_dataset else None
        self._seed = seed
        self._prompt_return_log_count = 0
        self._task_ids: List[int] = []

        log.debug("Computing state mean/std over {} trajectories...", len(trajectories))
        states = np.concatenate([t["observations"] for t in trajectories], axis=0)
        self.state_mean = np.mean(states, axis=0)
        self.state_std = np.std(states, axis=0) + 1e-6
        returns = np.array([t["rewards"].sum() for t in trajectories])
        self.return_min = float(returns.min())
        self.return_max = float(returns.max())
        self.return_avg = float(returns.mean())

    def _sample_num_context_trajectories(self) -> int:
        """Max prior count N = num_context_trajectories; m in {0..N} when randomize, else always N."""
        n = self.num_context_trajectories
        if n <= 0:
            return 0
        if not self.randomize_num_context_trajectories:
            return n
        return random.randint(0, n)

    def _choose_context_trajectories(
        self, prompt_list: List[Dict[str, np.ndarray]], query_traj: Dict[str, np.ndarray]
    ) -> List[Dict[str, np.ndarray]]:
        """Prior demos for the prompt; [] if m=0 or no pool; [query_traj] only when pool is empty (legacy)."""
        if not prompt_list:
            return [query_traj]
        m = self._sample_num_context_trajectories()
        if m <= 0:
            return []
        return sample_context_trajectories(
            prompt_list,
            m,
            ascending=self.context_sort_ascending,
            sampling=self.context_sampling,
        )

    @abstractmethod
    def _build_prompt(
        self,
        chosen: List[Dict[str, np.ndarray]],
        state_mean: np.ndarray,
        state_std: np.ndarray,
        total_plen: int,
    ) -> PromptArrays:
        """Build (ps, pa, pr, prtg, pts, pm) of length total_plen from chosen context trajectories."""
        raise NotImplementedError

    def _build_main_segment(
        self,
        traj: Dict[str, np.ndarray],
        si: int,
        traj_contexts: np.ndarray,
    ) -> Tuple[np.ndarray, ...]:
        """Build the 8 main-segment arrays for (traj, si). Uses last K steps ending at si (K = query_history_length or horizon).
        So: segment = [si-K+1 .. si] padded to length K at front. K=1 is OpenVLA-style (current obs only)."""
        K = self._query_length
        max_ep_len = self.max_episode_steps
        state_mean, state_std = self.state_mean, self.state_std
        # Last K steps ending at si (inclusive): indices start..si
        start = max(0, si - K + 1)
        state_seg = traj["observations"][start : si + 1]
        action_seg = traj["actions"][start : si + 1]
        reward_seg = traj["rewards"][start : si + 1].reshape(-1, 1)
        context_seg = traj_contexts[start : si + 1]
        done_seg = traj.get("terminals", traj.get("dones", np.zeros(len(traj["rewards"]))))[
            start : si + 1
        ]
        tlen = state_seg.shape[0]
        pad_len = K - tlen
        state_seg = np.concatenate([np.zeros((pad_len, state_seg.shape[1])), state_seg], axis=0)
        state_seg = (state_seg - state_mean) / state_std
        context_seg = np.concatenate(
            [np.zeros((pad_len, context_seg.shape[1])), context_seg], axis=0
        )
        action_seg = np.concatenate(
            [np.ones((pad_len, action_seg.shape[1])) * -10.0, action_seg], axis=0
        )
        reward_seg = np.concatenate([np.zeros((pad_len, 1)), reward_seg], axis=0)
        done_seg = np.concatenate([np.ones(pad_len) * 2, done_seg], axis=0)
        rtg_seg = discount_cumsum(traj["rewards"][start:], gamma=1.0)[:tlen].reshape(-1, 1)
        rtg_seg = np.concatenate([np.zeros((pad_len, 1)), rtg_seg], axis=0) / self.return_scale
        ts_seg = np.arange(start, start + tlen, dtype=np.float32)
        ts_seg[ts_seg >= max_ep_len] = max_ep_len - 1
        ts_seg = np.concatenate([np.zeros(pad_len), ts_seg], axis=0)
        mask_seg = np.concatenate([np.zeros(pad_len), np.ones(tlen)], axis=0)
        return state_seg, context_seg, action_seg, reward_seg, done_seg, rtg_seg, ts_seg, mask_seg

    def _log_prompt_context_returns_sample(
        self,
        chosen: List[Dict[str, np.ndarray]],
        traj_idx: int,
        si: int,
    ) -> None:
        """First 4 samples only: env return per context traj in prompt order (avoids log spam)."""
        if self._prompt_return_log_count >= 4:
            return
        self._prompt_return_log_count += 1
        ordered = [float(np.asarray(t["rewards"], dtype=np.float64).sum()) for t in chosen]
        log.info(
            "[prompt_context_returns] {}/4 | query traj_idx={} si={} | context_sort_ascending={} | "
            "env_returns_prompt_order={}",
            self._prompt_return_log_count,
            traj_idx,
            si,
            self.context_sort_ascending,
            [round(x, 3) for x in ordered],
        )

    def _get_one_sample(
        self,
        traj_idx: int,
        si: int,
    ) -> Tuple[np.ndarray, ...]:
        """Build one full sample (14 arrays + instruction str) for (traj_idx, si). Used by lazy __getitem__."""
        traj = self.trajectories[traj_idx]
        task_id = traj_idx // self.total_epi_per_task if self.total_epi_per_task else 0
        prompt_list = (
            self.prompt_trajectories_per_task[task_id]
            if task_id < len(self.prompt_trajectories_per_task)
            else []
        )
        traj_contexts = self.trajectory_contexts.get(traj_idx)
        if traj_contexts is None:
            traj_contexts = np.zeros(
                (len(traj["observations"]), self.context_dim), dtype=np.float32
            )
        chosen = self._choose_context_trajectories(prompt_list, traj)
        self._log_prompt_context_returns_sample(chosen, traj_idx, si)
        ps, pa, pr, prtg, pts, pm = self._build_prompt(
            chosen, self.state_mean, self.state_std, self.total_prompt_len
        )
        (state_seg, context_seg, action_seg, reward_seg, done_seg, rtg_seg, ts_seg, mask_seg) = (
            self._build_main_segment(traj, si, traj_contexts)
        )
        instruction = ""
        if self.task_instructions and task_id < len(self.task_instructions):
            instruction = self.task_instructions[task_id] or ""
        images_out: Optional[List[np.ndarray]] = None
        if self.use_vision and "images" in traj and isinstance(traj.get("images"), list):
            K = self._query_length
            start = max(0, si - K + 1)
            tlen = si - start + 1
            pad_len = K - tlen
            images_out = []
            for view_arr in traj["images"]:
                seg = view_arr[start : si + 1]
                if pad_len > 0:
                    pad_shape = (pad_len,) + seg.shape[1:]
                    seg = np.concatenate([np.zeros(pad_shape, dtype=seg.dtype), seg], axis=0)
                if seg.ndim == 4:
                    seg = np.transpose(seg, (0, 3, 1, 2))
                images_out.append(seg)
        return (
            state_seg,
            context_seg,
            action_seg,
            reward_seg,
            done_seg,
            rtg_seg,
            ts_seg,
            mask_seg,
            ps,
            pa,
            pr,
            prtg,
            pts,
            pm,
            instruction,
            images_out,
        )

    def _parse_segments(self) -> None:
        states, contexts, actions, rewards, dones, rtg, timesteps, masks = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        prompt_s, prompt_a, prompt_r, prompt_rtg, prompt_ts, prompt_m = [], [], [], [], [], []
        if self.task_instructions is not None:
            self._task_ids = []
        state_mean, state_std = self.state_mean, self.state_std
        total_plen = self.total_prompt_len

        for num, traj in enumerate(self.trajectories):
            task_id = num // self.total_epi_per_task if self.total_epi_per_task else 0
            prompt_list = (
                self.prompt_trajectories_per_task[task_id]
                if task_id < len(self.prompt_trajectories_per_task)
                else []
            )

            traj_contexts = self.trajectory_contexts.get(num)
            if traj_contexts is None:
                traj_contexts = np.zeros(
                    (len(traj["observations"]), self.context_dim), dtype=np.float32
                )

            for si in range(traj["rewards"].shape[0] - 1):
                chosen = self._choose_context_trajectories(prompt_list, traj)
                self._log_prompt_context_returns_sample(chosen, num, si)
                ps, pa, pr, prtg, pts, pm = self._build_prompt(
                    chosen, state_mean, state_std, total_plen
                )
                prompt_s.append(ps)
                prompt_a.append(pa)
                prompt_r.append(pr)
                prompt_rtg.append(prtg)
                prompt_ts.append(pts)
                prompt_m.append(pm)

                (
                    state_seg,
                    context_seg,
                    action_seg,
                    reward_seg,
                    done_seg,
                    rtg_seg,
                    ts_seg,
                    mask_seg,
                ) = self._build_main_segment(traj, si, traj_contexts)
                states.append(state_seg)
                contexts.append(context_seg)
                actions.append(action_seg)
                rewards.append(reward_seg)
                dones.append(done_seg)
                rtg.append(rtg_seg)
                timesteps.append(ts_seg)
                masks.append(mask_seg)
                if self.task_instructions is not None:
                    self._task_ids.append(
                        task_id
                    )  # task_id from loop: num // self.total_epi_per_task

        self.states = torch.from_numpy(np.stack(states)).float().to(self.device)
        self.contexts = torch.from_numpy(np.stack(contexts)).float().to(self.device)
        self.actions = torch.from_numpy(np.stack(actions)).float().to(self.device)
        self.rewards = torch.from_numpy(np.stack(rewards)).float().to(self.device)
        self.dones = torch.from_numpy(np.stack(dones)).long().to(self.device)
        self.rtg = torch.from_numpy(np.stack(rtg)).float().to(self.device)
        self.timesteps = torch.from_numpy(np.stack(timesteps)).long().to(self.device)
        self.masks = torch.from_numpy(np.stack(masks)).float().to(self.device)
        self.prompt_states = torch.from_numpy(np.stack(prompt_s)).float().to(self.device)
        self.prompt_actions = torch.from_numpy(np.stack(prompt_a)).float().to(self.device)
        self.prompt_rewards = torch.from_numpy(np.stack(prompt_r)).float().to(self.device)
        self.prompt_rtg = torch.from_numpy(np.stack(prompt_rtg)).float().to(self.device)
        self.prompt_timesteps = torch.from_numpy(np.stack(prompt_ts)).long().to(self.device)
        self.prompt_masks = torch.from_numpy(np.stack(prompt_m)).float().to(self.device)

    def __len__(self) -> int:
        if self._lazy:
            return self._max_examples
        return self.states.size(0)

    def _get_item_lazy(self, index: int) -> Tuple[Any, ...]:
        rng = np.random.default_rng(self._seed + index)
        traj_idx = int(rng.integers(0, len(self.trajectories)))
        traj = self.trajectories[traj_idx]
        T = traj["rewards"].shape[0] - 1
        if T < 1:
            traj_idx = (traj_idx + 1) % len(self.trajectories)
            traj = self.trajectories[traj_idx]
            T = max(0, traj["rewards"].shape[0] - 1)
        si = int(rng.integers(0, max(1, T)))
        out = self._get_one_sample(traj_idx, si)
        (
            state_seg,
            context_seg,
            action_seg,
            reward_seg,
            done_seg,
            rtg_seg,
            ts_seg,
            mask_seg,
            ps,
            pa,
            pr,
            prtg,
            pts,
            pm,
            instruction,
            images_out,
        ) = out

        def _to_t(x: np.ndarray, long_type: bool = False) -> torch.Tensor:
            t = torch.from_numpy(np.asarray(x))
            return t.long().to(self.device) if long_type else t.float().to(self.device)

        images_t: Optional[List[torch.Tensor]] = None
        if images_out is not None:
            images_t = []
            for arr in images_out:
                t = torch.from_numpy(np.asarray(arr)).float().to(self.device)
                if t.dim() == 4:
                    t = t.unsqueeze(0)
                images_t.append(t)

        result: Tuple[Any, ...] = (
            _to_t(state_seg),
            _to_t(context_seg),
            _to_t(action_seg),
            _to_t(reward_seg),
            _to_t(done_seg, True),
            _to_t(rtg_seg),
            _to_t(ts_seg, True),
            _to_t(mask_seg),
            _to_t(ps),
            _to_t(pa),
            _to_t(pr),
            _to_t(prtg),
            _to_t(pts, True),
            _to_t(pm),
            instruction,
        )
        if self.use_vision:
            result = result + (images_t,)
        return result

    def __getitem__(self, index: int) -> Tuple[Any, ...]:
        if self._lazy:
            return self._get_item_lazy(index)
        instruction = ""
        if self.task_instructions is not None and index < len(self._task_ids):
            tid = self._task_ids[index]
            if tid < len(self.task_instructions):
                instruction = self.task_instructions[tid] or ""
        out: Tuple[Any, ...] = (
            self.states[index],
            self.contexts[index],
            self.actions[index],
            self.rewards[index],
            self.dones[index],
            self.rtg[index],
            self.timesteps[index],
            self.masks[index],
            self.prompt_states[index],
            self.prompt_actions[index],
            self.prompt_rewards[index],
            self.prompt_rtg[index],
            self.prompt_timesteps[index],
            self.prompt_masks[index],
            instruction,
        )
        if self.use_vision:
            out = out + (None,)
        return out


class SubsampledICLTrajectoryDataset(ICLTrajectoryDatasetBase):
    """
    In-context prompt = fixed-length segments: prompt_length steps per trajectory (random start),
    concatenated and trimmed/padded to total_prompt_len.
    """

    def __init__(
        self,
        trajectories: List[Dict[str, np.ndarray]],
        horizon: int,
        max_episode_steps: int,
        return_scale: float,
        device: torch.device,
        prompt_trajectories_per_task: Optional[List[List[Dict[str, np.ndarray]]]] = None,
        context_dim: int = 16,
        state_dim: int = 27,
        act_dim: int = 8,
        prompt_length: int = 5,
        scale: float = 500.0,
        total_epi_per_task: int = 100,
        num_context_trajectories: int = 1,
        context_sort_ascending: bool = True,
        context_sampling: str = "random",
        max_total_prompt_length: Optional[int] = None,
        trajectory_contexts: Optional[Dict[int, np.ndarray]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            trajectories=trajectories,
            horizon=horizon,
            max_episode_steps=max_episode_steps,
            return_scale=return_scale,
            device=device,
            prompt_trajectories_per_task=prompt_trajectories_per_task,
            context_dim=context_dim,
            state_dim=state_dim,
            act_dim=act_dim,
            scale=scale,
            total_epi_per_task=total_epi_per_task,
            num_context_trajectories=num_context_trajectories,
            context_sort_ascending=context_sort_ascending,
            context_sampling=context_sampling,
            trajectory_contexts=trajectory_contexts,
            **kwargs,
        )
        self.prompt_length = prompt_length
        self.total_prompt_len = min(
            num_context_trajectories * prompt_length,
            max_total_prompt_length or (num_context_trajectories * prompt_length),
        )
        if not self._lazy:
            log.info(
                "Building segments (SubsampledICL, total_prompt_len={}; may take a minute)...",
                self.total_prompt_len,
            )
            self._parse_segments()
            log.info("Built {} segments.", len(self.states))
        else:
            log.info(
                "Lazy SubsampledICL: max_training_examples={}, total_prompt_len={}",
                self._max_examples,
                self.total_prompt_len,
            )

    def _segment_from_traj(
        self,
        traj: Dict[str, np.ndarray],
        state_mean: np.ndarray,
        state_std: np.ndarray,
    ) -> PromptArrays:
        """One segment of length prompt_length from a trajectory (random start)."""
        L = self.prompt_length
        T = traj["rewards"].shape[0]
        p_start = 0
        if T >= L:
            p_start = random.randint(0, max(0, T - L))
        plen = min(L, T - p_start)
        ps = (traj["observations"][p_start : p_start + plen] - state_mean) / state_std
        pa = traj["actions"][p_start : p_start + plen]
        pr = traj["rewards"][p_start : p_start + plen].reshape(-1, 1)
        pts = np.arange(p_start, p_start + plen, dtype=np.float32)
        prtg = discount_cumsum(traj["rewards"][p_start:], gamma=1.0)[: plen + 1].reshape(-1, 1)
        if prtg.shape[0] <= plen:
            prtg = np.concatenate([prtg, np.zeros((1, 1))], axis=0)
        pad = L - plen
        ps = np.concatenate([np.zeros((pad, ps.shape[1])), ps], axis=0)
        pa = np.concatenate([np.ones((pad, pa.shape[1])) * -10.0, pa], axis=0)
        pr = np.concatenate([np.zeros((pad, 1)), pr], axis=0)
        prtg = np.concatenate([np.zeros((pad, 1)), prtg], axis=0) / self.scale
        pts = np.concatenate([np.zeros(pad), pts], axis=0)
        pm = np.concatenate([np.zeros(pad), np.ones(plen)], axis=0)
        return ps, pa, pr, prtg, pts, pm

    def _build_prompt(
        self,
        chosen: List[Dict[str, np.ndarray]],
        state_mean: np.ndarray,
        state_std: np.ndarray,
        total_plen: int,
    ) -> PromptArrays:
        if not chosen:
            z = np.zeros((0, self.state_dim), dtype=np.float32)
            az = np.ones((0, self.act_dim), dtype=np.float32) * (-10.0)
            rz = np.zeros((0, 1), dtype=np.float32)
            rtz = np.zeros((0, 1), dtype=np.float32)
            tz = np.zeros(0, dtype=np.float32)
            mz = np.zeros(0, dtype=np.float32)
            return _pad_or_trim_prompt(
                z, az, rz, rtz, tz, mz, total_plen, self.state_dim, self.act_dim, take_last=False
            )
        segs_ps, segs_pa, segs_pr, segs_prtg, segs_pts, segs_pm = [], [], [], [], [], []
        for traj in chosen:
            s, a, r, rtg_, ts, m = self._segment_from_traj(traj, state_mean, state_std)
            segs_ps.append(s)
            segs_pa.append(a)
            segs_pr.append(r)
            segs_prtg.append(rtg_)
            segs_pts.append(ts)
            segs_pm.append(m)
        ps = np.concatenate(segs_ps, axis=0)
        pa = np.concatenate(segs_pa, axis=0)
        pr = np.concatenate(segs_pr, axis=0)
        prtg = np.concatenate(segs_prtg, axis=0)
        pts = np.concatenate(segs_pts, axis=0)
        pm = np.concatenate(segs_pm, axis=0)
        return _pad_or_trim_prompt(
            ps, pa, pr, prtg, pts, pm, total_plen, self.state_dim, self.act_dim, take_last=False
        )


class FullTrajectoryICLTrajectoryDataset(ICLTrajectoryDatasetBase):
    """
    In-context prompt = full trajectory(ies). Each trajectory is capped to
    max_prompt_trajectory_length steps (last N steps); concatenated sequence
    is then capped to max_total_prompt_length. ICRT-style.
    """

    def __init__(
        self,
        trajectories: List[Dict[str, np.ndarray]],
        horizon: int,
        max_episode_steps: int,
        return_scale: float,
        device: torch.device,
        prompt_trajectories_per_task: Optional[List[List[Dict[str, np.ndarray]]]] = None,
        context_dim: int = 16,
        state_dim: int = 27,
        act_dim: int = 8,
        scale: float = 500.0,
        total_epi_per_task: int = 100,
        num_context_trajectories: int = 1,
        context_sort_ascending: bool = True,
        context_sampling: str = "random",
        max_total_prompt_length: Optional[int] = None,
        max_prompt_trajectory_length: Optional[int] = None,
        trajectory_contexts: Optional[Dict[int, np.ndarray]] = None,
        **kwargs: Any,
    ):
        if not max_total_prompt_length:
            raise ValueError("FullTrajectoryICLTrajectoryDataset requires max_total_prompt_length")
        super().__init__(
            trajectories=trajectories,
            horizon=horizon,
            max_episode_steps=max_episode_steps,
            return_scale=return_scale,
            device=device,
            prompt_trajectories_per_task=prompt_trajectories_per_task,
            context_dim=context_dim,
            state_dim=state_dim,
            act_dim=act_dim,
            scale=scale,
            total_epi_per_task=total_epi_per_task,
            num_context_trajectories=num_context_trajectories,
            context_sort_ascending=context_sort_ascending,
            context_sampling=context_sampling,
            trajectory_contexts=trajectory_contexts,
            **kwargs,
        )
        self.total_prompt_len = max_total_prompt_length
        self.max_prompt_trajectory_length = max_prompt_trajectory_length
        if not self._lazy:
            log.info(
                "Building segments (FullTrajectoryICL, total_prompt_len={}; may take a minute)...",
                self.total_prompt_len,
            )
            self._parse_segments()
            log.info("Built {} segments.", len(self.states))
        else:
            log.info(
                "Lazy FullTrajectoryICL: max_training_examples={}, total_prompt_len={}",
                self._max_examples,
                self.total_prompt_len,
            )

    def _prompt_from_full_trajectory(
        self,
        traj: Dict[str, np.ndarray],
        state_mean: np.ndarray,
        state_std: np.ndarray,
    ) -> PromptArrays:
        """Full trajectory arrays with optional per-trajectory subsampling cap."""
        T = traj["rewards"].shape[0]
        if T == 0:
            raise ValueError("Empty trajectory in _prompt_from_full_trajectory")
        idx = _subsample_indices(T, self.max_prompt_trajectory_length, self.context_subsample_strategy)
        obs = np.asarray(traj["observations"], dtype=np.float32)
        act = np.asarray(traj["actions"], dtype=np.float32)
        rew = np.asarray(traj["rewards"], dtype=np.float32)
        full_rtg = discount_cumsum(rew, gamma=1.0).reshape(-1, 1)
        ps = (obs[idx] - state_mean) / state_std
        pa = act[idx]
        pr = rew[idx].reshape(-1, 1)
        prtg = full_rtg[idx] / self.scale
        pts = idx.astype(np.float32)
        T = idx.shape[0]
        pm = np.ones(T, dtype=np.float32)
        return ps, pa, pr, prtg, pts, pm

    def _build_prompt(
        self,
        chosen: List[Dict[str, np.ndarray]],
        state_mean: np.ndarray,
        state_std: np.ndarray,
        total_plen: int,
    ) -> PromptArrays:
        if not chosen:
            z = np.zeros((0, self.state_dim), dtype=np.float32)
            az = np.ones((0, self.act_dim), dtype=np.float32) * (-10.0)
            rz = np.zeros((0, 1), dtype=np.float32)
            rtz = np.zeros((0, 1), dtype=np.float32)
            tz = np.zeros(0, dtype=np.float32)
            mz = np.zeros(0, dtype=np.float32)
            return _pad_or_trim_prompt(
                z, az, rz, rtz, tz, mz, total_plen, self.state_dim, self.act_dim, take_last=True
            )
        segs_ps, segs_pa, segs_pr, segs_prtg, segs_pts, segs_pm = [], [], [], [], [], []
        for traj in chosen:
            s, a, r, rtg_, ts, m = self._prompt_from_full_trajectory(traj, state_mean, state_std)
            segs_ps.append(s)
            segs_pa.append(a)
            segs_pr.append(r)
            segs_prtg.append(rtg_)
            segs_pts.append(ts)
            segs_pm.append(m)
        ps = np.concatenate(segs_ps, axis=0)
        pa = np.concatenate(segs_pa, axis=0)
        pr = np.concatenate(segs_pr, axis=0)
        prtg = np.concatenate(segs_prtg, axis=0)
        pts = np.concatenate(segs_pts, axis=0)
        pm = np.concatenate(segs_pm, axis=0)
        return _pad_or_trim_prompt(
            ps, pa, pr, prtg, pts, pm, total_plen, self.state_dim, self.act_dim, take_last=True
        )


def _pad_to_length(
    t: torch.Tensor, target_len: int, dim: int = 0, pad_value: float = 0.0
) -> torch.Tensor:
    """Pad tensor along dim to target_len (at the end)."""
    if t.shape[dim] >= target_len:
        return t
    pad_size = target_len - t.shape[dim]
    pad_shape = list(t.shape)
    pad_shape[dim] = pad_size
    pad = torch.full(pad_shape, pad_value, dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=dim)


def _pad_to_length_left(
    t: torch.Tensor, target_len: int, dim: int = 0, pad_value: float = 0.0
) -> torch.Tensor:
    """Pad tensor along dim to target_len at the beginning. Used for prompt so valid content is right-aligned (contiguous with query) for causal attention."""
    if t.shape[dim] >= target_len:
        return t
    pad_size = target_len - t.shape[dim]
    pad_shape = list(t.shape)
    pad_shape[dim] = pad_size
    pad = torch.full(pad_shape, pad_value, dtype=t.dtype, device=t.device)
    return torch.cat([pad, t], dim=dim)


def collate_icl_batch(batch: List[Tuple[Any, ...]]) -> Tuple[Any, ...]:
    """
    Collate a list of ICL trajectory samples into a batch. Pads tensors with varying
    first dimension (e.g. prompt length) to the max in the batch so they can be stacked.
    Mask tensors (masks, prompt_m) are padded with 0 so padded positions are ignored.
    If samples have 16 elements, the 16th is images (list of view tensors); collate pads and stacks per view.
    """
    if not batch:
        return tuple()
    n = len(batch)
    num_elems = len(batch[0])
    out: List[Any] = []
    for idx in range(15):
        items = [sample[idx] for sample in batch]
        if idx == 14:
            out.append(items)
            continue
        tensors = [t for t in items if isinstance(t, torch.Tensor)]
        if not tensors:
            out.append(items)
            continue
        lengths = [t.shape[0] for t in tensors]
        max_len = max(lengths)
        if len(set(lengths)) == 1:
            out.append(torch.stack(tensors, dim=0))
            continue
        pad_val = 0.0
        if idx in (5, 7, 13):
            pad_val = 0.0
        elif idx in (8, 9, 10, 11, 12):
            pad_val = 0.0
        # Prompt tensors (8–13): left-pad so valid content is right-aligned with query (correct for causal attention)
        if idx in (8, 9, 10, 11, 12, 13):
            padded = [_pad_to_length_left(t, max_len, dim=0, pad_value=pad_val) for t in tensors]
        else:
            padded = [_pad_to_length(t, max_len, dim=0, pad_value=pad_val) for t in tensors]
        out.append(torch.stack(padded, dim=0))
    if num_elems > 15:
        elem_15 = batch[0][15]
        is_precomputed = any(
            sample[15] is not None
            and isinstance(sample[15], torch.Tensor)
            and sample[15].dim() in (2, 3)
            for sample in batch
        )
        if not is_precomputed and (
            elem_15 is None or not isinstance(elem_15, torch.Tensor) or elem_15.dim() not in (2, 3)
        ):
            elem_15 = None
        if elem_15 is None and not is_precomputed:
            out.append(None)
        elif is_precomputed:
            # Precomputed embeddings: each sample (1, T, D) or None; pad T and stack to (B, T_max, D)
            tensors = [sample[15] for sample in batch]
            non_none = [t for t in tensors if t is not None]
            if not non_none:
                out.append(None)
            else:
                # Validate embedding tensor shapes early to avoid hard-to-debug CUDA asserts.
                # Expected loader output: (1, T, D) where 1 is a dummy view/batch dim.
                d_dims = set()
                dtype_set = set()
                devices = set()
                for t in non_none:
                    if not isinstance(t, torch.Tensor):
                        raise TypeError(
                            f"Expected torch.Tensor for precomputed embeddings, got {type(t)}"
                        )
                    if t.dim() == 2:
                        # Support legacy (T, D) by unsqueezing to (1, T, D)
                        raise ValueError(
                            "Found precomputed embeddings with shape (T, D) (dim=2). "
                            "Expected shape (1, T, D). Please regenerate embeddings.npz."
                        )
                    if t.dim() != 3:
                        raise ValueError(
                            f"Expected precomputed embeddings dim=3 (1, T, D), got dim={t.dim()} with shape={tuple(t.shape)}"
                        )
                    if t.shape[0] != 1:
                        raise ValueError(
                            f"Expected precomputed embeddings shape[0]==1 (1, T, D), got shape={tuple(t.shape)}"
                        )
                    if t.shape[1] <= 0 or t.shape[2] <= 0:
                        raise ValueError(
                            f"Expected precomputed embeddings with positive (T, D); got shape={tuple(t.shape)}"
                        )
                    d_dims.add(int(t.shape[2]))
                    dtype_set.add(t.dtype)
                    devices.add(str(t.device))
                if len(d_dims) != 1:
                    raise ValueError(f"Inconsistent embedding D across batch: {sorted(d_dims)}")
                if len(dtype_set) != 1:
                    raise ValueError(
                        f"Inconsistent embedding dtype across batch: {sorted([str(x) for x in dtype_set])}"
                    )
                if len(devices) != 1:
                    raise ValueError(
                        f"Inconsistent embedding device across batch: {sorted(devices)}"
                    )

                t_len = max(int(t.shape[1]) for t in non_none)
                d_dim = int(next(iter(d_dims)))
                device = non_none[0].device
                dtype = non_none[0].dtype
                padded = []
                for t in tensors:
                    if t is None:
                        padded.append(
                            torch.zeros(1, t_len, d_dim, device=device, dtype=dtype).squeeze(0)
                        )
                    else:
                        p = t.squeeze(0)
                        if p.dim() != 2:
                            raise ValueError(
                                f"Expected squeezed embeddings to be (T, D) with dim=2, got shape={tuple(p.shape)}"
                            )
                        if p.shape[1] != d_dim:
                            raise ValueError(
                                f"Embedding D mismatch: expected {d_dim}, got {p.shape[1]} (shape={tuple(p.shape)})"
                            )
                        if p.shape[0] < t_len:
                            p = _pad_to_length(p, t_len, dim=0, pad_value=0.0)
                        padded.append(p)
                out.append(torch.stack(padded, dim=0))
        else:
            images_list = [sample[15] for sample in batch]
            if not all(
                im is not None and isinstance(im, list) and len(im) > 0 for im in images_list
            ):
                out.append(None)
            else:
                num_views = len(images_list[0])
                view_batches = []
                for v in range(num_views):
                    tensors_v = [im[v] for im in images_list]
                    t_len = max(t.shape[1] for t in tensors_v if t.dim() >= 2)
                    padded_v = []
                    for t in tensors_v:
                        if t.dim() == 5 and t.shape[1] < t_len:
                            pad_size = t_len - t.shape[1]
                            t = torch.cat(
                                [
                                    t,
                                    torch.zeros(
                                        t.shape[0],
                                        pad_size,
                                        *t.shape[2:],
                                        device=t.device,
                                        dtype=t.dtype,
                                    ),
                                ],
                                dim=1,
                            )
                        padded_v.append(t)
                    view_batches.append(torch.cat(padded_v, dim=0))
                out.append(view_batches if view_batches else None)
    if num_elems > 16:
        out.append([sample[16] for sample in batch])
    return tuple(out)


def get_icl_trajectory_dataset(
    context_style: str = "subsampled", **kwargs: Any
) -> ICLTrajectoryDatasetBase:
    """Factory: return SubsampledICLTrajectoryDataset or FullTrajectoryICLTrajectoryDataset by context_style."""
    if context_style == "full_trajectory":
        return FullTrajectoryICLTrajectoryDataset(**kwargs)
    return SubsampledICLTrajectoryDataset(**kwargs)


def ICLTrajectoryDataset(**kwargs: Any) -> ICLTrajectoryDatasetBase:
    """Backward-compatible alias: dispatches to Subsampled or FullTrajectory based on context_style."""
    context_style = kwargs.pop("context_style", "subsampled")
    return get_icl_trajectory_dataset(context_style=context_style, **kwargs)
