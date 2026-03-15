"""
In-context trajectory dataset: sample N context trajectories per segment (same task),
sort by increasing return, concatenate for training. At inference, previous rollouts
sorted ascending for zero-shot adaptation.
"""
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any, Dict, List, Optional, Tuple

from src.data.trajectories import discount_cumsum, sample_context_trajectories


class ICLTrajectoryDataset(Dataset):
    """
    Dataset of (state, context, action, reward, done, rtg, timesteps, mask) segments,
    with prompt = N in-context trajectories from the same task, sorted by ascending
    return (low→high) so the model sees increasing progress.
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
    ):
        self.trajectories = trajectories
        self.horizon = horizon
        self.max_episode_steps = max_episode_steps
        self.return_scale = return_scale
        self.device = device
        self.prompt_trajectories_per_task = prompt_trajectories_per_task or []
        self.context_dim = context_dim
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.prompt_length = prompt_length
        self.scale = scale
        self.total_epi_per_task = total_epi_per_task
        self.num_context_trajectories = num_context_trajectories
        self.context_sort_ascending = context_sort_ascending
        self.context_sampling = context_sampling
        self.max_total_prompt_length = max_total_prompt_length
        self.trajectory_contexts = trajectory_contexts or {}
        self.total_prompt_len = min(
            num_context_trajectories * prompt_length,
            max_total_prompt_length or (num_context_trajectories * prompt_length),
        )

        states = np.concatenate([t["observations"] for t in trajectories], axis=0)
        self.state_mean = np.mean(states, axis=0)
        self.state_std = np.std(states, axis=0) + 1e-6
        returns = np.array([t["rewards"].sum() for t in trajectories])
        self.return_min = float(returns.min())
        self.return_max = float(returns.max())
        self.return_avg = float(returns.mean())

        self._parse_segments()

    def _segment_from_traj(
        self,
        promt_traj: Dict[str, np.ndarray],
        state_mean: np.ndarray,
        state_std: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract one segment of length prompt_length from a trajectory (random start)."""
        L = self.prompt_length
        T = promt_traj["rewards"].shape[0]
        p_start = 0
        if T >= L:
            p_start = random.randint(0, max(0, T - L))
        plen = min(L, T - p_start)
        ps = (promt_traj["observations"][p_start : p_start + plen] - state_mean) / state_std
        pa = promt_traj["actions"][p_start : p_start + plen]
        pr = promt_traj["rewards"][p_start : p_start + plen].reshape(-1, 1)
        pts = np.arange(p_start, p_start + plen)
        prtg = discount_cumsum(promt_traj["rewards"][p_start:], gamma=1.0)[: plen + 1].reshape(-1, 1)
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

    def _parse_segments(self) -> None:
        states, contexts, actions, rewards, dones, rtg, timesteps, masks = [], [], [], [], [], [], [], []
        prompt_s, prompt_a, prompt_r, prompt_rtg, prompt_ts, prompt_m = [], [], [], [], [], []
        max_ep_len = self.max_episode_steps
        state_mean, state_std = self.state_mean, self.state_std
        total_plen = self.total_prompt_len

        for num, traj in enumerate(self.trajectories):
            task_id = num // self.total_epi_per_task if self.total_epi_per_task else 0
            prompt_list = self.prompt_trajectories_per_task[task_id] if task_id < len(self.prompt_trajectories_per_task) else []

            traj_contexts = self.trajectory_contexts.get(num)
            if traj_contexts is None:
                traj_contexts = np.zeros((len(traj["observations"]), self.context_dim), dtype=np.float32)

            for si in range(traj["rewards"].shape[0] - 1):
                # Sample N context trajectories (same task), sort by ascending return, concatenate segments
                if prompt_list and self.num_context_trajectories >= 1:
                    chosen = sample_context_trajectories(
                        prompt_list,
                        self.num_context_trajectories,
                        ascending=self.context_sort_ascending,
                        sampling=self.context_sampling,
                    )
                    segs_ps, segs_pa, segs_pr, segs_prtg, segs_pts, segs_pm = [], [], [], [], [], []
                    for promt_traj in chosen:
                        s, a, r, rtg_, ts, m = self._segment_from_traj(promt_traj, state_mean, state_std)
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
                    if ps.shape[0] > total_plen:
                        ps, pa, pr, prtg, pts, pm = ps[:total_plen], pa[:total_plen], pr[:total_plen], prtg[:total_plen], pts[:total_plen], pm[:total_plen]
                    elif ps.shape[0] < total_plen:
                        pad_len = total_plen - ps.shape[0]
                        ps = np.concatenate([np.zeros((pad_len, ps.shape[1])), ps], axis=0)
                        pa = np.concatenate([np.ones((pad_len, pa.shape[1])) * -10.0, pa], axis=0)
                        pr = np.concatenate([np.zeros((pad_len, 1)), pr], axis=0)
                        prtg = np.concatenate([np.zeros((pad_len, 1)), prtg], axis=0)
                        pts = np.concatenate([np.zeros(pad_len), pts], axis=0)
                        pm = np.concatenate([np.zeros(pad_len), pm], axis=0)
                else:
                    promt_traj = traj
                    ps, pa, pr, prtg, pts, pm = self._segment_from_traj(promt_traj, state_mean, state_std)
                    if ps.shape[0] != total_plen:
                        pad_len = total_plen - ps.shape[0]
                        if pad_len > 0:
                            ps = np.concatenate([np.zeros((pad_len, ps.shape[1])), ps], axis=0)
                            pa = np.concatenate([np.ones((pad_len, pa.shape[1])) * -10.0, pa], axis=0)
                            pr = np.concatenate([np.zeros((pad_len, 1)), pr], axis=0)
                            prtg = np.concatenate([np.zeros((pad_len, 1)), prtg], axis=0)
                            pts = np.concatenate([np.zeros(pad_len), pts], axis=0)
                            pm = np.concatenate([np.zeros(pad_len), pm], axis=0)
                        else:
                            ps, pa, pr, prtg, pts, pm = ps[-total_plen:], pa[-total_plen:], pr[-total_plen:], prtg[-total_plen:], pts[-total_plen:], pm[-total_plen:]
                prompt_s.append(ps)
                prompt_a.append(pa)
                prompt_r.append(pr)
                prompt_rtg.append(prtg)
                prompt_ts.append(pts)
                prompt_m.append(pm)

                # Main segment
                state_seg = traj["observations"][si : si + self.horizon]
                action_seg = traj["actions"][si : si + self.horizon]
                reward_seg = traj["rewards"][si : si + self.horizon].reshape(-1, 1)
                context_seg = traj_contexts[si : si + self.horizon]
                done_seg = traj.get("terminals", traj.get("dones", np.zeros(len(reward_seg))))[si : si + self.horizon]
                tlen = state_seg.shape[0]
                state_seg = np.concatenate([np.zeros((self.horizon - tlen, state_seg.shape[1])), state_seg], axis=0)
                state_seg = (state_seg - state_mean) / state_std
                context_seg = np.concatenate([np.zeros((self.horizon - tlen, context_seg.shape[1])), context_seg], axis=0)
                action_seg = np.concatenate([np.ones((self.horizon - tlen, action_seg.shape[1])) * -10.0, action_seg], axis=0)
                reward_seg = np.concatenate([np.zeros((self.horizon - tlen, 1)), reward_seg], axis=0)
                done_seg = np.concatenate([np.ones(self.horizon - tlen) * 2, done_seg], axis=0)
                rtg_seg = discount_cumsum(traj["rewards"][si:], gamma=1.0)[: tlen + 1].reshape(-1, 1)
                if rtg_seg.shape[0] <= tlen:
                    rtg_seg = np.concatenate([rtg_seg, np.zeros((1, 1))], axis=0)
                rtg_seg = np.concatenate([np.zeros((self.horizon - tlen, 1)), rtg_seg], axis=0) / self.return_scale
                ts_seg = np.arange(si, si + tlen)
                ts_seg[ts_seg >= max_ep_len] = max_ep_len - 1
                ts_seg = np.concatenate([np.zeros(self.horizon - tlen), ts_seg], axis=0)
                mask_seg = np.concatenate([np.zeros(self.horizon - tlen), np.ones(tlen)], axis=0)

                states.append(state_seg)
                contexts.append(context_seg)
                actions.append(action_seg)
                rewards.append(reward_seg)
                dones.append(done_seg)
                rtg.append(rtg_seg)
                timesteps.append(ts_seg)
                masks.append(mask_seg)

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
        return self.states.size(0)

    def __getitem__(self, index: int) -> Tuple[Any, ...]:
        return (
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
        )
