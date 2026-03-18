"""
Reward model for trajectories (per-frame / per-step rewards).

Implements wrappers for:
- RoboDopamine-8B: tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview (progress -> rewards)
- Robometer-4B: robometer/Robometer-4B (progress_pred -> rewards)

Intended for *offline* dataset preprocessing (before training / indexing).
API: given per-step frames (T, H, W, 3) + task text, return reward array [T].
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


def pad_or_trunc_1d(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Pad (with last value) or truncate 1D array to target_len."""
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.shape[0] == target_len:
        return arr
    if arr.shape[0] < target_len:
        if arr.shape[0] == 0:
            return np.zeros((target_len,), dtype=np.float32)
        pad_val = float(arr[-1])
        pad = np.full((target_len - arr.shape[0],), pad_val, dtype=np.float32)
        return np.concatenate([arr, pad], axis=0)
    return arr[:target_len]


class RewardModel:
    """Compute per-step rewards for one or more trajectories."""

    name: str

    def compute_rewards_one(self, *, frames: np.ndarray, task: str) -> np.ndarray:  # [T]
        raise NotImplementedError

    def compute_rewards_batch(
        self, *, frames_list: Sequence[np.ndarray], tasks: Sequence[str], batch_size: int
    ) -> List[np.ndarray]:  # list of [T_i]
        return [self.compute_rewards_one(frames=f, task=t) for (f, t) in zip(frames_list, tasks)]


class RoboDopamine8BRewardModel(RewardModel):
    """
    Wrapper over robometer's RoboDopamine baseline.
    Per-trajectory; internally batches VLLM prompts via batch_size.
    """

    name = "robodopamine_8b"

    def __init__(
        self,
        *,
        model_path: str = "tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview",
        frame_interval: int = 1,
        batch_size: int = 4,
        eval_mode: str = "incremental",
        clip_0_1: bool = True,
    ):
        self.model_path = model_path
        self.frame_interval = frame_interval
        self.batch_size = batch_size
        self.eval_mode = eval_mode
        self.clip_0_1 = clip_0_1

        from robometer.evals.baselines.robodopamine import RoboDopamine

        self._model = RoboDopamine(
            model_path=model_path,
            frame_interval=frame_interval,
            batch_size=batch_size,
            eval_mode=eval_mode,
        )

    def compute_rewards_one(self, *, frames: np.ndarray, task: str) -> np.ndarray:
        frames_u8 = np.asarray(frames)
        if frames_u8.dtype != np.uint8:
            frames_u8 = np.clip(frames_u8, 0, 255).astype(np.uint8)
        dense = self._model.compute_progress(frames_array=frames_u8, task_description=task)
        dense = np.asarray(dense, dtype=np.float32).reshape(-1)
        if self.clip_0_1:
            dense = np.clip(dense, 0.0, 1.0)
        return dense


class Robometer4BRewardModel(RewardModel):
    """Wrapper over robometer reward model; supports batched inference across trajectories."""

    name = "robometer_4b"

    def __init__(
        self,
        *,
        model_path: str = "robometer/Robometer-4B",
        device: Optional[str] = None,
    ):
        from robometer.data.dataset_types import ProgressSample, Trajectory
        from robometer.evals.eval_server import compute_batch_outputs
        from robometer.utils.save import load_model_from_hf
        from robometer.utils.setup_utils import setup_batch_collator

        self._Trajectory = Trajectory
        self._ProgressSample = ProgressSample
        self._compute_batch_outputs = compute_batch_outputs

        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        exp_config, tokenizer, processor, reward_model = load_model_from_hf(
            model_path=model_path,
            device=self.device,
        )
        reward_model.eval()
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.processor = processor
        self.exp_config = exp_config

        self.batch_collator = setup_batch_collator(processor, tokenizer, exp_config, is_eval=True)

        loss_config = getattr(exp_config, "loss", None)
        self.is_discrete = (
            getattr(loss_config, "progress_loss_type", "l2").lower() == "discrete"
            if loss_config
            else False
        )
        self.num_bins = getattr(loss_config, "progress_discrete_bins", None) or getattr(
            getattr(exp_config, "model", None), "progress_discrete_bins", 10
        )

    def compute_rewards_batch(
        self,
        *,
        frames_list: Sequence[np.ndarray],
        tasks: Sequence[str],
        batch_size: int,
    ) -> List[np.ndarray]:
        assert len(frames_list) == len(tasks), "frames_list and tasks must align"
        n = len(frames_list)
        out: List[np.ndarray] = []
        for i in range(0, n, batch_size):
            sub_frames = frames_list[i : i + batch_size]
            sub_tasks = tasks[i : i + batch_size]
            samples = []
            for j, (frames, task) in enumerate(zip(sub_frames, sub_tasks)):
                frames_u8 = np.asarray(frames)
                if frames_u8.dtype != np.uint8:
                    frames_u8 = np.clip(frames_u8, 0, 255).astype(np.uint8)
                T = int(frames_u8.shape[0])
                traj = self._Trajectory(
                    frames=frames_u8,
                    frames_shape=tuple(frames_u8.shape),
                    task=task,
                    id=str(i + j),
                    metadata={"subsequence_length": T},
                    video_embeddings=None,
                )
                samples.append(self._ProgressSample(trajectory=traj, sample_type="progress"))

            batch = self.batch_collator(samples)
            progress_inputs = batch["progress_inputs"]
            for key, value in progress_inputs.items():
                if hasattr(value, "to"):
                    progress_inputs[key] = value.to(self.device)

            results = self._compute_batch_outputs(
                self.reward_model,
                self.tokenizer,
                progress_inputs,
                sample_type="progress",
                is_discrete_mode=self.is_discrete,
                num_bins=self.num_bins,
            )

            progress_pred = results.get("progress_pred", [])
            if not progress_pred or len(progress_pred) == 0:
                for frames in sub_frames:
                    out.append(np.zeros((int(frames.shape[0]),), dtype=np.float32))
                continue

            if isinstance(progress_pred, (list, tuple)):
                pred_list = list(progress_pred)
            else:
                pred_list = [progress_pred]

            for j, frames in enumerate(sub_frames):
                if j < len(pred_list):
                    pred = pred_list[j]
                    arr = np.asarray(pred, dtype=np.float32).reshape(-1)
                    out.append(pad_or_trunc_1d(arr, int(frames.shape[0])))
                else:
                    out.append(np.zeros((int(frames.shape[0]),), dtype=np.float32))

        return out


def build_reward_model(
    *,
    model_name: str,
    device: Optional[str] = None,
    robometer_model_path: str = "robometer/Robometer-4B",
    robodopamine_model_path: str = "tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview",
    robodopamine_frame_interval: int = 1,
    robodopamine_batch_size: int = 4,
) -> RewardModel:
    if model_name in ("robodopamine_8b", "robodopamine", "dopamine_8b"):
        return RoboDopamine8BRewardModel(
            model_path=robodopamine_model_path,
            frame_interval=robodopamine_frame_interval,
            batch_size=robodopamine_batch_size,
        )
    if model_name in ("robometer_4b", "robometer", "robometer_4b_dense", "robometer_4b_progress"):
        return Robometer4BRewardModel(
            model_path=robometer_model_path,
            device=device,
        )
    raise ValueError(f"Unknown reward model: {model_name}")
