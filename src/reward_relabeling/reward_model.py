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
        self, *, frames_list: Sequence[np.ndarray], tasks: Sequence[str]
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
    """Robometer-4B: per-frame rewards. Splits trajectory into segments of batch_size frames per model forward."""

    name = "robometer_4b"

    def __init__(
        self,
        *,
        model_path: str = "robometer/Robometer-4B",
        device: Optional[str] = None,
        batch_size: int = 32,
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

        loss_config = exp_config.loss
        self.is_discrete = loss_config.progress_loss_type.lower() == "discrete"
        self.num_bins = loss_config.progress_discrete_bins

        self.batch_size = batch_size

    def compute_rewards_one(self, *, frames: np.ndarray, task: str) -> np.ndarray:
        """Per-frame rewards. Splits into segments of batch_size frames; runs each segment through the model separately."""
        frames_u8 = np.asarray(frames, dtype=np.uint8)
        if frames_u8.dtype != np.uint8:
            frames_u8 = np.clip(frames_u8, 0, 255).astype(np.uint8)
        T = frames_u8.shape[0]
        if T == 0:
            return np.zeros(0, dtype=np.float32)

        segments = [frames_u8[s : s + self.batch_size] for s in range(0, T, self.batch_size)]
        rewards_list: List[np.ndarray] = []
        for i, seg in enumerate(segments):
            sample = self._ProgressSample(
                trajectory=self._Trajectory(
                    frames=seg,
                    frames_shape=tuple(seg.shape),
                    task=task,
                    id=str(i),
                    metadata={"subsequence_length": seg.shape[0]},
                    video_embeddings=None,
                ),
                sample_type="progress",
            )
            batch = self.batch_collator([sample])
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
            pred_list = results.get("progress_pred", []) or []
            if not isinstance(pred_list, (list, tuple)):
                pred_list = [pred_list]
            arr = (
                np.asarray(pred_list[0], dtype=np.float32).reshape(-1)
                if pred_list
                else np.zeros(0, dtype=np.float32)
            )
            rewards_list.append(pad_or_trunc_1d(arr, seg.shape[0]))
        out = np.concatenate(rewards_list, axis=0)
        return pad_or_trunc_1d(out, T)


def build_reward_model(
    *,
    model_name: str,
    device: Optional[str] = None,
    batch_size: int = 32,
    robometer_model_path: str = "robometer/Robometer-4B",
    robodopamine_model_path: str = "tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview",
    dopamine_frame_interval: int = 1,
) -> RewardModel:
    """Build a reward model. batch_size: frames per segment (Robometer) / prompts per batch (RoboDopamine)."""
    if model_name in ("robodopamine_8b", "robodopamine", "dopamine_8b"):
        return RoboDopamine8BRewardModel(
            model_path=robodopamine_model_path,
            frame_interval=dopamine_frame_interval,
            batch_size=batch_size,
        )
    if model_name in ("robometer_4b", "robometer", "robometer_4b_dense", "robometer_4b_progress"):
        return Robometer4BRewardModel(
            model_path=robometer_model_path,
            device=device,
            batch_size=batch_size,
        )
    raise ValueError(f"Unknown reward model: {model_name}")
