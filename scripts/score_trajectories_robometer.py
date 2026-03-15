#!/usr/bin/env python3
"""
Offline step: compute dense rewards for robot trajectories using Robometer.
Use this once before training when you have rollout videos (or frame sequences)
and want progress/success as dense rewards instead of sparse task rewards.

Ref: https://huggingface.co/robometer/Robometer-4B

Usage:
  uv run python scripts/score_trajectories_robometer.py \\
    --input datasets/MyRobot-v0/demos/trajectories.pkl \\
    --output datasets/MyRobot-v0/demos/trajectories_robometer.pkl \\
    --task "pick up the block" \\
    [--model-path robometer/Robometer-4B]

Input trajectory format (per trajectory in the pkl list):
  - "video_path": path to .mp4 (or image folder), or
  - "frames": (T, H, W, C) numpy array, or
  - "images": list of paths to frames
  - "rewards": optional; overwritten with Robometer per-frame progress.

Output: same list of trajectories with "rewards" set to per-frame progress (0–1).
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger


def load_robometer(model_path: str = "robometer/Robometer-4B"):
    """Load Robometer model from Hugging Face. Returns a callable or None if not available."""
    try:
        # Robometer repo: github.com/robometer/robometer
        # Option 1: use their inference script / API
        from transformers import AutoModelForCausalLM, AutoProcessor
        import torch

        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        def score_video(video_path: str, task: str) -> np.ndarray:
            # Their API may take video path + task, return per-frame progress
            # Stub: see robometer repo for exact forward.
            from PIL import Image
            import torch

            # Placeholder: real impl would run model on video frames
            # and return progress shape (T,) or (T, 1)
            raise NotImplementedError(
                "Use robometer repo: uv run python scripts/example_inference_local.py "
                "--model-path robometer/Robometer-4B --video <path> --task <task>"
            )

        return score_video
    except Exception as e:
        logger.warning(
            "Robometer not available: {}. Install with: pip install robometer (or use their repo)",
            e,
        )
        return None


def score_trajectory_frames(frames: np.ndarray, task: str, model_path: str) -> np.ndarray:
    """
    Score one trajectory's frames with Robometer. Returns per-frame progress (T,) as rewards.
    frames: (T, H, W, C) or list of images.
    """
    try:
        from robometer import run_inference  # if they expose this

        return run_inference(frames, task, model_path=model_path)
    except ImportError:
        logger.error(
            "Robometer package not found. Install from https://github.com/robometer/robometer "
            "or run their example_inference_local.py per video, then merge rewards into trajectories manually."
        )
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Score trajectories with Robometer (dense rewards)"
    )
    parser.add_argument("--input", type=str, required=True, help="Input trajectories pkl")
    parser.add_argument("--output", type=str, required=True, help="Output pkl with rewards set")
    parser.add_argument("--task", type=str, required=True, help="Task description for Robometer")
    parser.add_argument("--model-path", type=str, default="robometer/Robometer-4B")
    parser.add_argument(
        "--dry-run", action="store_true", help="Only load and re-save without scoring"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        logger.error("Input not found: {}", input_path)
        raise SystemExit(1)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "rb") as f:
        trajectories = pickle.load(f)
    logger.info("Loaded {} trajectories from {}", len(trajectories), input_path)

    if args.dry_run:
        with open(output_path, "wb") as f:
            pickle.dump(trajectories, f)
        logger.info("Dry run: saved {} trajectories to {}", len(trajectories), output_path)
        return

    scorer = load_robometer(args.model_path)
    if scorer is None:
        logger.error(
            "Robometer not available. This script requires the Robometer model to score videos. "
            "For offline RL with existing rewards (e.g. D4RL), skip this step. "
            "For robotics: install robometer and ensure trajectories have 'video_path' or 'frames'."
        )
        raise SystemExit(1)

    for i, traj in enumerate(trajectories):
        video_path = traj.get("video_path")
        frames = traj.get("frames")
        if video_path:
            # Run Robometer on video file
            try:
                rewards = scorer(video_path, args.task)
                traj["rewards"] = np.asarray(rewards, dtype=np.float32).flatten()
            except NotImplementedError:
                logger.error(
                    "Per-video scoring not implemented in this stub. "
                    'Run: uv run python scripts/example_inference_local.py --model-path {} --video {} --task "{}" '
                    "and merge outputs into trajectory rewards.",
                    args.model_path,
                    video_path,
                    args.task,
                )
                raise SystemExit(1)
        elif frames is not None:
            rewards = score_trajectory_frames(frames, args.task, args.model_path)
            traj["rewards"] = np.asarray(rewards, dtype=np.float32).flatten()
        else:
            logger.warning(
                "Trajectory {} has no 'video_path' or 'frames'; keeping existing rewards", i
            )
        if (i + 1) % 10 == 0:
            logger.info("Scored {}/{}", i + 1, len(trajectories))

    with open(output_path, "wb") as f:
        pickle.dump(trajectories, f)
    logger.info(
        "Saved {} trajectories with Robometer rewards to {}", len(trajectories), output_path
    )


if __name__ == "__main__":
    main()
