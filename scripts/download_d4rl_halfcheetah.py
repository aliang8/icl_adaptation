#!/usr/bin/env python3
"""
Download HalfCheetah mixed-expertise trajectories via Minari (no D4RL/mujoco_py).
Minari uses Gymnasium + MuJoCo 2 and avoids Cython compilation.

Saves to datasets/HalfCheetah-v2/<quality>/trajectories.pkl for training.

Run: uv run python scripts/download_d4rl_halfcheetah.py [--output-dir datasets] [--qualities medium medium_expert]
"""

import argparse
import pickle
import sys
from pathlib import Path
from collections import OrderedDict

import numpy as np

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

# Minari dataset IDs (MuJoCo HalfCheetah from D4RL, hosted on Minari)
# https://minari.farama.org/datasets/mujoco/halfcheetah/
# medium_replay may not exist in Minari; omit from --qualities if download fails.
MINARI_HALFCHEETAH = {
    "medium": "mujoco/halfcheetah/medium-v0",
    "expert": "mujoco/halfcheetah/expert-v0",
    "simple": "mujoco/halfcheetah/simple-v0",
    "medium_replay": "mujoco/halfcheetah/medium-replay-v0",
    "medium_expert": None,  # we combine medium + expert
}
ENV_PREFIX = "HalfCheetah-v2"


def episode_to_trajectory(episode) -> OrderedDict:
    """Convert Minari EpisodeData to our trajectory dict (observations, actions, rewards, next_observations, terminals)."""
    obs = np.asarray(episode.observations)
    actions = np.asarray(episode.actions)
    rewards = np.asarray(episode.rewards)
    terminations = np.asarray(episode.terminations)
    truncations = np.asarray(episode.truncations)
    # observations: T+1 (includes initial), actions/rewards: T
    T = len(rewards)
    observations = obs[:T]
    next_observations = obs[1 : T + 1]
    terminals = np.logical_or(terminations, truncations).astype(np.float32)
    return OrderedDict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=terminals,
    )


def load_minari_as_trajectories(dataset_id: str, download: bool = True) -> list:
    """Load a Minari dataset and return list of trajectory dicts."""
    import minari

    if download:
        try:
            minari.download_dataset(dataset_id)
        except Exception as e:
            logger.warning("Download {} failed (may already exist): {}", dataset_id, e)
    dataset = minari.load_dataset(dataset_id, download=download)
    trajectories = []
    for episode in dataset.iterate_episodes():
        traj = episode_to_trajectory(episode)
        if len(traj["rewards"]) >= 10:
            trajectories.append(traj)
    return trajectories


def main():
    parser = argparse.ArgumentParser(
        description="Download HalfCheetah via Minari (mixed expertise, no mujoco_py)"
    )
    parser.add_argument("--output-dir", type=str, default="datasets")
    parser.add_argument(
        "--qualities",
        nargs="+",
        default=["medium", "expert", "medium_expert"],
        help="Qualities to save: medium, expert, medium_replay, medium_expert (medium_expert = combined medium+expert)",
    )
    parser.add_argument(
        "--no-download", action="store_true", help="Only convert already-downloaded Minari data"
    )
    args = parser.parse_args()

    try:
        import minari
    except ImportError as e:
        logger.error(
            "Minari not installed. Run: uv sync --extra d4rl or pip install 'minari[all]'. "
            "Minari uses Gymnasium + MuJoCo 2 (no mujoco_py / no Cython)."
        )
        raise SystemExit(1) from e

    out_base = Path(args.output_dir) / ENV_PREFIX
    out_base.mkdir(parents=True, exist_ok=True)

    for quality in args.qualities:
        if quality == "medium_expert":
            # Mixed expertise: combine medium and expert into one dataset
            logger.info("Loading medium + expert for mixed-expertise (medium_expert)")
            trajs_medium = load_minari_as_trajectories(
                MINARI_HALFCHEETAH["medium"], download=not args.no_download
            )
            trajs_expert = load_minari_as_trajectories(
                MINARI_HALFCHEETAH["expert"], download=not args.no_download
            )
            trajectories = trajs_medium + trajs_expert
            quality_dir = "medium_expert"
        else:
            dataset_id = MINARI_HALFCHEETAH.get(quality)
            if dataset_id is None:
                logger.warning(
                    "Unknown quality '{}'; skipping. Use medium, expert, medium_replay, or medium_expert.",
                    quality,
                )
                continue
            logger.info("Loading Minari dataset: {}", dataset_id)
            trajectories = load_minari_as_trajectories(dataset_id, download=not args.no_download)
            quality_dir = quality

        if not trajectories:
            logger.warning("No trajectories for {}; skipping.", quality)
            continue

        save_dir = out_base / quality_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / "trajectories.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(trajectories, f)
        returns = [t["rewards"].sum() for t in trajectories]
        logger.info(
            "Saved {} trajectories to {} (returns min={:.1f} max={:.1f} mean={:.1f})",
            len(trajectories),
            out_path,
            min(returns),
            max(returns),
            sum(returns) / len(returns),
        )
    logger.info("Done. Datasets under {} (Minari, no mujoco_py)", out_base)


if __name__ == "__main__":
    main()
