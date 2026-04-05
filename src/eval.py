"""
Evaluation: offline rollouts + viz under ``<experiment_root>/offline_eval/``.

Run from repo root: ``uv run python -m src.eval --checkpoint ...``. Merges Hydra defaults with the
checkpoint config (like ``scripts/run_d4rl_policy_eval.py``), uses ``build_model``, and calls
``run_rollouts_and_save_viz`` for training-aligned RTG, prompts, and plots.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from loguru import logger as log
from omegaconf import OmegaConf

from src.config.schema import resolved_max_total_prompt_length
from src.data import get_icl_trajectory_dataset
from src.data.d4rl_loader import load_halfcheetah_trajectories, parse_halfcheetah_data_qualities
from src.data.trajectories import sample_context_trajectories
from src.engine.eval_viz import run_rollouts_and_save_viz
from src.engine.run_dir import infer_experiment_root_from_checkpoint
from src.train import (
    ENV_DIMS,
    LIBERO_SUITES,
    build_model,
    resolve_paths,
    validate_dataset_paths,
)


def _get_config(config_dir: str, overrides: Optional[List[str]] = None):
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name="config", overrides=overrides or [])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Offline eval: rollouts + viz under <experiment_root>/offline_eval/ (uses eval_viz)."
        )
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=3,
        help="Number of env rollouts (maps to num_eval_rollouts for this run).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=0,
        help="Step label for viz path: offline_eval/viz/samples/step_XXXXXX/",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override rollout/viz root (default: <experiment_root>/offline_eval).",
    )
    parser.add_argument(
        "--experiment-root",
        type=str,
        default=None,
        help=(
            "Experiment root for default offline_eval path "
            "(default: parent of ckpts/ inferred from --checkpoint)."
        ),
    )
    parser.add_argument("--config-dir", type=str, default="configs")
    parser.add_argument(
        "--override",
        action="append",
        nargs="*",
        default=[],
        metavar="KEY=VAL",
        help="Hydra overrides (e.g. paths.data_root=/data).",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--strict", action="store_true", help="load_state_dict(strict=True).")
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="torch.load(weights_only=True); usually fails on full training checkpoints.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Record rollout videos under offline_eval/viz/.../videos/.",
    )
    args = parser.parse_args()

    overrides: List[str] = []
    for ov in args.override:
        overrides.extend(ov if isinstance(ov, list) else [ov])

    config_dir = os.path.abspath(args.config_dir)
    if not os.path.isdir(config_dir):
        config_dir = str(_REPO / "configs")

    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    ckpt: dict[str, Any] = torch.load(
        str(ckpt_path), map_location="cpu", weights_only=bool(args.weights_only)
    )
    if "model" not in ckpt:
        raise SystemExit("Checkpoint missing 'model' state_dict.")

    cfg = _get_config(config_dir, overrides)
    if "config" in ckpt and ckpt["config"]:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(ckpt["config"]))
    OmegaConf.resolve(cfg)

    sys_cfg = cfg.system
    data_cfg = cfg.data
    device = torch.device(args.device or (sys_cfg.device if torch.cuda.is_available() else "cpu"))
    env_name = str(data_cfg.env_name)

    if env_name != "HalfCheetah-v2":
        raise SystemExit(
            f"src.eval currently supports HalfCheetah-v2 only; got {env_name!r}. "
            "Extend like scripts/run_d4rl_policy_eval.py or use Hydra overrides."
        )

    paths = resolve_paths(cfg)
    validate_dataset_paths(env_name, paths, data_cfg)

    trajectories, prompt_per_task = load_halfcheetah_trajectories(
        str(paths.data_root),
        env_name=data_cfg.env_name,
        data_quality=data_cfg.data_quality,
    )
    if not trajectories:
        qs = parse_halfcheetah_data_qualities(data_cfg.data_quality)
        expected = "\n".join(
            f"  {paths.data_root}/{data_cfg.env_name}/{q}/trajectories.pkl" for q in qs
        )
        raise SystemExit(f"No trajectories loaded. Expected one of:\n{expected}")

    state_dim, action_dim = ENV_DIMS.get(env_name, (cfg.model.state_dim, cfg.model.act_dim))
    total_plen = resolved_max_total_prompt_length(data_cfg)
    dataset = get_icl_trajectory_dataset(
        trajectories=trajectories,
        horizon=data_cfg.horizon,
        max_episode_steps=data_cfg.max_episode_steps,
        rtg_scale=float(data_cfg.rtg_scale),
        device=device,
        prompt_trajectories_per_task=prompt_per_task,
        context_dim=data_cfg.context_dim,
        state_dim=state_dim,
        act_dim=action_dim,
        prompt_length=data_cfg.prompt_length,
        total_epi_per_task=max(1, len(trajectories) // max(1, data_cfg.num_train_tasks)),
        num_context_trajectories=data_cfg.num_context_trajectories,
        randomize_num_context_trajectories=data_cfg.randomize_num_context_trajectories,
        context_sort_ascending=data_cfg.context_sort_ascending,
        context_sampling=data_cfg.context_sampling,
        max_total_prompt_length=total_plen,
        max_prompt_trajectory_length=data_cfg.max_prompt_trajectory_length,
        context_subsample_strategy=data_cfg.context_subsample_strategy,
        context_style=data_cfg.context_style,
        lazy_dataset=data_cfg.lazy_dataset,
        max_training_examples=data_cfg.max_training_examples,
        task_instructions=data_cfg.task_instructions,
        seed=data_cfg.seed,
        query_history_length=data_cfg.query_history_length,
        use_vision=data_cfg.use_vision,
        image_keys=data_cfg.image_keys or [],
    )

    sm_ckpt = ckpt["state_mean"] if "state_mean" in ckpt else None
    std_ckpt = ckpt["state_std"] if "state_std" in ckpt else None
    if sm_ckpt is not None and std_ckpt is not None:
        state_mean = np.asarray(sm_ckpt, dtype=np.float32)
        state_std = np.asarray(std_ckpt, dtype=np.float32)
        log.info("Using state_mean/state_std from checkpoint.")
    else:
        state_mean = dataset.state_mean
        state_std = dataset.state_std
        log.info("Using state_mean/state_std from dataset.")

    num_instructions = len(dataset.task_instructions) if dataset.task_instructions else None
    model = build_model(cfg, state_dim, action_dim, num_instructions=num_instructions).to(device)
    inc = model.load_state_dict(ckpt["model"], strict=bool(args.strict))
    if not args.strict:
        log.info(
            "load_state_dict strict=False | missing_keys={} unexpected_keys={}",
            len(inc.missing_keys),
            len(inc.unexpected_keys),
        )
    model.eval()

    exp_root = (
        Path(args.experiment_root).resolve()
        if args.experiment_root
        else infer_experiment_root_from_checkpoint(ckpt_path)
    )
    if args.output_dir:
        run_dir = Path(args.output_dir).resolve()
    else:
        run_dir = exp_root / "offline_eval"
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("Rollout viz directory: {}", run_dir.resolve())

    exp = cfg.experiment
    eval_mode = exp.eval_context_mode
    if eval_mode == "zero_shot_adaptation":
        if exp.eval_context_k is not None:
            eval_k = exp.eval_context_k
        else:
            ml = cfg.model.max_length
            eval_k = int(ml) if ml is not None else 20
    else:
        eval_k = exp.eval_context_k or data_cfg.num_context_trajectories

    eval_query_window = (
        int(data_cfg.query_history_length)
        if data_cfg.query_history_length is not None
        else int(data_cfg.horizon)
    )

    prompt_trajectories = None
    if eval_mode == "prompt" and dataset.trajectories:
        prompt_trajectories = sample_context_trajectories(
            dataset.trajectories,
            n=eval_k,
            ascending=True,
            sampling=data_cfg.context_sampling,
        )

    task_desc = (dataset.task_instructions or [None])[0] if dataset.task_instructions else None
    eval_render_both_views = bool(exp.eval_render_both_views) and (env_name in LIBERO_SUITES)
    minari_halfcheetah_id = None
    if "halfcheetah" in str(data_cfg.env_name).lower():
        from src.envs.minari_halfcheetah_eval import resolve_minari_halfcheetah_eval_id

        minari_halfcheetah_id = resolve_minari_halfcheetah_eval_id(
            ",".join(parse_halfcheetah_data_qualities(data_cfg.data_quality))
        )

    _env_l = str(data_cfg.env_name).lower()
    d4rl_score_ref = None
    if minari_halfcheetah_id is not None or (
        "halfcheetah" in _env_l and not _env_l.startswith("vd4rl/")
    ):
        from src.envs.d4rl_normalized_score import MUJOCO_HALFCHEETAH_D4RL_REF

        d4rl_score_ref = MUJOCO_HALFCHEETAH_D4RL_REF

    eval_target_returns_list = OmegaConf.select(cfg, "experiment.eval_target_returns", default=None)
    if eval_target_returns_list is not None:
        eval_target_returns_list = [float(x) for x in list(eval_target_returns_list)]

    num_rollouts = max(0, int(args.num_episodes))
    save_video = bool(args.save_video) or bool(exp.save_eval_video)

    log.info(
        "Offline eval: mode={} rollouts={} eval_num_trials={} max_episode_steps={} rtg_scale={}",
        eval_mode,
        num_rollouts,
        int(exp.eval_num_trials),
        data_cfg.max_episode_steps,
        float(data_cfg.rtg_scale),
    )

    with torch.inference_mode():
        metrics = run_rollouts_and_save_viz(
            model=model,
            env_name=str(data_cfg.env_name),
            state_mean=state_mean,
            state_std=state_std,
            device=device,
            run_dir=run_dir,
            step=int(args.step),
            num_rollouts=num_rollouts,
            max_episode_steps=int(data_cfg.max_episode_steps),
            rtg_scale=float(data_cfg.rtg_scale),
            save_video=save_video,
            eval_context_mode=eval_mode,
            prompt_trajectories=prompt_trajectories,
            eval_num_trials=int(exp.eval_num_trials),
            eval_context_k=eval_k,
            eval_reward_source=str(exp.eval_reward_source),
            eval_reward_model=exp.eval_reward_model,
            total_prompt_len=dataset.total_prompt_len,
            max_prompt_trajectory_length=dataset.max_prompt_trajectory_length,
            context_subsample_strategy=dataset.context_subsample_strategy,
            context_style=str(data_cfg.context_style),
            task_description=task_desc,
            logger=None,
            eval_render_both_views=eval_render_both_views,
            wandb_defer_step_commit=False,
            vd4rl_eval_pixel_hw=None,
            vd4rl_eval_obs_downsample=None,
            vd4rl_eval_seed=int(data_cfg.seed),
            eval_target_return=OmegaConf.select(cfg, "experiment.eval_target_return", default=None),
            eval_target_returns=eval_target_returns_list,
            num_context_trajectories=int(data_cfg.num_context_trajectories),
            query_window=eval_query_window,
            minari_halfcheetah_dataset_id=minari_halfcheetah_id,
            num_eval_rollout_videos=OmegaConf.select(
                cfg, "experiment.num_eval_rollout_videos", default=None
            ),
            d4rl_score_ref=d4rl_score_ref,
        )

    for k, v in sorted(metrics.items()):
        log.info("{} = {:.6g}", k, v)
    print(metrics)


if __name__ == "__main__":
    main()
