#!/usr/bin/env python3
"""
Load a training checkpoint and run the same Gymnasium rollouts as training eval
(`run_rollouts_and_save_viz`: prompt / zero-shot modes, RTG, viz under run_dir).

Merges Hydra defaults with the config stored in the checkpoint (so new config keys
get defaults). Override data paths or eval knobs on the CLI like `src.train`.

Currently wires **HalfCheetah-v2** offline data + **HalfCheetah-v5** env (same as training).

Example:
  uv run python scripts/run_d4rl_policy_eval.py \\
    --checkpoint outputs/.../ckpts/best/checkpoint.pt \\
    --override paths.data_root=/path/to/datasets data=[base,halfcheetah]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch
from loguru import logger as log
from omegaconf import OmegaConf

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hydra import compose, initialize_config_dir

from src.config.schema import resolved_max_total_prompt_length
from src.data import get_icl_trajectory_dataset
from src.data.d4rl_loader import load_halfcheetah_trajectories, parse_halfcheetah_data_qualities
from src.data.trajectories import sample_context_trajectories
from src.engine.eval_viz import run_rollouts_and_save_viz
from src.train import (
    ENV_DIMS,
    LIBERO_SUITES,
    build_model,
    resolve_paths,
    validate_dataset_paths,
)


def _infer_run_dir(ckpt_path: str) -> Path:
    p = Path(ckpt_path).resolve()
    if "ckpts" in p.parts:
        idx = p.parts.index("ckpts")
        return Path(*p.parts[:idx]).resolve()
    return p.parent.parent


def _get_config(config_dir: str, overrides: Optional[List[str]] = None):
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name="config", overrides=overrides or [])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="D4RL-style policy eval (training-equivalent rollouts)."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint.pt (best/last/periodic)."
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs",
        help="Hydra config directory (repo root relative ok).",
    )
    parser.add_argument(
        "--override",
        action="append",
        nargs="*",
        default=[],
        metavar="KEY=VAL",
        help="Hydra overrides (same as src.train), e.g. paths.data_root=/data data=[base,halfcheetah]",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Run directory for viz/ (default: parent of ckpts/ from checkpoint path).",
    )
    parser.add_argument(
        "--step", type=int, default=0, help="Step label for viz/samples/step_XXXXXX (default 0)."
    )
    parser.add_argument(
        "--device", type=str, default=None, help="cuda:0 or cpu (default: cuda if available)."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict state_dict load (default: False, tolerates minor architecture drift).",
    )
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help=(
            "torch.load(weights_only=True). Default is False: full training checkpoints include "
            "NumPy arrays (state_mean/std) and config; PyTorch 2.6+ rejects those under weights_only."
        ),
    )
    args = parser.parse_args()

    overrides: List[str] = []
    for ov in args.override:
        overrides.extend(ov if isinstance(ov, list) else [ov])

    config_dir = os.path.abspath(args.config_dir)
    if not os.path.isdir(config_dir):
        config_dir = str(_REPO / "configs")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    ckpt: dict[str, Any] = torch.load(
        str(ckpt_path), map_location="cpu", weights_only=bool(args.weights_only)
    )
    if "model" not in ckpt:
        raise SystemExit("Checkpoint missing 'model' state_dict.")

    cfg = _get_config(config_dir, overrides)
    saved = ckpt.get("config")
    if saved:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(saved))
    OmegaConf.resolve(cfg)

    sys_cfg = cfg.system
    data_cfg = cfg.data
    device = torch.device(args.device or (sys_cfg.device if torch.cuda.is_available() else "cpu"))
    env_name = str(data_cfg.env_name)

    if env_name != "HalfCheetah-v2":
        raise SystemExit(
            f"scripts/run_d4rl_policy_eval.py only supports env HalfCheetah-v2; got {env_name!r}. "
            "For other envs, extend this script or use training-time eval."
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

    sm_ckpt = ckpt.get("state_mean")
    std_ckpt = ckpt.get("state_std")
    if sm_ckpt is not None and std_ckpt is not None:
        state_mean = np.asarray(sm_ckpt, dtype=np.float32)
        state_std = np.asarray(std_ckpt, dtype=np.float32)
        log.info("Using state_mean/state_std from checkpoint.")
    else:
        state_mean = dataset.state_mean
        state_std = dataset.state_std
        log.info("Using state_mean/state_std from dataset (checkpoint had no stats).")

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

    run_dir = Path(args.output_dir) if args.output_dir else _infer_run_dir(str(ckpt_path))
    run_dir.mkdir(parents=True, exist_ok=True)

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

    num_rollouts = int(exp.num_eval_rollouts)
    log.info(
        "Running eval: mode={} num_rollouts={} eval_num_trials={} max_episode_steps={} rtg_scale={} run_dir={}",
        eval_mode,
        num_rollouts,
        int(exp.eval_num_trials),
        data_cfg.max_episode_steps,
        float(data_cfg.rtg_scale),
        run_dir,
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
            save_video=bool(exp.save_eval_video),
            eval_context_mode=eval_mode,
            prompt_trajectories=prompt_trajectories,
            eval_num_trials=int(exp.eval_num_trials),
            eval_context_k=eval_k,
            eval_reward_source=str(exp.eval_reward_source),
            eval_reward_model=exp.eval_reward_model,
            total_prompt_len=dataset.total_prompt_len,
            max_prompt_trajectory_length=dataset.max_prompt_trajectory_length,
            context_subsample_strategy=dataset.context_subsample_strategy,
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
