#!/usr/bin/env python3
"""Download PPO state ``walltime_efficient`` hyperparameters from W&B (stonet2000/ManiSkill).

Filters match the public report
https://wandb.ai/stonet2000/ManiSkill/reports/PPO-Results--VmlldzoxMDQzNDMzOA
(group PPO, run names ``*-state-*-walltime_efficient``).

Writes one JSON per ``env_id`` under ``scripts/maniskill/ppo_wandb_repro/configs/``.
Requires: ``pip install wandb`` (or ``uv run`` from repo root).

Example::

  uv run python scripts/maniskill/fetch_ppo_wandb_configs.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Keys stored on each W&B run that map 1:1 to ``Args`` in ``ppo_train_icldata.py``.
_PPO_ARG_KEYS = frozenset(
    {
        "env_id",
        "total_timesteps",
        "learning_rate",
        "num_envs",
        "num_eval_envs",
        "num_steps",
        "num_eval_steps",
        "partial_reset",
        "eval_partial_reset",
        "reconfiguration_freq",
        "eval_reconfiguration_freq",
        "control_mode",
        "anneal_lr",
        "gamma",
        "gae_lambda",
        "num_minibatches",
        "update_epochs",
        "norm_adv",
        "clip_coef",
        "clip_vloss",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
        "target_kl",
        "reward_scale",
        "eval_freq",
        "save_train_video_freq",
        "finite_horizon_gae",
        "torch_deterministic",
        "cuda",
        "capture_video",
        "save_model",
        "evaluate",
    }
)


def _map_sim_backend(wb: str) -> str:
    if wb == "gpu":
        return "physx_cuda"
    return wb


def _extract_ppo_args(cfg: dict) -> dict:
    out = {k: cfg[k] for k in _PPO_ARG_KEYS if k in cfg}
    ec = cfg.get("env_cfg") or {}
    if "reward_mode" in ec:
        out["reward_mode"] = ec["reward_mode"]
    sb = ec.get("sim_backend") or cfg.get("sim_backend")
    if sb:
        out["sim_backend"] = _map_sim_backend(sb)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entity-project",
        default="stonet2000/ManiSkill",
        help="W&B entity/project path",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "ppo_wandb_repro" / "configs",
    )
    args_ns = parser.parse_args()

    import wandb

    api = wandb.Api(timeout=300)
    runs = api.runs(args_ns.entity_project)
    grouped: dict[str, list[tuple[str, int, dict]]] = defaultdict(list)
    for r in runs:
        if r.group != "PPO":
            continue
        if "-state-" not in r.name or "walltime_efficient" not in r.name:
            continue
        full = api.run(f"{args_ns.entity_project}/{r.id}")
        cfg = dict(full.config)
        env_id = cfg.get("env_id")
        if not env_id:
            continue
        seed = int(cfg.get("seed", -1))
        grouped[env_id].append((full.id, seed, cfg))

    args_ns.out_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]
    manifest = {
        "source": args_ns.entity_project,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "filter": "group=PPO, name contains -state- and walltime_efficient",
        "environments": {},
    }

    for env_id in sorted(grouped.keys()):
        rows = grouped[env_id]
        rows.sort(key=lambda x: x[1])
        seeds = [s for _, s, _ in rows if s >= 0]
        ref_id, _, ref_cfg = rows[0]
        ppo_args = _extract_ppo_args(ref_cfg)
        ppo_args["env_id"] = env_id
        if "reward_mode" not in ppo_args:
            ppo_args["reward_mode"] = "normalized_dense"

        safe = env_id.replace("/", "_")
        payload = {
            "env_id": env_id,
            "seeds": seeds,
            "wandb_reference_run_id": ref_id,
            "wandb_reference_run": f"{args_ns.entity_project}/runs/{ref_id}",
            "ppo_args": ppo_args,
        }
        out_path = args_ns.out_dir / f"{safe}.json"
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        manifest["environments"][env_id] = {
            "config_file": str(out_path.relative_to(repo_root)),
            "seeds": seeds,
            "num_seeds": len(seeds),
            "reference_run": payload["wandb_reference_run"],
        }

    manifest_path = args_ns.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(grouped)} env configs -> {args_ns.out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
