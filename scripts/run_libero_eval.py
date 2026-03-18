#!/usr/bin/env python3
"""
Offline evaluation for LIBERO-Cosmos: load checkpoint, run forward on held-out episodes,
report action MSE (and optional success-rate proxy) per suite (in-distribution).

Requires the new dataset format: manifest.parquet + episodes/ (from convert_libero_hdf5_to_dataset.py).
Val episodes = last 10% of episodes from the manifest (by episode_index); use --max-val-episodes to cap.

Usage:
  python scripts/run_libero_eval.py --ckpt path/to/checkpoint.pt --data-dir datasets --output-dir outputs/libero_eval
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_config_from_run_dir(run_dir: Path):
    """Load Hydra config from run_dir/.hydra/config.yaml if present."""
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.is_file():
        return None
    from omegaconf import OmegaConf

    return OmegaConf.load(cfg_path)


def main():
    parser = argparse.ArgumentParser(
        description="LIBERO-Cosmos offline eval (action MSE per suite)"
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint.pt")
    parser.add_argument(
        "--data-dir", type=str, default="datasets", help="Data root (default: datasets)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Write metrics JSON here (default: same as run_dir/eval)",
    )
    parser.add_argument(
        "--max-val-episodes",
        type=int,
        default=500,
        help="Cap number of val episodes (default: 500)",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt).resolve()
    if not ckpt_path.is_file():
        raise SystemExit("Checkpoint not found: " + str(ckpt_path))

    run_dir = ckpt_path.parent.parent if "ckpts" in ckpt_path.parts else ckpt_path.parent
    cfg = load_config_from_run_dir(run_dir)
    if cfg is None:
        cfg = load_config_from_run_dir(Path.cwd())
    if cfg is None:
        raise SystemExit(
            "Could not load config; run from project root or pass run_dir with .hydra/config.yaml"
        )

    data_dir = args.data_dir
    from src.data.libero_dataset import _has_new_format, load_libero_episodes_for_eval

    root = Path(data_dir).resolve() / "LIBERO-Cosmos-Policy"
    if not _has_new_format(root):
        raise SystemExit(
            "LIBERO eval requires manifest.parquet + episodes/. "
            "Run: python scripts/convert_libero_hdf5_to_dataset.py --input-dir <LIBERO-Cosmos-Policy>"
        )
    trajectories = load_libero_episodes_for_eval(
        data_dir,
        last_n_fraction=0.1,
        max_episodes=args.max_val_episodes,
    )
    for t in trajectories:
        t["_suite"] = "unknown"
    if not trajectories:
        print("No val episodes to evaluate.")
        return

    state_dim, act_dim = 9, 7
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Build model to match training (use saved config from run_dir if present)
    from src.train import get_config, build_model

    config_dir = Path(__file__).resolve().parent.parent / "configs"
    if (run_dir / ".hydra" / "config.yaml").is_file():
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    else:
        cfg = get_config(str(config_dir), overrides=["data=[base,libero_cosmos]"])
    model = build_model(cfg, state_dim, act_dim).to(device)
    from src.engine.checkpointing import load_checkpoint

    load_checkpoint(str(ckpt_path), model, device=device, weights_only=True)
    model.eval()

    # Normalization: use state mean/std from dataset (we don't have it in checkpoint; use 0/1 for now)
    state_mean = np.zeros(state_dim, dtype=np.float32)
    state_std = np.ones(state_dim, dtype=np.float32)

    horizon = cfg.data.horizon
    max_ep_len = cfg.data.max_episode_steps
    return_scale = cfg.data.return_scale

    results_by_suite = {}
    all_mse = []
    all_success = []

    prompt_len = 8
    for traj in trajectories:
        suite = traj.pop("_suite", "unknown")
        obs = traj["observations"]
        actions_gt = traj["actions"]
        T = obs.shape[0]
        if T < 2:
            continue
        obs_norm = (obs - state_mean) / state_std
        mse_list = []
        with torch.no_grad():
            for start in range(0, T - 1, horizon):
                end = min(start + horizon + 1, T)
                seg_len = end - start - 1
                if seg_len < 1:
                    continue
                states_t = (
                    torch.from_numpy(obs_norm[start : end - 1]).float().unsqueeze(0).to(device)
                )
                actions_gt_t = (
                    torch.from_numpy(actions_gt[start : end - 1]).float().unsqueeze(0).to(device)
                )
                rewards = traj["rewards"][start : end - 1]
                rtg = np.array(
                    [rewards[i:].sum() / return_scale for i in range(len(rewards))],
                    dtype=np.float32,
                ).reshape(-1, 1)
                rtg_t = torch.from_numpy(rtg).float().unsqueeze(0).to(device)
                timesteps_t = (
                    torch.arange(start, end - 1, dtype=torch.long, device=device)
                    .unsqueeze(0)
                    .clamp(max=max_ep_len - 1)
                )
                ps = torch.zeros(1, prompt_len, state_dim, device=device)
                pa = torch.ones(1, prompt_len, act_dim, device=device) * -10.0
                pr = torch.zeros(1, prompt_len, 1, device=device)
                prtg_p = torch.zeros(1, prompt_len, 1, device=device)
                pts = torch.zeros(1, prompt_len, dtype=torch.long, device=device)
                pm = torch.zeros(1, prompt_len, device=device)
                prompt = (ps, pa, pr, prtg_p, pts, pm)
                _, action_preds, _ = model(
                    states_t,
                    torch.zeros(1, seg_len, model.context_dim, device=device),
                    actions_gt_t,
                    torch.from_numpy(rewards).float().unsqueeze(0).unsqueeze(2).to(device),
                    rtg_t,
                    timesteps_t,
                    attention_mask=torch.ones(1, seg_len, device=device),
                    prompt=prompt,
                )
                pred = action_preds[0].cpu().numpy()
                gt = actions_gt[start : end - 1]
                mse = ((pred - gt) ** 2).mean()
                mse_list.append(float(mse))
        if mse_list:
            mse_ep = np.mean(mse_list)
            results_by_suite.setdefault(suite, {"mse": [], "success": []})
            results_by_suite[suite]["mse"].append(mse_ep)
            results_by_suite[suite]["success"].append(traj.get("success") is True)
            all_mse.append(mse_ep)
            all_success.append(traj.get("success") is True)

    # Summary
    summary = {
        "eval/action_mse_mean": float(np.mean(all_mse)) if all_mse else 0.0,
        "eval/action_mse_std": float(np.std(all_mse)) if all_mse else 0.0,
        "eval/num_episodes": len(all_mse),
        "eval/success_rate": float(np.mean(all_success)) if all_success else 0.0,
        "by_suite": {},
    }
    for suite, v in results_by_suite.items():
        summary["by_suite"][suite] = {
            "action_mse_mean": float(np.mean(v["mse"])),
            "action_mse_std": float(np.std(v["mse"])),
            "num_episodes": len(v["mse"]),
            "success_rate": float(np.mean(v["success"])),
        }

    out_dir = Path(args.output_dir) if args.output_dir else (run_dir / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "libero_eval_metrics.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print("Wrote", out_file)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
