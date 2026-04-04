#!/usr/bin/env python3
"""Run ``ppo_train_icldata.py`` using hyperparameters saved from W&B (see ``fetch_ppo_wandb_configs.py``).

Uses the JSON files under ``scripts/maniskill/ppo_wandb_repro/configs/*.json``.

Example (ManiSkill venv, from repo root)::

  ./scripts/maniskill/run_ppo_wandb_repro.sh

Or directly::

  python scripts/maniskill/run_ppo_wandb_repro.py --config scripts/maniskill/ppo_wandb_repro/configs/PickCube-v1.json --seed 1788
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _ppo_args_to_argv(ppo: dict) -> list[str]:
    argv: list[str] = []
    for key, val in ppo.items():
        if key == "env_id":
            argv.extend(["--env-id", str(val)])
            continue
        flag = "--" + key.replace("_", "-")
        if val is None:
            continue
        if isinstance(val, bool):
            kebab = key.replace("_", "-")
            argv.append(f"--{kebab}" if val else f"--no-{kebab}")
            continue
        if isinstance(val, float):
            argv.extend([flag, format(val, "g")])
            continue
        argv.extend([flag, str(val)])
    return argv


def _default_maniskill_python(repo_root: Path) -> Path:
    venv = repo_root / ".venv-maniskill" / "bin" / "python"
    if venv.is_file():
        return venv
    return Path(sys.executable)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    ppo_script = repo_root / "scripts" / "maniskill" / "ppo_train_icldata.py"
    default_config_dir = repo_root / "scripts" / "maniskill" / "ppo_wandb_repro" / "configs"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="Single env JSON from fetch_ppo_wandb_configs.py",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=default_config_dir,
        help="Run every *.json in this directory except manifest.json (default: bundled configs)",
    )
    parser.add_argument(
        "--only-env",
        type=str,
        default=None,
        help="When using --config-dir, restrict to this env_id (e.g. PickCube-v1)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Single seed; default = all seeds in JSON")
    parser.add_argument(
        "--python",
        type=Path,
        default=None,
        help="Interpreter with mani_skill installed (default: .venv-maniskill or current)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Pass through to ppo_train_icldata.py after a bare --",
    )
    args = parser.parse_args()
    if args.extra and args.extra[0] == "--":
        args.extra = args.extra[1:]

    py = args.python or Path(os.environ.get("MANISKILL_PYTHON", _default_maniskill_python(repo_root)))

    config_files: list[Path] = []
    if args.config:
        config_files = [args.config]
    else:
        for p in sorted(args.config_dir.glob("*.json")):
            if p.name == "manifest.json":
                continue
            config_files.append(p)

    if not config_files:
        parser.error("No config files found (pass --config or use --config-dir with JSON exports).")

    for cfg_path in config_files:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        env_id = data["env_id"]
        if args.only_env and env_id != args.only_env:
            continue
        seeds = [args.seed] if args.seed is not None else list(data["seeds"])
        if not seeds:
            print(f"[skip] {cfg_path}: no seeds", file=sys.stderr)
            continue
        ppo = dict(data["ppo_args"])
        base_argv = _ppo_args_to_argv(ppo)

        for seed in seeds:
            cmd = [
                str(py),
                str(ppo_script),
                *base_argv,
                "--seed",
                str(seed),
                *args.extra,
            ]
            tag = f"{env_id} seed={seed}"
            if args.dry_run:
                print("[dry-run]", tag)
                print(subprocess.list2cmdline(cmd))
                continue
            print("==========", tag, "==========", flush=True)
            subprocess.run(cmd, cwd=str(repo_root), check=True)


if __name__ == "__main__":
    main()
