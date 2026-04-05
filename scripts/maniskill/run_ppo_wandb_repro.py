#!/usr/bin/env python3
"""Run ``ppo_train_icldata.py`` using hyperparameters saved from W&B (see ``fetch_ppo_wandb_configs.py``).

With ``--config``: one JSON file. With no ``--config``: runs the default table-top list in
``default_tabletop_repro_envs.txt`` (next to ``configs/``). Use ``--all-configs`` to run every
JSON except ``manifest.json``.

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


def _load_default_repro_env_ids(list_path: Path) -> list[str]:
    """First column of each non-empty, non-comment line in ``default_tabletop_repro_envs.txt``."""
    text = list_path.read_text(encoding="utf-8")
    ids: list[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        ids.append(line.split()[0])
    return ids


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
        help="Directory of JSON exports (used when --config is omitted)",
    )
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="With --config-dir (default), run every *.json except manifest.json; "
        "default is only envs in default_tabletop_repro_envs.txt next to configs/",
    )
    parser.add_argument(
        "--only-env",
        type=str,
        default=None,
        help="When using --config-dir, restrict to this env_id (e.g. PickCube-v1)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Single seed; default = all seeds in JSON"
    )
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
        help="Extra args for ppo_train_icldata.py (must appear after --extra). Prefer a bare -- below.",
    )

    # Everything after a bare `--` goes to PPO (same as GNU tools). A bare `--` alone does *not*
    # populate argparse.REMAINDER on `--extra`, so sbatch/scripts use `-- ...` and we split here.
    argv = sys.argv[1:]
    passthrough: list[str] = []
    if "--" in argv:
        split_at = argv.index("--")
        passthrough = argv[split_at + 1 :]
        argv = argv[:split_at]

    args = parser.parse_args(argv)
    extra_from_flag: list[str] = []
    if args.extra:
        ex = list(args.extra)
        if ex and ex[0] == "--":
            ex = ex[1:]
        extra_from_flag = ex
    args.extra = extra_from_flag + passthrough

    py = args.python or Path(
        os.environ.get("MANISKILL_PYTHON", _default_maniskill_python(repo_root))
    )

    config_files: list[Path] = []
    if args.config:
        config_files = [args.config]
    else:
        all_json = sorted(p for p in args.config_dir.glob("*.json") if p.name != "manifest.json")
        if args.all_configs:
            config_files = all_json
        else:
            list_path = args.config_dir.parent / "default_tabletop_repro_envs.txt"
            if not list_path.is_file():
                parser.error(
                    f"No --config given and missing {list_path}; pass --config, "
                    f"--all-configs, or add default_tabletop_repro_envs.txt."
                )
            allow = _load_default_repro_env_ids(list_path)
            if not allow:
                parser.error(f"No env ids parsed from {list_path}.")
            by_stem = {p.stem: p for p in all_json}
            for e in allow:
                if e not in by_stem:
                    print(
                        f"[skip] no JSON for default env {e} in {args.config_dir}", file=sys.stderr
                    )
            config_files = [by_stem[e] for e in allow if e in by_stem]

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
