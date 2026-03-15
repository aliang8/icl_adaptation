"""
Run directory layout and experiment naming.

Standard layout:
  outputs/<project_name>/<YYYY-MM-DD>/<run_name>__seed_<X>__<short_hash>/
    .hydra/config.yaml, overrides.yaml
    logs/train.log, stderr.log
    metrics/history.jsonl, summary.json
    ckpts/last/, ckpts/best/, ckpts/step_*/
    artifacts/inference/, artifacts/export/
    eval/val/, eval/test/
    viz/curves.png, viz/samples/
    code/git.txt, diff.patch
    README.md
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf


def get_git_short_hash(length: int = 7) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", str(length), "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip() or None
    except Exception:
        return None


def get_git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def get_git_diff_patch() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "diff", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None


def sanitize_run_name(name: str) -> str:
    """Replace characters that are bad for directory names."""
    return name.replace("/", "_").replace(" ", "_").strip() or "run"


def build_run_slug(run_name: str, seed: int, short_hash: Optional[str] = None) -> str:
    """e.g. my_exp__seed_0__a1b2c3d"""
    slug = f"{sanitize_run_name(run_name)}__seed_{seed}"
    if short_hash:
        slug += f"__{short_hash}"
    return slug


# Subdirs to create under run_dir
RUN_SUBDIRS = [
    ".hydra",
    "logs",
    "metrics",
    "ckpts",
    "ckpts/last",
    "ckpts/best",
    "artifacts",
    "artifacts/inference",
    "artifacts/export",
    "eval",
    "eval/val",
    "eval/test",
    "viz",
    "viz/samples",
    "code",
]


def create_run_dir(
    project_name: str,
    run_name: str,
    seed: int,
    base_dir: str = "outputs",
    overrides: Optional[List[str]] = None,
) -> Path:
    """
    Create the standard run directory and subdirs. Write .hydra, code/, README.
    Returns the run directory Path.
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    short_hash = get_git_short_hash()
    slug = build_run_slug(run_name, seed, short_hash)
    run_dir = Path(base_dir) / project_name / date_str / slug
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    for sub in RUN_SUBDIRS:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    # Code snapshot
    commit = get_git_commit()
    if commit:
        (run_dir / "code" / "git.txt").write_text(
            f"commit: {commit}\nshort: {short_hash or 'n/a'}\n"
        )
    diff = get_git_diff_patch()
    if diff:
        (run_dir / "code" / "diff.patch").write_text(diff)

    # README placeholder
    readme = run_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            f"# Run: {slug}\n\n"
            f"- project: {project_name}\n"
            f"- run_name: {run_name}\n"
            f"- seed: {seed}\n"
            f"- created: {datetime.now().isoformat()}\n\n"
            "## Layout\n"
            "- `.hydra/` config and overrides\n"
            "- `logs/` train.log\n"
            "- `metrics/` history.jsonl, summary.json\n"
            "- `ckpts/last/`, `ckpts/best/`, `ckpts/step_*/` checkpoints\n"
            "- `artifacts/inference/` exported model\n"
            "- `eval/val/`, `eval/test/` eval metrics\n"
            "- `viz/samples/step_*/` rollout visualizations\n"
            "- `code/` git commit and diff\n"
        )

    return run_dir


def write_hydra_config(run_dir: Path, cfg: Any, overrides: Optional[List[str]] = None) -> None:
    """Write .hydra/config.yaml and .hydra/overrides.yaml."""
    hydra_dir = run_dir / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    with open(hydra_dir / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    overrides = overrides or []
    with open(hydra_dir / "overrides.yaml", "w") as f:
        f.write(OmegaConf.to_yaml({"override": overrides}))


def append_metrics_history(run_dir: Path, step: int, metrics: Dict[str, float]) -> None:
    """Append one line to metrics/history.jsonl."""
    import json

    path = run_dir / "metrics" / "history.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({"step": step, **metrics}) + "\n")


def write_metrics_summary(run_dir: Path, summary: Dict[str, Any]) -> None:
    """Write metrics/summary.json at end of run."""
    import json

    path = run_dir / "metrics" / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
