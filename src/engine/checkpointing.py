"""
Checkpointing: full resume state (model, optimizer, scheduler, scaler, epoch, step, best_metric, config, git, RNG).
Saves under ckpts/last/, ckpts/best/, ckpts/step_XXXXX/.
Separate inference/export artifact under artifacts/inference/.
"""

import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf


def _get_git_commit() -> Optional[str]:
    """Return full commit hash, or None if git fails (e.g. no commits, not a repo)."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        return out or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _get_rng_state() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state().tolist(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [t.cpu().tolist() for t in torch.cuda.get_rng_state_all()]
    else:
        state["torch_cuda"] = None
    return state


def _set_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(torch.tensor(state["torch_cpu"], dtype=torch.uint8))
    if state.get("torch_cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [torch.tensor(s, dtype=torch.uint8) for s in state["torch_cuda"]]
        )


def save_checkpoint(
    save_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_metric: float,
    cfg: Any,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[Any] = None,
    sampler_state: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
    kind: str = "latest",
    rank: int = 0,
) -> str:
    """
    Save a full training checkpoint (resumable).
    kind: "latest" | "best" | "periodic_{step}"
    Only rank 0 should call this when using DDP.
    """
    if rank != 0:
        return ""
    os.makedirs(save_dir, exist_ok=True)
    config_container = (
        OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else cfg
    )

    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "config": config_container,
        "git_commit": _get_git_commit(),
        "rng": _get_rng_state(),
        "sampler": sampler_state,
        **(extra or {}),
    }
    # Save under ckpts/last/, ckpts/best/, or ckpts/step_XXXXX/
    if kind == "latest":
        subdir = os.path.join(save_dir, "last")
    elif kind == "best":
        subdir = os.path.join(save_dir, "best")
    else:
        subdir = os.path.join(save_dir, f"step_{global_step:06d}")
    os.makedirs(subdir, exist_ok=True)
    path = os.path.join(subdir, "checkpoint.pt")
    torch.save(ckpt, path)
    metadata = {
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "kind": kind,
    }
    with open(os.path.join(subdir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    return path


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[Any] = None,
    device: Optional[torch.device] = None,
    restore_rng: bool = True,
    weights_only: bool = False,
) -> Tuple[int, int, float, Optional[Dict], Optional[Dict]]:
    """
    Load a training checkpoint. Returns (epoch, global_step, best_metric, config, rng_state).
    If weights_only=True, only model weights are loaded (safer for untrusted checkpoints).
    """
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=weights_only)
    model.load_state_dict(ckpt["model"], strict=True)
    if not weights_only and optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if not weights_only and scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if not weights_only and scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    epoch = ckpt.get("epoch", 0)
    global_step = ckpt.get("global_step", 0)
    best_metric = ckpt.get("best_metric", float("-inf"))
    config = ckpt.get("config")
    rng = ckpt.get("rng")
    if restore_rng and rng is not None:
        _set_rng_state(rng)
    return epoch, global_step, best_metric, config, rng


def save_inference_artifact(
    save_dir: str,
    model: torch.nn.Module,
    cfg: Any,
    state_mean: Optional[np.ndarray] = None,
    state_std: Optional[np.ndarray] = None,
    filename: str = "model_export.pt",
    rank: int = 0,
) -> str:
    """
    Save a minimal artifact for inference only (no optimizer/scheduler/RNG).
    Include state normalization stats if provided.
    """
    if rank != 0:
        return ""
    os.makedirs(save_dir, exist_ok=True)
    artifact = {
        "model": model.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True) if cfg is not None else None,
        "state_mean": state_mean,
        "state_std": state_std,
    }
    path = os.path.join(save_dir, filename)
    torch.save(artifact, path)
    return path
