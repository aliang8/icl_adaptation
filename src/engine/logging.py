"""
Logging: train/val loss, main metric, lr, grad norm, throughput, GPU memory, checkpoint path.
Supports TensorBoard and W&B; save resolved config and run metadata.
"""
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch


class Logger:
    """Unified logger (TensorBoard + optional W&B)."""

    def __init__(
        self,
        log_dir: str,
        use_wandb: bool = False,
        project: str = "icl_adaptation",
        entity: str = "clvr",
        run_name: Optional[str] = None,
        config: Optional[Any] = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = None
        self._wandb = None
        self.use_wandb = use_wandb
        self._step_timer = time.perf_counter()
        self._step_count = 0

        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=str(self.log_dir))
        except Exception:
            pass

        if use_wandb:
            try:
                import wandb
                from omegaconf import OmegaConf
                cfg_dict = OmegaConf.to_container(config, resolve=True) if config is not None else {}
                self._wandb = wandb.init(project=project, entity=entity, name=run_name, config=cfg_dict)
            except Exception:
                self._wandb = None

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                if self._writer is not None:
                    self._writer.add_scalar(k, v, step)
                if self._wandb is not None:
                    self._wandb.log({k: v, "train/global_step": step}, step=step)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self._writer is not None:
            self._writer.add_scalar(tag, value, step)
        if self._wandb is not None:
            self._wandb.log({tag: value}, step=step)

    def flush(self) -> None:
        if self._writer is not None:
            self._writer.flush()
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
        self.flush()


def setup_logging(
    log_dir: str,
    config: Any,
    use_wandb: bool = False,
    run_name: Optional[str] = None,
    project: str = "icl_adaptation",
    entity: str = "clvr",
) -> Logger:
    """Create logger and optionally save resolved config."""
    logger = Logger(
        log_dir=log_dir,
        use_wandb=use_wandb,
        project=project,
        entity=entity,
        run_name=run_name,
        config=config,
    )
    return logger


def log_metrics(
    logger: Logger,
    step: int,
    train_loss: float,
    lr: float,
    grad_norm: Optional[float] = None,
    throughput: Optional[float] = None,
    gpu_mem_mb: Optional[float] = None,
    eval_metrics: Optional[Dict[str, float]] = None,
    checkpoint_path: Optional[str] = None,
) -> None:
    metrics = {
        "train/loss": train_loss,
        "train/lr": lr,
    }
    if grad_norm is not None:
        metrics["train/grad_norm"] = grad_norm
    if throughput is not None:
        metrics["train/throughput"] = throughput
    if gpu_mem_mb is not None:
        metrics["system/gpu_mem_mb"] = gpu_mem_mb
    if eval_metrics:
        metrics.update(eval_metrics)
    logger.log_metrics(metrics, step)
    if checkpoint_path:
        logger.log_scalar("checkpoint/path_step", step, step)
