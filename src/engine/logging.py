"""
Logging: train/val loss, main metric, lr, grad norm, throughput, GPU memory, checkpoint path.
Supports TensorBoard and W&B; save resolved config and run metadata.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

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
        self.use_wandb = use_wandb
        self._step_timer = time.perf_counter()
        self._step_count = 0

        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir=str(self.log_dir))

        self._wandb = None
        if use_wandb:
            import wandb
            from omegaconf import OmegaConf

            cfg_dict = OmegaConf.to_container(config, resolve=True) if config is not None else {}
            self._wandb = wandb.init(project=project, entity=entity, name=run_name, config=cfg_dict)

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

    def log_video(
        self,
        tag: str,
        frames: list,
        step: int,
        fps: int = 20,
        format: str = "mp4",
    ) -> None:
        """Log a list of numpy frames (H,W,3) uint8 as a single video to W&B. Uses (T,C,H,W) for wandb."""
        if self._wandb is None or not frames:
            return
        import numpy as np
        import wandb

        arr = np.stack(frames)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        # W&B expects (T, C, H, W); we have (T, H, W, C)
        if arr.ndim == 4 and arr.shape[-1] == 3:
            arr = np.moveaxis(arr, -1, 1)
        self._wandb.log(
            {tag: wandb.Video(arr, fps=fps, format=format)},
            step=step,
        )

    def log_video_from_path(self, tag: str, path: Union[str, Path], step: int) -> None:
        """Log a video from a file path (e.g. mp4). W&B uses the file's frame rate; passing fps is ignored and triggers a warning."""
        if self._wandb is None:
            return
        import wandb

        path = Path(path)
        if not path.exists():
            return
        self._wandb.log(
            {tag: wandb.Video(str(path), format="mp4")},
            step=step,
        )

    def log_image(self, tag: str, image: Any, step: int) -> None:
        """Log an image to W&B (PIL/ndarray or path). For matplotlib, pass fig or save to buffer."""
        if self._wandb is None:
            return
        import wandb

        if hasattr(image, "savefig"):
            import io

            buf = io.BytesIO()
            image.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            from PIL import Image

            img = Image.open(buf)
            self._wandb.log({tag: wandb.Image(img)}, step=step)
        elif isinstance(image, (str, Path)) and Path(image).exists():
            self._wandb.log({tag: wandb.Image(str(image))}, step=step)
        else:
            self._wandb.log({tag: wandb.Image(image)}, step=step)

    def flush(self) -> None:
        if self._writer is not None:
            self._writer.flush()
        if self._wandb is not None:
            self._wandb.finish()

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
    batch_metrics: Optional[Dict[str, float]] = None,
    checkpoint_path: Optional[str] = None,
) -> None:
    metrics = {
        "train/action_loss": train_loss,
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
    if batch_metrics:
        metrics.update(batch_metrics)
    logger.log_metrics(metrics, step)
    if checkpoint_path:
        logger.log_scalar("checkpoint/path_step", step, step)
