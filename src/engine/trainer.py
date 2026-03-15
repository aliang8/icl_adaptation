"""
Training loop: forward, loss, backward, grad clip, optimizer step, scheduler step,
log metrics, checkpoint (latest/best/periodic), eval hook. Uses tqdm progress bar.
"""

import os
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from loguru import logger as loguru_logger
from tqdm import tqdm

from src.engine.checkpointing import save_checkpoint, load_checkpoint, save_inference_artifact
from src.engine.logging import Logger, log_metrics


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: torch.device,
        cfg: Any,
        logger: Logger,
        scaler: Optional[Any] = None,
        grad_clip_norm: float = 0.25,
        save_dir: Optional[str] = None,
    ):
        self.model = model
        self._save_dir_override = save_dir
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.cfg = cfg
        self.logger = logger
        self.scaler = scaler
        self.grad_clip_norm = grad_clip_norm
        self.exp_cfg = cfg.experiment
        self.sys_cfg = cfg.system
        self.optim_cfg = cfg.optim
        self.save_dir = getattr(self, "_save_dir_override", None) or self.sys_cfg.save_dir
        self.rank = self.sys_cfg.rank
        self.best_metric_name = self.exp_cfg.best_metric_name
        self.best_metric_mode = self.exp_cfg.best_metric_mode
        self._is_better = max if self.best_metric_mode == "max" else min

    def train_step(
        self,
        batch: Tuple[Any, ...],
        step_fn: Callable[
            [torch.nn.Module, Tuple[Any, ...]], Tuple[torch.Tensor, Optional[torch.Tensor]]
        ],
    ) -> Tuple[float, Optional[float]]:
        """Single step: forward, loss, backward, grad clip, step. Returns (loss, grad_norm)."""
        self.model.train()
        self.optimizer.zero_grad()
        loss, grad_norm = step_fn(self.model, batch)
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            if self.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.grad_clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        loss_val = loss.detach().cpu().item()
        grad_norm_val = (
            grad_norm.detach().cpu().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        )
        return loss_val, grad_norm_val

    def run_training(
        self,
        train_loader: DataLoader,
        global_step_start: int,
        best_metric_start: float,
        step_fn: Callable[
            [torch.nn.Module, Tuple[Any, ...]], Tuple[torch.Tensor, Optional[torch.Tensor]]
        ],
        eval_fn: Optional[Callable[[int], Dict[str, float]]] = None,
        state_mean: Optional[Any] = None,
        state_std: Optional[Any] = None,
        export_dir: Optional[str] = None,
    ) -> Tuple[int, float]:
        """
        Run training until max_steps. Saves latest/best/periodic checkpoints.
        step_fn(model, batch) -> (loss, grad_norm).
        eval_fn(step) -> dict of metrics (e.g. eval/return_mean).
        Returns (final_global_step, best_metric).
        """
        max_steps = self.exp_cfg.max_steps
        eval_every = self.exp_cfg.eval_every_steps
        save_latest_every = self.exp_cfg.save_latest_every_steps
        save_periodic_every = self.exp_cfg.save_periodic_every_steps
        save_best = self.exp_cfg.save_best
        export_final = self.exp_cfg.export_final

        global_step = global_step_start
        best_metric = best_metric_start
        epoch = 0
        iter_loader = iter(train_loader)
        loguru_logger.info(
            "Training started: max_steps={}, eval_every={}, save_latest_every={}",
            max_steps,
            eval_every,
            save_latest_every,
        )

        pbar = tqdm(
            total=max_steps,
            initial=global_step_start,
            unit="step",
            dynamic_ncols=True,
            leave=True,
            desc="Train",
        )

        while global_step <= max_steps:
            try:
                batch = next(iter_loader)
            except StopIteration:
                epoch += 1
                iter_loader = iter(train_loader)
                batch = next(iter_loader)
            batch = tuple(x.to(self.device) if isinstance(x, torch.Tensor) else x for x in batch)
            loss_val, grad_norm_val = self.train_step(batch, step_fn)
            lr = self.optimizer.param_groups[0]["lr"]
            gpu_mem = (
                torch.cuda.max_memory_allocated(self.device) / 1e6
                if torch.cuda.is_available()
                else None
            )
            log_metrics(
                self.logger,
                global_step,
                train_loss=loss_val,
                lr=lr,
                grad_norm=grad_norm_val,
                gpu_mem_mb=gpu_mem,
            )

            pbar.update(1)
            pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{lr:.2e}", gn=f"{grad_norm_val:.3f}")

            # Eval
            eval_metrics = None
            if eval_fn and eval_every > 0 and global_step % eval_every == 0 and global_step > 0:
                pbar.write(f"Evaluating at step {global_step}...")
                eval_metrics = eval_fn(global_step)
                log_metrics(self.logger, global_step, loss_val, lr, eval_metrics=eval_metrics)
                current = eval_metrics.get(self.best_metric_name)
                if current is not None:
                    pbar.write(
                        f"Step {global_step} | loss={loss_val:.4f} | lr={lr:.2e} | {self.best_metric_name}={current:.4f}"
                    )
                    pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{lr:.2e}", eval=f"{current:.4f}")
                    if self._is_better(current, best_metric):
                        best_metric = current
                        if save_best and self.rank == 0:
                            save_checkpoint(
                                self.save_dir,
                                self.model,
                                self.optimizer,
                                epoch,
                                global_step,
                                best_metric,
                                self.cfg,
                                scheduler=self.scheduler,
                                scaler=self.scaler,
                                kind="best",
                                rank=self.rank,
                            )
                            pbar.write(
                                f"Saved best checkpoint ({self.best_metric_name}={best_metric:.4f})"
                            )

            # Checkpoints
            if self.rank == 0:
                if save_latest_every and global_step % save_latest_every == 0:
                    save_checkpoint(
                        self.save_dir,
                        self.model,
                        self.optimizer,
                        epoch,
                        global_step,
                        best_metric,
                        self.cfg,
                        scheduler=self.scheduler,
                        scaler=self.scaler,
                        kind="latest",
                        rank=self.rank,
                    )
                if save_periodic_every and global_step % save_periodic_every == 0:
                    save_checkpoint(
                        self.save_dir,
                        self.model,
                        self.optimizer,
                        epoch,
                        global_step,
                        best_metric,
                        self.cfg,
                        scheduler=self.scheduler,
                        scaler=self.scaler,
                        kind=f"periodic_{global_step}",
                        rank=self.rank,
                    )

            global_step += 1
            if global_step > max_steps:
                break

        pbar.close()
        if export_final and self.rank == 0:
            out_dir = export_dir or self.save_dir
            if export_dir:
                os.makedirs(out_dir, exist_ok=True)
            save_inference_artifact(
                out_dir,
                self.model,
                self.cfg,
                state_mean=state_mean,
                state_std=state_std,
                filename="model_export.pt",
                rank=self.rank,
            )
        return global_step, best_metric
