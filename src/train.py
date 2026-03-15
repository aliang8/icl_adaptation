"""
Single training entrypoint: start fresh, resume from checkpoint, eval-only, or save final export.
No notebook-only training logic. Use notebooks only for analysis.
"""
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from loguru import logger as log

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine.checkpointing import load_checkpoint, save_inference_artifact
from src.engine.logging import setup_logging
from src.engine.trainer import Trainer
from src.models import MetaDecisionTransformer, RNNContextEncoder
from src.data import ICLTrajectoryDataset
from src.data.trajectories import convert_data_to_trajectories, sort_trajectories_by_return


def _print_config(cfg):
    """Print resolved config with rich formatting."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    yaml_str = OmegaConf.to_yaml(cfg)
    syntax = Syntax(yaml_str, "yaml", theme="monokai", line_numbers=False)
    console = Console()
    console.print(Panel(syntax, title="[bold]Resolved config[/bold]", border_style="blue"))


def _print_model_architecture(model, title="Model architecture"):
    """Print model summary (param counts and top-level structure) with rich."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    def _count_params(module):
        return sum(p.numel() for p in module.parameters())

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Top-level modules and their param counts
    table = Table(show_header=True, header_style="bold")
    table.add_column("Module", style="cyan")
    table.add_column("Parameters", justify="right", style="green")
    for name, child in model.named_children():
        n = _count_params(child)
        table.add_row(name, f"{n:,}")
    table.add_row("[bold]Total[/bold]", f"[bold]{total:,}[/bold]")
    table.add_row("Trainable", f"{trainable:,}")
    console = Console()
    console.print(Panel(table, title=f"[bold]{title}[/bold] (MetaDecisionTransformer)", border_style="green"))
    # Optional: full repr in a collapsed/smaller panel
    console.print(Panel(str(model), title="[dim]Full model repr[/dim]", border_style="dim"))

# Env observation/action dims (for known envs)
ENV_DIMS = {
    "HalfCheetah-v2": (17, 6),
    "AntDir-v0": (27, 8),
    "WalkerRandParams-v0": (17, 6),
    "HopperRandParams-v0": (11, 3),
}


def get_config(config_dir: str, overrides: list = None):
    """Load composed Hydra config from config_dir."""
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config", overrides=overrides or [])
    return cfg


def build_model(cfg, state_dim: int, action_dim: int):
    m = cfg.model
    n_inner = getattr(m, "n_inner", None) or (4 * m.hidden_size)
    model = MetaDecisionTransformer(
        state_dim=state_dim,
        act_dim=action_dim,
        hidden_size=m.hidden_size,
        context_dim=m.context_dim,
        max_length=m.max_length,
        max_ep_len=m.max_ep_len,
        n_layer=m.n_layer,
        n_head=m.n_head,
        n_inner=n_inner,
        activation_function=m.activation_function,
        resid_pdrop=m.resid_pdrop,
        attn_pdrop=m.attn_pdrop,
        action_tanh=getattr(m, "action_tanh", True),
    )
    return model


def build_optimizer_scheduler(model, cfg):
    o = cfg.optim
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=o.lr,
        weight_decay=o.weight_decay,
    )
    warmup = o.warmup_steps
    def lr_lambda(step):
        return min((step + 1) / warmup, 1.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def train_step_fn(model, batch):
    """One step: forward, MSE loss on actions, return (loss, grad_norm)."""
    (states, contexts, actions, rewards, dones, rtg, timesteps, masks,
     prompt_s, prompt_a, prompt_r, prompt_rtg, prompt_ts, prompt_m) = batch
    prompt_rtg = prompt_rtg[:, :-1, :]
    prompt = (prompt_s, prompt_a, prompt_r, prompt_rtg, prompt_ts, prompt_m)
    state_preds, action_preds, return_preds = model(
        states, contexts, actions, rewards, rtg[:, :-1], timesteps,
        attention_mask=masks, prompt=prompt,
    )
    act_dim = action_preds.shape[2]
    action_preds = action_preds.reshape(-1, act_dim)[masks.reshape(-1) > 0]
    action_target = actions.reshape(-1, act_dim)[masks.reshape(-1) > 0]
    loss = torch.nn.functional.mse_loss(action_preds, action_target)
    with torch.no_grad():
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
    return loss, grad_norm


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=str, default="configs", help="Hydra config directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    parser.add_argument("--export-only", type=str, default=None, help="Export inference artifact from this checkpoint")
    parser.add_argument(
        "--override",
        action="append",
        nargs="*",
        default=[],
        metavar="KEY=VAL",
        help="Hydra overrides (e.g. --override experiment.max_steps=1000 or --override system.wandb_entity=clvr system.wandb_project=icl_adaptation)",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--run-name", type=str, default=None, help="W&B / run name")
    args = parser.parse_args()

    config_dir = os.path.abspath(args.config_dir)
    if not os.path.isdir(config_dir):
        config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))
    # Flatten overrides: --override a=1 b=2 and --override a=1 --override b=2 both supported
    overrides = []
    for ov in args.override:
        overrides.extend(ov if isinstance(ov, list) else [ov])
    cfg = get_config(config_dir, overrides=overrides)
    OmegaConf.resolve(cfg)

    sys_cfg = cfg.system
    data_cfg = cfg.data
    device = torch.device(sys_cfg.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(sys_cfg.seed)
    np.random.seed(sys_cfg.seed)
    random.seed(sys_cfg.seed)
    if getattr(sys_cfg, "deterministic", True):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    env_name = getattr(data_cfg, "env_name", "HalfCheetah-v2")
    state_dim, action_dim = ENV_DIMS.get(env_name, (cfg.model.state_dim, cfg.model.act_dim))
    log.info("Env {} -> state_dim={}, action_dim={}", env_name, state_dim, action_dim)
    save_dir = os.path.join(sys_cfg.output_dir, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    log_dir = os.path.join(sys_cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Save resolved config
    with open(os.path.join(sys_cfg.output_dir, "resolved_config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    _print_config(cfg)

    if args.export_only:
        model = build_model(cfg, state_dim, action_dim)
        load_checkpoint(args.export_only, model, device=device, weights_only=True)
        save_inference_artifact(save_dir, model, cfg, filename="model_export.pt", rank=0)
        log.info("Exported model_export.pt to {}", save_dir)
        return

    # Build data: D4RL HalfCheetah (trajectories.pkl) or AntDir-style (dataset_task_*.pkl) or dummy
    data_dir = os.path.join(data_cfg.data_dir, data_cfg.env_name, data_cfg.data_quality)
    trajectories = []
    prompt_per_task = []

    if env_name == "HalfCheetah-v2":
        from src.data.d4rl_loader import load_halfcheetah_trajectories
        trajectories, prompt_per_task = load_halfcheetah_trajectories(
            data_cfg.data_dir,
            env_name=data_cfg.env_name,
            data_quality=data_cfg.data_quality,
        )
        if trajectories:
            log.info("Loaded {} HalfCheetah trajectories (mixed returns) from {}", len(trajectories), data_dir)

    if not trajectories and os.path.isdir(data_dir):
        import pickle
        for task_id in range(getattr(data_cfg, "num_train_tasks", 1)):
            path = os.path.join(data_dir, f"dataset_task_{task_id}.pkl")
            if os.path.isfile(path):
                with open(path, "rb") as f:
                    d = pickle.load(f)
                if "states" in d:
                    d["observations"] = d["states"]
                    d["next_observations"] = d.get("next_states", d["states"])
                    d["terminals"] = d.get("dones", d.get("terminals", np.zeros(len(d["rewards"]))))
                trajs = convert_data_to_trajectories(d, data_cfg.max_episode_steps, max_trajectories=500)
                trajectories.extend(trajs)
        for task_id in range(getattr(data_cfg, "num_tasks", 1)):
            path = os.path.join(data_dir, f"dataset_task_prompt{task_id}.pkl")
            if os.path.isfile(path):
                with open(path, "rb") as f:
                    prompt_per_task.append(pickle.load(f))
            else:
                prompt_per_task.append(trajectories[:5] if trajectories else [])
        if trajectories:
            log.info("Loaded {} trajectories from {}", len(trajectories), data_dir)

    if not trajectories:
        # Dummy data for dry run
        n_traj = 20
        T = data_cfg.max_episode_steps
        trajectories = [
            {
                "observations": np.random.randn(T, state_dim).astype(np.float32),
                "actions": np.random.randn(T, action_dim).astype(np.float32),
                "rewards": np.random.randn(T).astype(np.float32),
                "next_observations": np.random.randn(T, state_dim).astype(np.float32),
                "terminals": np.zeros(T),
            }
            for _ in range(n_traj)
        ]
        prompt_per_task = [sort_trajectories_by_return(trajectories[:5], ascending=False)] * max(1, getattr(data_cfg, "num_tasks", 1))
        log.warning("No dataset found at {}; using dummy data for dry run", data_dir)

    dataset = ICLTrajectoryDataset(
        trajectories=trajectories,
        horizon=data_cfg.horizon,
        max_episode_steps=data_cfg.max_episode_steps,
        return_scale=data_cfg.return_scale,
        device=device,
        prompt_trajectories_per_task=prompt_per_task,
        context_dim=data_cfg.context_dim,
        state_dim=state_dim,
        act_dim=action_dim,
        prompt_length=data_cfg.prompt_length,
        scale=data_cfg.return_scale,
        total_epi_per_task=max(1, len(trajectories) // max(1, getattr(data_cfg, "num_train_tasks", 1))),
        num_context_trajectories=getattr(data_cfg, "num_context_trajectories", 1),
        context_sort_ascending=getattr(data_cfg, "context_sort_ascending", True),
        context_sampling=getattr(data_cfg, "context_sampling", "random"),
        max_total_prompt_length=getattr(data_cfg, "max_total_prompt_length", None),
    )
    state_mean = dataset.state_mean
    state_std = dataset.state_std
    log.info("Dataset size: {} segments, state_mean/std computed", len(dataset))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=data_cfg.batch_size,
        shuffle=True,
        num_workers=getattr(data_cfg, "num_workers", 0),
    )

    model = build_model(cfg, state_dim, action_dim).to(device)
    _print_model_architecture(model)
    optimizer, scheduler = build_optimizer_scheduler(model, cfg)
    use_wandb = getattr(args, "wandb", False) or getattr(sys_cfg, "use_wandb", False)
    run_name = getattr(args, "run_name", None) or getattr(cfg, "run_name", None) or getattr(sys_cfg, "run_name", None)
    logger = setup_logging(
        log_dir,
        cfg,
        use_wandb=use_wandb,
        run_name=run_name,
        project=getattr(sys_cfg, "wandb_project", "icl_adaptation"),
        entity=getattr(sys_cfg, "wandb_entity", "clvr"),
    )
    if use_wandb:
        log.info("W&B logging enabled (entity: {}, project: {})", sys_cfg.wandb_entity, sys_cfg.wandb_project)

    global_step_start = 0
    best_metric_start = float("-inf")

    if args.resume:
        _, global_step_start, best_metric_start, _, _ = load_checkpoint(
            args.resume, model, optimizer=optimizer, scheduler=scheduler, device=device, restore_rng=True,
        )

    if args.eval_only:
        model.eval()
        logger.log_scalar("eval/return_mean", 0.0, global_step_start)
        logger.flush()
        log.info("Eval-only: logged placeholder metric. Wire env for full eval.")
        return

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        cfg=cfg,
        logger=logger,
        grad_clip_norm=cfg.optim.grad_clip_norm,
    )

    def eval_fn(step):
        # Placeholder: real eval would run env with model.get_action and context encoder
        return {"eval/return_mean": 0.0}

    log.info("Starting training loop (max_steps={})", cfg.experiment.max_steps)
    trainer.run_training(
        train_loader=loader,
        global_step_start=global_step_start,
        best_metric_start=best_metric_start,
        step_fn=train_step_fn,
        eval_fn=eval_fn,
        state_mean=state_mean,
        state_std=state_std,
    )
    log.info("Training finished.")
    logger.close()


if __name__ == "__main__":
    main()
