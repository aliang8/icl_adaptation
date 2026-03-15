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
from src.engine.run_dir import (
    create_run_dir,
    write_hydra_config,
    append_metrics_history,
    write_metrics_summary,
)
from src.engine.eval_viz import run_rollouts_and_save_viz
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


def _print_dataset_stats(dataset, loader, env_name: str = "", data_quality: str = ""):
    """Print dataset statistics with rich formatting."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    n_trajectories = len(dataset.trajectories)
    n_segments = len(dataset)
    total_steps = sum(t["rewards"].shape[0] for t in dataset.trajectories)

    table = Table(show_header=True, header_style="bold")
    table.add_column("Stat", style="cyan")
    table.add_column("Value", justify="right", style="green")
    table.add_row("Trajectories", f"{n_trajectories:,}")
    table.add_row("Segments (training)", f"{n_segments:,}")
    table.add_row("Total steps", f"{total_steps:,}")
    table.add_row("Return (min)", f"{dataset.return_min:.2f}")
    table.add_row("Return (max)", f"{dataset.return_max:.2f}")
    table.add_row("Return (mean)", f"{dataset.return_avg:.2f}")
    table.add_row("State dim", str(dataset.state_dim))
    table.add_row("Action dim", str(dataset.act_dim))
    table.add_row("Horizon", str(dataset.horizon))
    table.add_row("Max episode steps", str(dataset.max_episode_steps))
    table.add_row("Context trajectories", str(dataset.num_context_trajectories))
    table.add_row("Prompt length", str(dataset.prompt_length))
    table.add_row("Total prompt length", str(dataset.total_prompt_len))
    table.add_row("Batch size", str(loader.batch_size))
    table.add_row("Batches per epoch", f"{len(loader):,}")

    title = "Dataset stats"
    if env_name or data_quality:
        title += f" — {env_name or '?'}"
        if data_quality:
            title += f" / {data_quality}"
    console = Console()
    console.print(Panel(table, title=f"[bold]{title}[/bold]", border_style="magenta"))


# Env observation/action dims (for known envs)
ENV_DIMS = {
    "HalfCheetah-v2": (17, 6),
    "AntDir-v0": (27, 8),
    "ICRT-MT": (8, 8),  # proprio (e.g. 3+1 or 6+1), action same
    "LIBERO-Cosmos": (9, 7),  # proprio 9, action 7
    "WalkerRandParams-v0": (17, 6),
    "HopperRandParams-v0": (11, 3),
}


def get_config(config_dir: str, overrides: list = None):
    """Load composed Hydra config from config_dir."""
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config", overrides=overrides or [])
    return cfg


def resolve_paths(cfg):
    """Resolve path config to absolute Paths. Relative paths are under repo_root."""
    paths_cfg = cfg.paths
    sys_cfg = cfg.system
    data_cfg = cfg.data
    repo_root = Path(__file__).resolve().parent.parent
    if str(paths_cfg.repo_root) != ".":
        repo_root = Path(paths_cfg.repo_root).resolve()
    data_root_str = paths_cfg.data_root
    data_root = Path(data_root_str).resolve() if Path(data_root_str).is_absolute() else (repo_root / data_root_str).resolve()
    output_root_str = paths_cfg.output_root
    output_root = Path(output_root_str).resolve() if Path(output_root_str).is_absolute() else (repo_root / output_root_str).resolve()
    from types import SimpleNamespace
    return SimpleNamespace(repo_root=repo_root, data_root=data_root, output_root=output_root)


def validate_dataset_paths(env_name: str, paths, data_cfg) -> None:
    """Fail fast at startup if required dataset paths are missing."""
    if env_name == "ICRT-MT":
        config_path = paths.data_root / "ICRT-MT" / "dataset_config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"ICRT-MT dataset not found at {config_path}.\n"
                f"Run: python scripts/download_icrt_dataset.py --output-dir {paths.data_root}"
            )
        log.info("ICRT-MT dataset config found: {}", config_path)
    elif env_name == "LIBERO-Cosmos":
        manifest_path = paths.data_root / "LIBERO-Cosmos-Policy" / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"LIBERO-Cosmos manifest not found at {manifest_path}.\n"
                f"Run: python scripts/download_libero_cosmos.py --output-dir {paths.data_root}"
            )
        log.info("LIBERO-Cosmos manifest found: {}", manifest_path)


def build_model(cfg, state_dim: int, action_dim: int):
    m = cfg.model
    n_inner = m.n_inner or (4 * m.hidden_size)
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
        action_tanh=m.action_tanh,
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
    """One step: forward, MSE loss on actions, return (loss, grad_norm). Batch includes 15 elements (14 tensors + instructions)."""
    (states, contexts, actions, rewards, dones, rtg, timesteps, masks,
     prompt_s, prompt_a, prompt_r, prompt_rtg, prompt_ts, prompt_m, instructions) = batch
    # instructions: ICRT-style language (one per sample); unused for HalfCheetah, used later for robot manipulation
    # Trim last step from all prompt tensors so lengths match (prompt_rtg is used as returns_to_go and trimmed to T-1)
    prompt_rtg = prompt_rtg[:, :-1, :]
    prompt_s = prompt_s[:, :-1, :]
    prompt_a = prompt_a[:, :-1, :]
    prompt_r = prompt_r[:, :-1, :]
    prompt_ts = prompt_ts[:, :-1]
    prompt_m = prompt_m[:, :-1]
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
    # Use spawn for DataLoader workers so CUDA is not re-initialized in forked subprocesses.
    try:
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
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
    if sys_cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    env_name = data_cfg.env_name
    state_dim, action_dim = ENV_DIMS.get(env_name, (cfg.model.state_dim, cfg.model.act_dim))
    log.info("Env {} -> state_dim={}, action_dim={}", env_name, state_dim, action_dim)

    paths = resolve_paths(cfg)
    log.debug("Resolved paths: data_root={}, output_root={}", paths.data_root, paths.output_root)
    validate_dataset_paths(env_name, paths, data_cfg)

    # Run directory: outputs/<project_name>/<date>/<run_name>__seed_X__<hash>/
    def _infer_run_dir(ckpt_path: str) -> Path:
        p = Path(ckpt_path).resolve()
        if "ckpts" in p.parts:
            idx = p.parts.index("ckpts")
            return Path(*p.parts[:idx]).resolve()
        return p.parent.parent

    run_name = args.run_name or cfg.run_name or sys_cfg.run_name or "train"
    project_name = sys_cfg.project_name
    seed = sys_cfg.seed
    if args.resume and os.path.isfile(args.resume):
        run_dir = _infer_run_dir(args.resume)
        save_dir = str(run_dir / "ckpts")
        log_dir = str(run_dir / "logs")
        run_dir.mkdir(parents=True, exist_ok=True)
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log.info("Resuming: run_dir={}", run_dir)
    elif args.export_only and os.path.isfile(args.export_only):
        run_dir = _infer_run_dir(args.export_only)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "artifacts" / "inference").mkdir(parents=True, exist_ok=True)
        save_dir = str(run_dir / "ckpts")
        log_dir = str(run_dir / "logs")
    else:
        run_dir = create_run_dir(
            project_name=project_name,
            run_name=run_name,
            seed=seed,
            base_dir=str(paths.output_root),
            overrides=overrides,
        )
        write_hydra_config(run_dir, cfg, overrides=overrides)
        save_dir = str(run_dir / "ckpts")
        log_dir = str(run_dir / "logs")
        log.info("Run dir: {}", run_dir)

    # Optional: log to run_dir/logs/train.log
    train_log = Path(log_dir) / "train.log"
    try:
        log.add(train_log, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    except Exception:
        pass
    _print_config(cfg)

    if args.export_only:
        model = build_model(cfg, state_dim, action_dim)
        load_checkpoint(args.export_only, model, device=device, weights_only=True)
        export_dir = run_dir / "artifacts" / "inference"
        export_dir.mkdir(parents=True, exist_ok=True)
        save_inference_artifact(str(export_dir), model, cfg, filename="model_export.pt", rank=0)
        log.info("Exported model_export.pt to {}", export_dir)
        return

    # Build data: D4RL HalfCheetah, LIBERO-Cosmos, AntDir-style (dataset_task_*.pkl), or dummy
    data_root = paths.data_root
    data_dir = data_root / data_cfg.env_name
    if data_cfg.data_quality:
        data_dir = data_dir / data_cfg.data_quality
    if env_name == "LIBERO-Cosmos":
        data_dir = data_root
    trajectories = []
    prompt_per_task = []
    task_instructions_from_loader = None

    if env_name == "HalfCheetah-v2":
        from src.data.d4rl_loader import load_halfcheetah_trajectories
        trajectories, prompt_per_task = load_halfcheetah_trajectories(
            str(data_root),
            env_name=data_cfg.env_name,
            data_quality=data_cfg.data_quality,
        )
        if trajectories:
            log.info("Loaded {} HalfCheetah trajectories (mixed returns) from {}", len(trajectories), data_root)

    if env_name == "ICRT-MT":
        config_path = data_root / "ICRT-MT" / "dataset_config.json"
        assert config_path.exists(), f"ICRT-MT config missing (validated earlier): {config_path}"
        from src.data.icrt_dataset import load_icrt_trajectories
        proprio_keys = data_cfg.proprio_keys or ["observation/cartesian_position", "observation/gripper_position"]
        action_keys = data_cfg.action_keys or ["action/cartesian_position", "action/gripper_position"]
        trajectories, prompt_per_task, task_instructions_from_loader = load_icrt_trajectories(
            str(config_path.resolve()),
            proprio_keys=proprio_keys,
            action_keys=action_keys,
            min_trajectory_length=data_cfg.min_trajectory_length,
            max_trajectory_length=data_cfg.max_trajectory_length,
        )
        if not trajectories:
            raise FileNotFoundError(
                f"ICRT-MT config at {config_path} found but no trajectories loaded. "
                "Check that HDF5 paths in dataset_config.json exist (e.g. merged_data_part1.hdf5 in the same directory) "
                "and that episode lengths are within min_trajectory_length and max_trajectory_length."
            )
        state_dim = int(trajectories[0]["observations"].shape[1])
        action_dim = int(trajectories[0]["actions"].shape[1])
        log.info("Loaded {} ICRT-MT trajectories from {} (state_dim={}, action_dim={})", len(trajectories), config_path, state_dim, action_dim)

    if env_name == "LIBERO-Cosmos":
        from src.data.libero_dataset import load_libero_trajectories
        manifest_path = data_root / "LIBERO-Cosmos-Policy" / "manifest.json"
        trajectories, prompt_per_task, task_instructions_from_loader = load_libero_trajectories(
            str(data_root),
            manifest_path=str(manifest_path) if manifest_path.exists() else None,
            repo_id=data_cfg.libero_repo_id,
        )
        if trajectories:
            log.info("Loaded {} LIBERO-Cosmos trajectories (manifest: {})", len(trajectories), manifest_path.resolve())

    if not trajectories and data_dir.is_dir():
        import pickle
        for task_id in range(data_cfg.num_train_tasks):
            path = data_dir / f"dataset_task_{task_id}.pkl"
            if path.is_file():
                with path.open("rb") as f:
                    d = pickle.load(f)
                if "states" in d:
                    d["observations"] = d["states"]
                    d["next_observations"] = d.get("next_states", d["states"])
                    d["terminals"] = d.get("dones", d.get("terminals", np.zeros(len(d["rewards"]))))
                trajs = convert_data_to_trajectories(d, data_cfg.max_episode_steps, max_trajectories=500)
                trajectories.extend(trajs)
        for task_id in range(data_cfg.num_tasks):
            path = data_dir / f"dataset_task_prompt{task_id}.pkl"
            if path.is_file():
                with path.open("rb") as f:
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
        prompt_per_task = [sort_trajectories_by_return(trajectories[:5], ascending=False)] * max(1, data_cfg.num_tasks)
        log.warning("No dataset found at {}; using dummy data for dry run", data_dir.resolve())

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
        total_epi_per_task=max(1, len(trajectories) // max(1, data_cfg.num_train_tasks)),
        num_context_trajectories=data_cfg.num_context_trajectories,
        context_sort_ascending=data_cfg.context_sort_ascending,
        context_sampling=data_cfg.context_sampling,
        max_total_prompt_length=data_cfg.max_total_prompt_length,
        context_style=data_cfg.context_style,
        lazy_dataset=data_cfg.lazy_dataset,
        max_training_examples=data_cfg.max_training_examples,
        task_instructions=task_instructions_from_loader if task_instructions_from_loader is not None else data_cfg.task_instructions,
        seed=data_cfg.seed,
    )
    state_mean = dataset.state_mean
    state_std = dataset.state_std
    log.info("Dataset size: {} segments, state_mean/std computed", len(dataset))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=data_cfg.batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
    )
    _print_dataset_stats(dataset, loader, env_name=data_cfg.env_name, data_quality=data_cfg.data_quality or "")

    model = build_model(cfg, state_dim, action_dim).to(device)
    _print_model_architecture(model)
    optimizer, scheduler = build_optimizer_scheduler(model, cfg)
    use_wandb = args.wandb or sys_cfg.use_wandb
    run_name = args.run_name or cfg.run_name or sys_cfg.run_name
    logger = setup_logging(
        log_dir,
        cfg,
        use_wandb=use_wandb,
        run_name=run_name,
        project=sys_cfg.wandb_project,
        entity=sys_cfg.wandb_entity,
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
        save_dir=save_dir,
    )

    def eval_fn(step):
        num_rollouts = cfg.experiment.num_eval_rollouts
        metrics = run_rollouts_and_save_viz(
            model=model,
            env_name=env_name,
            state_mean=state_mean,
            state_std=state_std,
            device=device,
            run_dir=run_dir,
            step=step,
            num_rollouts=num_rollouts,
            max_episode_steps=data_cfg.max_episode_steps,
            scale=data_cfg.return_scale,
        )
        append_metrics_history(run_dir, step, metrics)
        return metrics

    log.info("Starting training loop (max_steps={})", cfg.experiment.max_steps)
    export_dir = str(run_dir / "artifacts" / "inference")
    final_step, best_metric = trainer.run_training(
        train_loader=loader,
        global_step_start=global_step_start,
        best_metric_start=best_metric_start,
        step_fn=train_step_fn,
        eval_fn=eval_fn,
        state_mean=state_mean,
        state_std=state_std,
        export_dir=export_dir,
    )
    write_metrics_summary(run_dir, {
        "final_step": final_step,
        "best_metric": best_metric,
        "best_metric_name": cfg.experiment.best_metric_name,
    })
    log.info("Training finished.")
    logger.close()


if __name__ == "__main__":
    main()
