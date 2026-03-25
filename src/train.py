"""
Single training entrypoint: start fresh, resume from checkpoint, eval-only, or save final export.
No notebook-only training logic. Use notebooks only for analysis.
"""

import os
import random
import sys
from pathlib import Path
from typing import Optional

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
from src.engine.eval_action_compare import run_action_compare_eval
from src.engine.trainer import Trainer
from src.models import MetaDecisionTransformer, RNNContextEncoder, VLADecisionTransformer
from src.models.types import DTBatch
from src.data import ICLTrajectoryDataset, collate_icl_batch
from src.data.trajectories import (
    convert_data_to_trajectories,
    sample_context_trajectories,
    sort_trajectories_by_return,
)


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
    """Print model summary (param counts, trainable status, and top-level structure) with rich."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    def _count_params(module):
        return sum(p.numel() for p in module.parameters())

    def _trainable_status(module):
        params = list(module.parameters())
        if not params:
            return "—"
        n_train = sum(1 for p in params if p.requires_grad)
        if n_train == 0:
            return "[dim]no[/dim]"
        if n_train == len(params):
            return "[green]yes[/green]"
        return "[yellow]partial[/yellow]"

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Top-level modules: param counts and trainable status
    table = Table(show_header=True, header_style="bold")
    table.add_column("Module", style="cyan")
    table.add_column("Parameters", justify="right", style="green")
    table.add_column("Trainable", justify="center", style="blue")
    for name, child in model.named_children():
        n = _count_params(child)
        status = _trainable_status(child)
        table.add_row(name, f"{n:,}", status)
    table.add_row("[bold]Total[/bold]", f"[bold]{total:,}[/bold]", "—")
    table.add_row("Trainable", f"{trainable:,}", "—")
    console = Console()
    console.print(
        Panel(table, title=f"[bold]{title}[/bold] (MetaDecisionTransformer)", border_style="green")
    )
    # Optional: full repr in a collapsed/smaller panel
    console.print(Panel(str(model), title="[dim]Full model repr[/dim]", border_style="dim"))


def _print_dataset_stats(
    dataset,
    loader,
    env_name: str = "",
    data_quality: str = "",
    image_keys=None,
    proprio_keys=None,
    use_vision: bool = False,
):
    """Print dataset statistics with rich formatting."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    if image_keys is None:
        image_keys = []
    if proprio_keys is None:
        proprio_keys = []

    trajectories = dataset.trajectories if hasattr(dataset, "trajectories") else None
    n_trajectories = len(trajectories) if trajectories else 0
    n_segments = len(dataset)
    traj_lengths = (
        [t["rewards"].shape[0] for t in trajectories] if isinstance(trajectories, list) else []
    )
    total_steps = sum(traj_lengths)
    min_len = min(traj_lengths) if traj_lengths else 0
    max_len = max(traj_lengths) if traj_lengths else 0
    mean_len = total_steps / n_trajectories if n_trajectories else 0

    table = Table(show_header=True, header_style="bold")
    table.add_column("Stat", style="cyan")
    table.add_column("Value", justify="right", style="green")
    table.add_row("Trajectories", f"{n_trajectories:,}")
    table.add_row("Segments (training)", f"{n_segments:,}")
    table.add_row("Total steps", f"{total_steps:,}")
    table.add_row("Traj length (min / max / mean)", f"{min_len} / {max_len} / {mean_len:.1f}")
    if dataset.task_instructions:
        tasks = dataset.task_instructions
        num_tasks = len(tasks)
        table.add_row("Num tasks", str(num_tasks))
        if num_tasks > 0:
            table.add_row("Episodes per task (approx)", f"{n_trajectories / num_tasks:.1f}")
            # Example tasks (first 5, truncated)
            max_examples = 5
            max_chars = 55
            examples = [
                str(t).strip()[:max_chars] + ("…" if len(str(t).strip()) > max_chars else "")
                for t in tasks[:max_examples]
            ]
            table.add_row("Example tasks", "\n".join(examples) if examples else "—")
    return_min = dataset.return_min if hasattr(dataset, "return_min") else None
    return_max = dataset.return_max if hasattr(dataset, "return_max") else None
    return_avg = dataset.return_avg if hasattr(dataset, "return_avg") else None
    for label, val in [
        ("Return (min)", return_min),
        ("Return (max)", return_max),
        ("Return (mean)", return_avg),
    ]:
        table.add_row(label, f"{val:.2f}" if val is not None else "—")
    table.add_row("State dim", str(dataset.state_dim))
    table.add_row("Action dim", str(dataset.act_dim))
    table.add_row("Horizon", str(dataset.horizon))
    k = dataset._query_length
    table.add_row("Query history length (K)", str(k) + (" (OpenVLA-style)" if k == 1 else ""))
    table.add_row("Max episode steps", str(dataset.max_episode_steps))
    table.add_row("Context trajectories", str(dataset.num_context_trajectories))
    prompt_len = dataset.prompt_length
    table.add_row(
        "Prompt length",
        str(prompt_len) if prompt_len is not None else "— (full traj)",
    )
    max_pt = dataset.max_prompt_trajectory_length
    if max_pt is not None:
        table.add_row("Max prompt trajectory length", str(max_pt))
        table.add_row("Context subsample strategy", str(dataset.context_subsample_strategy))
    table.add_row("Total prompt length", str(dataset.total_prompt_len))
    table.add_row("Use vision", "yes" if use_vision else "no")
    table.add_row("Image keys (config)", ", ".join(image_keys) if image_keys else "—")
    table.add_row("Proprio keys (config)", ", ".join(proprio_keys) if proprio_keys else "—")
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
from src.envs.libero_env import LIBERO_SUITES

ENV_DIMS = {
    "HalfCheetah-v2": (17, 6),
    "AntDir-v0": (27, 8),
    "ICRT-MT": (8, 8),  # proprio (e.g. 3+1 or 6+1), action same
    "WalkerRandParams-v0": (17, 6),
    "HopperRandParams-v0": (11, 3),
    # Placeholder; train overwrites from first trajectory after V-D4RL load
    "VD4RL": (768, 6),
}
for _suite in LIBERO_SUITES:
    ENV_DIMS[_suite] = (9, 7)  # LIBERO proprio 9, action 7


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
    data_root = (
        Path(data_root_str).resolve()
        if Path(data_root_str).is_absolute()
        else (repo_root / data_root_str).resolve()
    )
    output_root_str = paths_cfg.output_root
    output_root = (
        Path(output_root_str).resolve()
        if Path(output_root_str).is_absolute()
        else (repo_root / output_root_str).resolve()
    )
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
    elif env_name in LIBERO_SUITES:
        log.info("LIBERO: dataset at data_dir/LIBERO-Cosmos-Policy/data/ (episode_index)")
    elif env_name == "VD4RL":
        p = (
            paths.data_root
            / data_cfg.vd4rl_suite
            / data_cfg.vd4rl_task
            / data_cfg.vd4rl_split
            / data_cfg.vd4rl_pixel_size
        )
        if not p.is_dir():
            raise FileNotFoundError(
                f"V-D4RL npz directory not found: {p}\n"
                "Download datasets from the Google Drive linked in "
                "https://github.com/conglu1997/v-d4rl and point paths.data_root at the parent "
                f"of `{data_cfg.vd4rl_suite}/` (expected layout: "
                f".../{data_cfg.vd4rl_task}/{data_cfg.vd4rl_split}/{data_cfg.vd4rl_pixel_size}/*.npz)."
            )
        log.info("V-D4RL data directory found: {}", p)


def build_model(cfg, state_dim: int, action_dim: int, num_instructions: Optional[int] = None):
    m = cfg.model
    n_inner = m.n_inner or (4 * m.hidden_size)
    common = dict(
        state_dim=state_dim,
        act_dim=action_dim,
        hidden_size=m.hidden_size,
        context_dim=m.context_dim,
        max_length=m.max_length,
        max_ep_len=m.max_ep_len,
        n_layer=m.n_layer,
        n_head=m.n_head,
        n_inner=n_inner,
        n_positions=m.n_positions,
        activation_function=m.activation_function,
        resid_pdrop=m.resid_pdrop,
        attn_pdrop=m.attn_pdrop,
        action_tanh=m.action_tanh,
        transformer_backbone=m.transformer_backbone,
        llama_model_name=m.llama_model_name,
        query_loss_only=m.query_loss_only,
        predict_returns=m.predict_returns,
        predict_state=m.predict_state,
        condition_rtg=m.condition_rtg,
    )
    if m.use_vision or m.use_language:
        use_precomputed = cfg.data.use_precomputed_embeddings
        precomputed_dim = m.precomputed_vision_embed_dim
        model = VLADecisionTransformer(
            **common,
            use_vision=m.use_vision,
            use_language=m.use_language,
            use_language_input=m.use_language_input,
            num_instructions=num_instructions or 0,
            num_views=m.num_views,
            image_embed_dim=m.image_embed_dim,
            vision_encoder_type=m.vision_encoder_type,
            vision_encoder_pool=m.vision_encoder_pool,
            vision_encoder_attention_pool=m.vision_encoder_attention_pool,
            freeze_vision_encoder=m.freeze_vision_encoder,
            vision_encoder_chunk_size=m.vision_encoder_chunk_size,
            use_precomputed_embeddings=use_precomputed,
            precomputed_vision_embed_dim=precomputed_dim,
            vision_proprio_attention_fusion=m.vision_proprio_attention_fusion,
        )
    else:
        model = MetaDecisionTransformer(**common)
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


def make_train_step_fn(task_instructions, use_precomputed_embeddings: bool = False):
    """Build a train step that passes instruction_indices and optional image_embeddings to VLA-DT."""
    task_list = list(task_instructions) if task_instructions else []
    _train_batch_logged = [False]
    _vision_encoder_logged = [False]

    _BATCH_NAMES = [
        "states",
        "contexts",
        "actions",
        "rewards",
        "dones",
        "rtg",
        "timesteps",
        "masks",
        "prompt_s",
        "prompt_a",
        "prompt_r",
        "prompt_rtg",
        "prompt_ts",
        "prompt_m",
        "instructions",
    ]

    def _print_shapes(model, batch, dt_batch, image_embeddings):
        log.info("[train_batch] === shapes (first train step) ===")
        log.info(
            "[train_batch] layout: [0-7] query (states,contexts,actions,rewards,dones,rtg,timesteps,masks); "
            "[8-13] prompt (prompt_s,a,r,rtg,ts,m); [14] instructions; [15] images/embeddings; "
            "[16] sample index (LIBERO index-backed: query_episode_id, ...)."
        )
        for i, name in enumerate(_BATCH_NAMES):
            if i >= len(batch):
                break
            x = batch[i]
            if isinstance(x, torch.Tensor):
                log.info("[train_batch]   {}: {} {}", name, x.dtype, tuple(x.shape))
            elif isinstance(x, (list, tuple)) and x and isinstance(x[0], torch.Tensor):
                log.info(
                    "[train_batch]   {}: list of {} tensors {}",
                    name,
                    len(x),
                    [tuple(t.shape) for t in x],
                )
            else:
                log.info(
                    "[train_batch]   {}: type={} len={}",
                    name,
                    type(x).__name__,
                    len(x) if hasattr(x, "__len__") else "?",
                )
        if len(batch) > 15 and batch[15] is not None:
            imgs = batch[15]
            if isinstance(imgs, (list, tuple)):
                for v, t in enumerate(imgs):
                    if isinstance(t, torch.Tensor):
                        log.info("[train_batch]   images view[{}]: {}", v, tuple(t.shape))
            elif isinstance(imgs, torch.Tensor):
                log.info("[train_batch]   image_embeddings (precomputed): {}", tuple(imgs.shape))
            else:
                log.info("[train_batch]   images/embeddings: {}", type(imgs).__name__)
        log.info("[train_batch]   --- DTBatch (prompt trimmed for RTG align in step) ---")
        log.info("[train_batch]   states: {}", tuple(dt_batch.states.shape))
        log.info("[train_batch]   actions: {}", tuple(dt_batch.actions.shape))
        if dt_batch.prompt and dt_batch.prompt[0] is not None:
            log.info(
                "[train_batch]   prompt (s,a,r,rtg,ts,m): {}",
                [tuple(p.shape) for p in dt_batch.prompt],
            )
        if image_embeddings is not None:
            log.info("[train_batch]   image_embeddings: {}", tuple(image_embeddings.shape))
        if dt_batch.instruction_indices is not None:
            log.info(
                "[train_batch]   instruction_indices: {}",
                tuple(dt_batch.instruction_indices.shape),
            )
        if len(batch) > 16 and batch[16] is not None:
            log.info("[train_batch]   --- sample index (first 2 rows) ---")
            for b_idx, row in enumerate(batch[16][:2]):
                if isinstance(row, dict):
                    log.info(
                        "[train_batch]   [{}] query_ep={} query_start={} query_len={} task_id={} "
                        "prompt_eps={} prompt_starts={} prompt_lens={}",
                        b_idx,
                        row.get("query_episode_id"),
                        row.get("query_start"),
                        row.get("query_len"),
                        row.get("task_id"),
                        row.get("prompt_episode_ids", []),
                        row.get("prompt_starts", []),
                        row.get("prompt_lens", []),
                    )
        if len(batch) > 13:
            pr, pm = batch[10], batch[13]
            if isinstance(pr, torch.Tensor) and isinstance(pm, torch.Tensor):
                log.info(
                    "[train_batch] --- prompt_r masked sum (first 2 batch rows, pre-trim tensors) ---"
                )
                B = min(2, pr.shape[0])
                for b in range(B):
                    pr_b = pr[b].float()
                    pm_b = pm[b].float()
                    if pr_b.dim() > 1:
                        masked = (pr_b.squeeze(-1) * pm_b).sum().item()
                    else:
                        masked = (pr_b * pm_b).sum().item()
                    log.info("[train_batch]   row {}: sum(prompt_r * prompt_m) = {:.4f}", b, masked)
                log.info(
                    "[train_batch] per-context env returns (prompt order): see [prompt_context_returns] "
                    "(first 4 dataset samples / workers)"
                )
        log.info("[train_batch] === end ===")

    def train_step_fn(model, batch):
        """One step: forward, MSE loss on actions. Batch: 15 elements (14 tensors + instructions); optional 16th = images."""
        (
            states,
            contexts,
            actions,
            rewards,
            dones,
            rtg,
            timesteps,
            masks,
            prompt_s,
            prompt_a,
            prompt_r,
            prompt_rtg,
            prompt_ts,
            prompt_m,
            instructions,
        ) = batch[:15]
        # Trim last step from all prompt tensors so lengths match
        prompt_rtg = prompt_rtg[:, :-1, :]
        prompt_s = prompt_s[:, :-1, :]
        prompt_a = prompt_a[:, :-1, :]
        prompt_r = prompt_r[:, :-1, :]
        prompt_ts = prompt_ts[:, :-1]
        prompt_m = prompt_m[:, :-1]
        prompt = (prompt_s, prompt_a, prompt_r, prompt_rtg, prompt_ts, prompt_m)

        # VLA-DT: instruction indices (one per sample) from task_instructions
        instruction_indices = None
        if model.use_language and task_list and instructions is not None:
            device = next(model.parameters()).device
            idx_list = [
                task_list.index(instr) if instr in task_list else 0 for instr in instructions
            ]
            instruction_indices = torch.tensor(idx_list, dtype=torch.long, device=device)

        # VLA-DT: image_embeddings from precomputed npz or from vision encoder (optional 16th element)
        image_embeddings = None
        if len(batch) > 15 and batch[15] is not None:
            if use_precomputed_embeddings:
                image_embeddings = batch[15]
            elif model.vision_encoder is not None:
                imgs = batch[15]
                if isinstance(imgs, (list, tuple)):
                    shapes_str = [tuple(t.shape) for t in imgs]
                    if not _vision_encoder_logged[0]:
                        log.info("Vision encoder input: {} views, shapes {}", len(imgs), shapes_str)
                else:
                    shapes_str = imgs.shape if hasattr(imgs, "shape") else type(imgs)
                    if not _vision_encoder_logged[0]:
                        log.info("Vision encoder input: {}", shapes_str)
                image_embeddings = model.vision_encoder(batch[15])
                if not _vision_encoder_logged[0]:
                    if image_embeddings is not None:
                        log.info("Vision encoder output: {}", tuple(image_embeddings.shape))
                    _vision_encoder_logged[0] = True

        dt_batch = DTBatch(
            states=states,
            contexts=contexts,
            actions=actions,
            returns_to_go=rtg,
            timesteps=timesteps,
            attention_mask=masks,
            prompt=prompt,
            image_embeddings=image_embeddings,
            instruction_indices=instruction_indices,
        )
        if not _train_batch_logged[0]:
            _print_shapes(model, batch, dt_batch, image_embeddings)
            _train_batch_logged[0] = True
        out = model(dt_batch)
        loss = out.loss if out.loss is not None else torch.tensor(0.0, device=states.device)
        with torch.no_grad():
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)

        # Batch stats for logging (min/max/mean return and in-context prompt length)
        batch_stats = _batch_stats(rewards, masks, prompt_m)
        return loss, grad_norm, batch_stats

    return train_step_fn


def _batch_stats(
    rewards: torch.Tensor,
    masks: torch.Tensor,
    prompt_m: torch.Tensor,
) -> dict:
    """Compute min/max/mean return and prompt length per batch for W&B."""
    # Return per sample: sum of rewards over valid (masked) steps. rewards (B,T,1), masks (B,T)
    valid_return = (rewards.squeeze(-1) * masks).sum(dim=1)
    n = valid_return.shape[0]
    if n == 0:
        return {}
    r_min = valid_return.min().cpu().item()
    r_max = valid_return.max().cpu().item()
    r_mean = valid_return.float().mean().cpu().item()
    # Prompt length per sample: number of valid steps in context
    prompt_len = prompt_m.sum(dim=1).float()
    pl_min = prompt_len.min().cpu().item()
    pl_max = prompt_len.max().cpu().item()
    pl_mean = prompt_len.mean().cpu().item()
    return {
        "batch/return_min": r_min,
        "batch/return_max": r_max,
        "batch/return_mean": r_mean,
        "batch/prompt_len_min": pl_min,
        "batch/prompt_len_max": pl_max,
        "batch/prompt_len_mean": pl_mean,
    }


def main():
    import argparse

    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=str, default="configs", help="Hydra config directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    parser.add_argument(
        "--export-only",
        type=str,
        default=None,
        help="Export inference artifact from this checkpoint",
    )
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

    train_log = Path(log_dir) / "train.log"
    log.add(train_log, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    _print_config(cfg)

    if args.export_only:
        model = build_model(cfg, state_dim, action_dim)
        load_checkpoint(args.export_only, model, device=device, weights_only=True)
        export_dir = run_dir / "artifacts" / "inference"
        export_dir.mkdir(parents=True, exist_ok=True)
        save_inference_artifact(str(export_dir), model, cfg, filename="model_export.pt", rank=0)
        log.info("Exported model_export.pt to {}", export_dir)
        return

    # Build data: D4RL HalfCheetah, LIBERO (libero_10 etc.), AntDir-style (dataset_task_*.pkl), or dummy
    data_root = paths.data_root
    data_dir = data_root / data_cfg.env_name
    if data_cfg.data_quality:
        data_dir = data_dir / data_cfg.data_quality
    if env_name in LIBERO_SUITES:
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
            reward_normalization=data_cfg.reward_normalization,
            reward_norm_constant=float(data_cfg.reward_norm_constant),
            reward_norm_epsilon=float(data_cfg.reward_norm_epsilon),
            reward_normalization_stats_path=data_cfg.reward_normalization_stats_path,
        )
        if trajectories:
            log.info(
                "Loaded {} HalfCheetah trajectories (mixed returns) from {}",
                len(trajectories),
                data_root,
            )

    if env_name == "ICRT-MT":
        config_path = data_root / "ICRT-MT" / "dataset_config.json"
        assert config_path.exists(), f"ICRT-MT config missing (validated earlier): {config_path}"
        from src.data.icrt_dataset import load_icrt_trajectories

        proprio_keys = data_cfg.proprio_keys or [
            "observation/cartesian_position",
            "observation/gripper_position",
        ]
        action_keys = data_cfg.action_keys or [
            "action/cartesian_position",
            "action/gripper_position",
        ]
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
        log.info(
            "Loaded {} ICRT-MT trajectories from {} (state_dim={}, action_dim={})",
            len(trajectories),
            config_path,
            state_dim,
            action_dim,
        )

    if env_name == "VD4RL":
        from src.data.vd4rl_loader import load_vd4rl_npz_trajectories

        vd4rl_dir = (
            paths.data_root
            / data_cfg.vd4rl_suite
            / data_cfg.vd4rl_task
            / data_cfg.vd4rl_split
            / data_cfg.vd4rl_pixel_size
        )
        trajectories, prompt_per_task = load_vd4rl_npz_trajectories(
            str(vd4rl_dir),
            max_episodes=data_cfg.vd4rl_max_episodes,
            obs_downsample=int(data_cfg.vd4rl_obs_downsample),
            store_images=bool(data_cfg.use_vision),
            shuffle=bool(data_cfg.vd4rl_shuffle_npz_order),
            seed=int(data_cfg.seed),
        )
        if trajectories:
            state_dim = int(trajectories[0]["observations"].shape[1])
            action_dim = int(trajectories[0]["actions"].shape[1])
            log.info(
                "VD4RL: inferred state_dim={}, action_dim={} (set model dims to match)",
                state_dim,
                action_dim,
            )
        else:
            log.error("VD4RL: no trajectories loaded from {}", vd4rl_dir)

    in_context_result = None
    if env_name in LIBERO_SUITES:
        import src.data.libero_dataset
        from src.data.sample_index import build_in_context_dataset

        in_context_result = build_in_context_dataset(
            "libero",
            str(data_root),
            data_cfg,
            device,
            state_dim,
            action_dim,
            collate_icl_batch,
        )
        if in_context_result is None:
            log.error(
                "LIBERO requires manifest.parquet + sample_index.parquet + episodes/. "
                "Run: python scripts/convert_libero_hdf5_to_dataset.py --input-dir <LIBERO-Cosmos-Policy>"
            )
            return
        dataset = in_context_result.dataset
        loader = in_context_result.loader
        state_mean = in_context_result.state_mean
        state_std = in_context_result.state_std

    if in_context_result is None and not trajectories:
        log.warning("No dataset found at {}", data_dir.resolve())
        return

    if in_context_result is None:
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
            randomize_num_context_trajectories=data_cfg.randomize_num_context_trajectories,
            context_sort_ascending=data_cfg.context_sort_ascending,
            context_sampling=data_cfg.context_sampling,
            max_total_prompt_length=data_cfg.max_total_prompt_length,
            max_prompt_trajectory_length=data_cfg.max_prompt_trajectory_length,
            context_subsample_strategy=data_cfg.context_subsample_strategy,
            context_style=data_cfg.context_style,
            lazy_dataset=data_cfg.lazy_dataset,
            max_training_examples=data_cfg.max_training_examples,
            task_instructions=task_instructions_from_loader
            if task_instructions_from_loader is not None
            else data_cfg.task_instructions,
            seed=data_cfg.seed,
            query_history_length=data_cfg.query_history_length,
            use_vision=data_cfg.use_vision,
            image_keys=data_cfg.image_keys or [],
        )
    state_mean = dataset.state_mean
    state_std = dataset.state_std
    log.info("Dataset size: {} segments, state_mean/std computed", len(dataset))
    if (
        data_cfg.use_vision
        and hasattr(dataset, "trajectories")
        and dataset.trajectories
        and not any(isinstance(t, dict) and t.get("images") for t in dataset.trajectories)
    ):
        log.warning(
            "use_vision=true but no trajectories have 'images'. "
            "Ensure data/ dataset has image columns (e.g. primary_images_jpeg, wrist_images_jpeg) from convert_libero_hdf5_to_dataset.py."
        )
    if in_context_result is None:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=data_cfg.batch_size,
            shuffle=True,
            num_workers=data_cfg.num_workers,
            collate_fn=collate_icl_batch,
        )
    _print_dataset_stats(
        dataset,
        loader,
        env_name=data_cfg.env_name,
        data_quality=data_cfg.data_quality or "",
        image_keys=data_cfg.image_keys or [],
        proprio_keys=data_cfg.proprio_keys or [],
        use_vision=data_cfg.use_vision,
    )

    if bool(cfg.experiment.save_training_sample_videos):
        from src.engine.training_debug_viz import save_training_sample_videos

        trs = getattr(dataset, "trajectories", None)
        has_images = bool(
            data_cfg.use_vision
            or (
                trs
                and any(isinstance(t, dict) and t.get("images") for t in trs[: min(32, len(trs))])
            )
        )
        if has_images:
            save_training_sample_videos(
                Path(run_dir),
                dataset,
                return_scale=float(data_cfg.return_scale),
                num_clips=int(cfg.experiment.num_training_sample_videos),
                fps=int(cfg.experiment.training_sample_video_fps),
            )
        else:
            log.info(
                "experiment.save_training_sample_videos=true but no trajectory images; skipping debug MP4s."
            )

    num_instructions = len(dataset.task_instructions) if dataset.task_instructions else None
    model = build_model(cfg, state_dim, action_dim, num_instructions=num_instructions).to(device)
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
        log.info(
            "W&B logging enabled (entity: {}, project: {})",
            sys_cfg.wandb_entity,
            sys_cfg.wandb_project,
        )

    global_step_start = 0
    best_metric_start = float("-inf")

    if args.resume:
        _, global_step_start, best_metric_start, _, _ = load_checkpoint(
            args.resume,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            restore_rng=True,
        )

    if args.eval_only:
        model.eval()
        logger.log_scalar("eval/return_mean", 0.0, global_step_start)
        logger.close()
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
        eval_mode = cfg.experiment.eval_context_mode
        # eval_context_k is overloaded by mode:
        # - prompt: how many context trajectories to sample (default: data.num_context_trajectories).
        # - zero_shot_adaptation: last-K env steps of the live trial in the prompt (default: cfg.model.max_length).
        # Do not use num_context_trajectories for zero-shot K — it wrongly matched e.g. 3 demos -> K=3.
        if eval_mode == "zero_shot_adaptation":
            if cfg.experiment.eval_context_k is not None:
                eval_k = cfg.experiment.eval_context_k
            else:
                ml = cfg.model.max_length
                eval_k = int(ml) if ml is not None else 20
        else:
            eval_k = cfg.experiment.eval_context_k or data_cfg.num_context_trajectories
        prompt_trajectories = None
        if eval_mode == "prompt" and hasattr(dataset, "trajectories") and dataset.trajectories:
            prompt_trajectories = sample_context_trajectories(
                dataset.trajectories,
                n=eval_k,
                ascending=True,
                sampling=data_cfg.context_sampling,
            )
        task_desc = (dataset.task_instructions or [None])[0] if dataset.task_instructions else None
        # Dual-camera stitch is LIBERO-only; Gymnasium (HalfCheetah, etc.) has a single render view.
        eval_render_both_views = bool(cfg.experiment.eval_render_both_views) and (
            env_name in LIBERO_SUITES
        )
        # inference_mode: no autograd graph. Without this, rollout chains ~max_episode_steps forwards
        # via actions_t = cat(..., action) and retains the full graph -> CUDA OOM.
        was_training = model.training
        model.eval()
        try:
            with torch.inference_mode():
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
                    save_video=cfg.experiment.save_eval_video,
                    eval_context_mode=eval_mode,
                    prompt_trajectories=prompt_trajectories,
                    eval_num_trials=cfg.experiment.eval_num_trials,
                    eval_context_k=eval_k,
                    eval_reward_source=cfg.experiment.eval_reward_source,
                    eval_reward_model=cfg.experiment.eval_reward_model,
                    total_prompt_len=dataset.total_prompt_len,
                    max_prompt_trajectory_length=dataset.max_prompt_trajectory_length,
                    context_subsample_strategy=dataset.context_subsample_strategy,
                    task_description=task_desc,
                    logger=logger,
                    eval_render_both_views=eval_render_both_views,
                    reward_normalization=data_cfg.reward_normalization,
                    reward_norm_constant=float(data_cfg.reward_norm_constant),
                    reward_norm_epsilon=float(data_cfg.reward_norm_epsilon),
                    reward_normalization_stats_path=data_cfg.reward_normalization_stats_path,
                    wandb_defer_step_commit=use_wandb,
                )
                if (
                    cfg.experiment.run_action_compare_eval
                    and hasattr(dataset, "trajectories")
                    and dataset.trajectories
                ):
                    action_metrics = run_action_compare_eval(
                        model=model,
                        trajectories=dataset.trajectories,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                        run_dir=run_dir,
                        step=step,
                        num_demos=cfg.experiment.num_action_compare_demos,
                        max_episode_steps=data_cfg.max_episode_steps,
                        scale=data_cfg.return_scale,
                        use_gt_action=True,
                        warm_train_steps=cfg.experiment.warm_train_steps,
                    )
                    metrics = {**metrics, **action_metrics}
        finally:
            if was_training:
                model.train()
        append_metrics_history(run_dir, step, metrics)
        return metrics

    log.info("Starting training loop (max_steps={})", cfg.experiment.max_steps)
    export_dir = str(run_dir / "artifacts" / "inference")
    train_step_fn = make_train_step_fn(
        dataset.task_instructions or [],
        use_precomputed_embeddings=cfg.data.use_precomputed_embeddings,
    )
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
    write_metrics_summary(
        run_dir,
        {
            "final_step": final_step,
            "best_metric": best_metric,
            "best_metric_name": cfg.experiment.best_metric_name,
        },
    )
    log.info("Training finished.")
    logger.close()


if __name__ == "__main__":
    main()
