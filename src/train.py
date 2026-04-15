"""
Single training entrypoint: start fresh, resume from checkpoint, eval-only, or save final export.
No notebook-only training logic. Use notebooks only for analysis.
"""

import os
import random
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

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
from src.config.schema import resolved_max_total_prompt_length
from src.data import collate_icl_batch, get_icl_trajectory_dataset
from src.data.d4rl_loader import format_data_quality_for_log, parse_halfcheetah_data_qualities
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

    trajectories = dataset.trajectories
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
    return_min = dataset.return_min
    return_max = dataset.return_max
    return_avg = dataset.return_avg
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


def _vd4rl_split_list(data_cfg) -> list:
    """Use vd4rl_splits when set; otherwise single vd4rl_split."""
    splits = OmegaConf.select(data_cfg, "vd4rl_splits", default=None)
    if splits is not None:
        raw = OmegaConf.to_container(splits, resolve=True)
        if isinstance(raw, list) and len(raw) > 0:
            return [str(x) for x in raw]
    single = OmegaConf.select(data_cfg, "vd4rl_split", default=None)
    if single is None:
        single = "random"
    return [str(single)]


def _trajectory_hdf5_paths_from_cfg(data_cfg) -> Optional[List[str]]:
    """
    ``data.trajectory_hdf5_paths`` (list of strings): flat v2 ``.h5`` inputs for ManiSkill ICL.

    None / empty is invalid for ``ManiSkill/`` — set explicit shard paths in the data config.
    """
    raw = OmegaConf.select(data_cfg, "trajectory_hdf5_paths", default=None)
    if raw is None:
        return None
    cont = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(cont, (list, tuple)) or len(cont) == 0:
        return None
    out = [str(x).strip() for x in cont if str(x).strip()]
    return out or None


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
    elif env_name.startswith("ManiSkill/"):
        from src.data.maniskill_io import resolve_maniskill_trajectory_paths

        ms_task = env_name.split("/", 1)[1]
        explicit = _trajectory_hdf5_paths_from_cfg(data_cfg)
        try:
            traj_paths = resolve_maniskill_trajectory_paths(
                paths.data_root, ms_task, explicit, repo_root=paths.repo_root
            )
        except (FileNotFoundError, ValueError) as e:
            d = Path(paths.data_root) / "maniskill" / ms_task.replace("/", "_").replace(" ", "_")
            raise type(e)(
                f"{e}\n"
                "Export shards with: python scripts/maniskill/ppo_train_icldata.py --env-id "
                f"{ms_task} ... (see docs/MANISKILL.md), then set data.trajectory_hdf5_paths to those "
                f"files (see configs/data/maniskill_pickcube.yaml). Expected layout under: {d}"
            ) from e
        for p in traj_paths:
            if not p.is_file():
                raise FileNotFoundError(f"ManiSkill trajectory path missing: {p}")
        log.info(
            "ManiSkill dataset: {} HDF5 file(s): {}",
            len(traj_paths),
            ", ".join(str(x) for x in traj_paths[:6]) + (" …" if len(traj_paths) > 6 else ""),
        )
    elif env_name == "VD4RL":
        for split in _vd4rl_split_list(data_cfg):
            p = (
                paths.data_root
                / data_cfg.vd4rl_suite
                / data_cfg.vd4rl_task
                / split
                / data_cfg.vd4rl_pixel_size
            )
            if not p.is_dir():
                raise FileNotFoundError(
                    f"V-D4RL data directory not found: {p}\n"
                    "Download datasets from the Google Drive linked in "
                    "https://github.com/conglu1997/v-d4rl and set paths.data_root to the folder that "
                    f"contains `{data_cfg.vd4rl_suite}/` (expected leaf dir: "
                    f".../{data_cfg.vd4rl_task}/<split>/{data_cfg.vd4rl_pixel_size}/ "
                    "with `*.npz` for 64px or `*.hdf5` for 84px DrQ shards)."
                )
            has_npz = any(p.glob("*.npz"))
            has_h5 = any(p.glob("*.hdf5"))
            if not has_npz and not has_h5:
                raise FileNotFoundError(
                    f"V-D4RL directory exists but has no *.npz or *.hdf5 files: {p}\n"
                    "64px (DreamerV2) uses .npz; 84px (DrQ-v2) uses .hdf5 (e.g. shard_*_reward_*.hdf5). "
                    "Install h5py for hdf5: uv pip install h5py"
                )
            log.info(
                "V-D4RL data directory found: {} split={} ({})",
                p,
                split,
                "npz" if has_npz else "hdf5",
            )


def _infer_hw_from_trajectory_images(
    trajectories: Optional[List[Any]],
) -> Optional[Tuple[int, int]]:
    """First (T,H,W,3) uint8 view in trajectory ``images`` list."""
    if not trajectories:
        return None
    for t in trajectories:
        if not isinstance(t, dict):
            continue
        imgs = t.get("images")
        if not isinstance(imgs, list) or not imgs:
            continue
        a = np.asarray(imgs[0])
        if a.ndim == 4 and a.shape[-1] == 3:
            return int(a.shape[1]), int(a.shape[2])
        if a.ndim == 4 and a.shape[1] == 3:
            return int(a.shape[2]), int(a.shape[3])
    return None


def resolve_vision_encoder_hw(
    cfg,
    trajectories: Optional[List[Any]],
) -> Tuple[int, int]:
    """
    Order: ``model.vision_encoder_img_size``, ``data.image_size``, then trajectory RGB shapes.
    """
    m = cfg.model
    raw = OmegaConf.select(m, "vision_encoder_img_size", default=None)
    if raw is not None:
        vis = OmegaConf.to_container(raw, resolve=True)
        if isinstance(vis, (list, tuple)) and len(vis) >= 2:
            return int(vis[0]), int(vis[1])
        raise ValueError(
            f"model.vision_encoder_img_size must be [H, W] with two ints when set; got {vis!r}"
        )
    d_raw = OmegaConf.select(cfg.data, "image_size", default=None)
    if d_raw is not None:
        img = OmegaConf.to_container(d_raw, resolve=True)
        if isinstance(img, (list, tuple)) and len(img) >= 2:
            return int(img[0]), int(img[1])
    hw = _infer_hw_from_trajectory_images(trajectories)
    if hw is not None:
        return hw
    raise ValueError(
        "Could not resolve vision (H, W): set model.vision_encoder_img_size=[H,W] or "
        "data.image_size=[H,W], or use trajectories with RGB under key 'images' "
        "(per-view arrays shaped (T, H, W, 3))."
    )


def resolve_sequence_token_layout(cfg: Any) -> str:
    """Model input layout: AD defaults to (state, action, reward) without RTG conditioning."""
    from omegaconf import OmegaConf

    raw = OmegaConf.select(cfg, "model.sequence_token_layout", default=None)
    if raw is not None:
        rs = str(raw).strip().lower().replace("-", "_")
        if rs and rs not in ("none", "null", "~"):
            return rs
    ctx = str(OmegaConf.select(cfg, "data.context_style", default="")).strip().lower()
    if ctx in ("algorithm_distillation", "ad", "ad_timeline"):
        return "state_action_reward"
    m = cfg.model
    return "rtg_state_action" if bool(m.condition_rtg) else "state_action"


def build_model(
    cfg,
    state_dim: int,
    action_dim: int,
    num_instructions: Optional[int] = None,
    trajectories: Optional[List[Any]] = None,
):
    m = cfg.model
    seq_layout = resolve_sequence_token_layout(cfg)
    if seq_layout == "state_action_reward":
        log.info(
            "sequence_token_layout=state_action_reward: (s,a,r) sequence, no RTG; "
            "experiment.eval_target_return is ignored during rollouts."
        )
    n_inner = m.n_inner or (4 * m.hidden_size)
    num_ctx = int(cfg.data.num_context_trajectories)
    enable_trial_index_embedding = bool(m.use_trial_index_embedding) and num_ctx > 1
    if m.use_trial_index_embedding and not enable_trial_index_embedding:
        log.info(
            "Trial index embedding off: requires data.num_context_trajectories > 1; got {}",
            num_ctx,
        )
    common = dict(
        state_dim=state_dim,
        act_dim=action_dim,
        hidden_size=m.hidden_size,
        context_dim=m.context_dim,
        num_context_trajectories=num_ctx,
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
        condition_rtg=(seq_layout == "rtg_state_action"),
        sequence_token_layout=seq_layout,
        use_trial_index_embedding=enable_trial_index_embedding,
        max_trial_embeddings=m.max_trial_embeddings,
    )
    if m.use_vision or m.use_language:
        use_precomputed = cfg.data.use_precomputed_embeddings
        precomputed_dim = m.precomputed_vision_embed_dim
        vision_encoder_img_size = None
        if m.use_vision:
            vision_encoder_img_size = resolve_vision_encoder_hw(cfg, trajectories)
            log.info(
                "vision_encoder_img_size={} (model override, data.image_size, or trajectory RGB)",
                vision_encoder_img_size,
            )
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
            vision_encoder_img_size=vision_encoder_img_size,
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
        "query_trial_idx",
        "prompt_s",
        "prompt_a",
        "prompt_r",
        "prompt_rtg",
        "prompt_ts",
        "prompt_m",
        "prompt_trial_idx",
        "instructions",
        "images",
    ]

    def _print_shapes(model, batch, dt_batch, image_embeddings):
        log.info("[train_batch] === shapes (first train step) ===")
        log.info(
            "[train_batch] layout: [0-7] query; [8] query_trial_idx; [9-14] prompt (s,a,r,rtg,ts,m); "
            "[15] prompt_trial_idx; [16] instructions; [17] images/embeddings; "
            "[18] sample index (LIBERO)."
        )
        prompt_t_len = 0
        if len(batch) > 9 and isinstance(batch[9], torch.Tensor) and batch[9].dim() >= 2:
            prompt_t_len = int(batch[9].shape[1])
        for i, name in enumerate(_BATCH_NAMES):
            if i >= len(batch):
                break
            if prompt_t_len == 0 and 9 <= i <= 15:
                continue
            x = batch[i]
            if x is None:
                log.info("[train_batch]   {}: None", name)
                continue
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
                try:
                    lx = len(x)
                except TypeError:
                    lx = "?"
                log.info(
                    "[train_batch]   {}: type={} len={}",
                    name,
                    type(x).__name__,
                    lx,
                )
        if prompt_t_len == 0:
            log.info("[train_batch]   prompt_*: (no ICL context, T=0 — skipped per-field lines)")
        if prompt_t_len > 0:
            log.info("[train_batch]   --- DTBatch (prompt + query) ---")
        else:
            log.info("[train_batch]   --- DTBatch (query only) ---")
        log.info("[train_batch]   states: {}", tuple(dt_batch.states.shape))
        log.info("[train_batch]   actions: {}", tuple(dt_batch.actions.shape))
        if dt_batch.prompt and dt_batch.prompt[0] is not None and dt_batch.prompt[0].shape[1] > 0:
            log.info(
                "[train_batch]   prompt (s,a,r,rtg,ts,m[,trial]): {}",
                [tuple(p.shape) for p in dt_batch.prompt],
            )
        if image_embeddings is not None:
            log.info("[train_batch]   image_embeddings: {}", tuple(image_embeddings.shape))
        if dt_batch.instruction_indices is not None:
            log.info(
                "[train_batch]   instruction_indices: {}",
                tuple(dt_batch.instruction_indices.shape),
            )
        if len(batch) > 18 and batch[18] is not None:
            log.info("[train_batch]   --- sample index (first 2 rows) ---")
            for b_idx, row in enumerate(batch[18][:2]):
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
        if len(batch) > 14:
            pr, pm = batch[11], batch[14]
            if (
                isinstance(pr, torch.Tensor)
                and isinstance(pm, torch.Tensor)
                and pr.dim() >= 2
                and pr.shape[1] > 0
            ):
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
        """One step: forward, MSE on actions. Batch: 17 slots (tensors + instructions); optional images at [17]."""
        (
            states,
            contexts,
            actions,
            rewards,
            dones,
            rtg,
            timesteps,
            masks,
            query_trial_idx,
            prompt_s,
            prompt_a,
            prompt_r,
            prompt_rtg,
            prompt_ts,
            prompt_m,
            prompt_trial_idx,
            instructions,
        ) = batch[:17]

        # Always pass trial id tensors from the dataset. MetaDecisionTransformer only applies
        # nn.Embedding when model.use_trial_index_embedding / embed_trial_idx is set; if that flag
        # is off, indices are ignored but must still be passed when the flag is on (otherwise
        # forward() substitutes zeros and trial conditioning is silently disabled).
        trial_batch = query_trial_idx.long()
        # Collate keeps a (B, 0, …) prompt for no-ICL runs; DTBatch uses None so forward sees “no prompt”.
        prompt = None
        if int(prompt_s.shape[1]) > 0:
            prompt = (
                prompt_s,
                prompt_a,
                prompt_r,
                prompt_rtg,
                prompt_ts,
                prompt_m,
                prompt_trial_idx.long(),
            )
        # VLA-DT: instruction indices (one per sample) from task_instructions
        instruction_indices = None
        if model.use_language and task_list and instructions is not None:
            device = next(model.parameters()).device
            idx_list = [
                task_list.index(instr) if instr in task_list else 0 for instr in instructions
            ]
            instruction_indices = torch.tensor(idx_list, dtype=torch.long, device=device)

        # VLA-DT: image_embeddings from precomputed npz or from vision encoder (optional [17])
        image_embeddings = None
        if len(batch) > 17 and batch[17] is not None:
            if use_precomputed_embeddings:
                image_embeddings = batch[17]
            elif model.vision_encoder is not None:
                imgs = batch[17]
                if isinstance(imgs, (list, tuple)):
                    shapes_str = [tuple(t.shape) for t in imgs]
                    if not _vision_encoder_logged[0]:
                        log.info("Vision encoder input: {} views, shapes {}", len(imgs), shapes_str)
                else:
                    shapes_str = (
                        tuple(imgs.shape) if isinstance(imgs, torch.Tensor) else type(imgs).__name__
                    )
                    if not _vision_encoder_logged[0]:
                        log.info("Vision encoder input: {}", shapes_str)
                image_embeddings = model.vision_encoder(batch[17])
                if not _vision_encoder_logged[0]:
                    if image_embeddings is not None:
                        log.info("Vision encoder output: {}", tuple(image_embeddings.shape))
                    _vision_encoder_logged[0] = True

        rew_batch = None
        if getattr(model, "_sequence_token_layout", "") == "state_action_reward":
            rew_batch = rewards
            if rew_batch.dim() == 2:
                rew_batch = rew_batch.unsqueeze(-1)
        dt_batch = DTBatch(
            states=states,
            contexts=contexts,
            actions=actions,
            returns_to_go=rtg,
            rewards=rew_batch,
            timesteps=timesteps,
            attention_mask=masks,
            trial_indices=trial_batch,
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

        # Batch stats for logging (min/max/mean return, RTG, and in-context prompt length)
        batch_stats = _batch_stats(rewards, masks, prompt_m, rtg)
        return loss, grad_norm, batch_stats

    return train_step_fn


def _batch_stats(
    rewards: torch.Tensor,
    masks: torch.Tensor,
    prompt_m: torch.Tensor,
    rtg: Optional[torch.Tensor] = None,
) -> dict:
    """Compute min/max/mean return, RTG (query), and prompt length per batch for W&B."""
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
    out: dict = {
        "batch/return_min": r_min,
        "batch/return_max": r_max,
        "batch/return_mean": r_mean,
        "batch/prompt_len_min": pl_min,
        "batch/prompt_len_max": pl_max,
        "batch/prompt_len_mean": pl_mean,
    }
    # RTG: per-timestep targets on the query segment; stats over masked positions only.
    if rtg is not None and rtg.numel() > 0:
        rtg_2d = rtg.squeeze(-1) if rtg.dim() == 3 else rtg
        if rtg_2d.shape == masks.shape:
            valid = masks > 0 if masks.dtype != torch.bool else masks
            flat = rtg_2d[valid]
            if flat.numel() > 0:
                out["batch/rtg_min"] = flat.min().cpu().item()
                out["batch/rtg_max"] = flat.max().cpu().item()
                out["batch/rtg_mean"] = flat.float().mean().cpu().item()
    return out


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
    if env_name.startswith("ManiSkill/"):
        log.info("Env {} (ManiSkill: state/action dims set from trajectory file)", env_name)
    else:
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
        model = build_model(cfg, state_dim, action_dim, trajectories=None)
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
        if env_name == "HalfCheetah-v2":
            _qs = parse_halfcheetah_data_qualities(data_cfg.data_quality)
            if len(_qs) == 1:
                data_dir = data_dir / _qs[0]
        else:
            data_dir = data_dir / str(data_cfg.data_quality)
    if env_name in LIBERO_SUITES:
        data_dir = data_root
    trajectories = []
    prompt_per_task = []
    task_instructions_from_loader = None
    prebuilt_dataset = None

    if env_name == "HalfCheetah-v2":
        from src.data.d4rl_loader import load_halfcheetah_trajectories

        trajectories, prompt_per_task = load_halfcheetah_trajectories(
            str(data_root),
            env_name=data_cfg.env_name,
            data_quality=data_cfg.data_quality,
        )
        if trajectories:
            log.info(
                "Loaded {} HalfCheetah trajectories (mixed returns) from {}",
                len(trajectories),
                data_root,
            )

    if env_name.startswith("ManiSkill/"):
        from src.data.maniskill_io import resolve_maniskill_trajectory_paths
        from src.data.ic_replay_buffer_hdf5 import load_ic_replay_buffer_bundle

        ms_task = env_name.split("/", 1)[1]
        explicit = _trajectory_hdf5_paths_from_cfg(data_cfg)
        traj_paths = resolve_maniskill_trajectory_paths(
            paths.data_root, ms_task, explicit, repo_root=paths.repo_root
        )
        _ms_ctx = str(data_cfg.context_style).strip().lower()
        _ms_ad = _ms_ctx in ("algorithm_distillation", "ad", "ad_timeline")
        from src.data.maniskill_state_filter import (
            apply_maniskill_vision_proprio_to_bundle,
            maniskill_task_from_env_name,
            vision_proprio_slice_for_task,
        )

        _ms_task = maniskill_task_from_env_name(env_name)
        _ms_obs_slice = (
            vision_proprio_slice_for_task(_ms_task) if bool(data_cfg.use_vision) else None
        )
        if _ms_ad:
            from src.data.ic_replay_buffer_dataset import ICReplayBufferDataset

            prebuilt_dataset = ICReplayBufferDataset(
                traj_paths,
                horizon=int(data_cfg.horizon),
                rtg_scale=float(data_cfg.rtg_scale),
                device=device,
                context_dim=int(data_cfg.context_dim),
                min_traj_len=int(data_cfg.min_trajectory_length),
                context_sort_ascending=bool(data_cfg.context_sort_ascending),
                use_vision=bool(data_cfg.use_vision),
                seed=int(data_cfg.seed),
                max_training_examples=int(data_cfg.max_training_examples),
                observation_slice=_ms_obs_slice,
            )
            state_dim = int(prebuilt_dataset.state_dim)
            action_dim = int(prebuilt_dataset.act_dim)
        else:
            trajectories, prompt_per_task = load_ic_replay_buffer_bundle(
                traj_paths,
                min_episode_length=None,
                log_summary=True,
            )
        if bool(data_cfg.use_vision):
            if prebuilt_dataset is None:
                if _ms_obs_slice is not None:
                    trajectories, prompt_per_task = apply_maniskill_vision_proprio_to_bundle(
                        trajectories, prompt_per_task, env_name
                    )
                    log.info(
                        "ManiSkill use_vision: state trimmed to proprio + tcp_pose only (task {})",
                        _ms_task,
                    )
                else:
                    log.warning(
                        "ManiSkill use_vision: no proprio trim layout for {}; using full state vector",
                        _ms_task,
                    )
            elif _ms_obs_slice is not None:
                log.info(
                    "ManiSkill use_vision: lazy AD applies proprio + tcp_pose state slice (task {})",
                    _ms_task,
                )
            else:
                log.warning(
                    "ManiSkill use_vision: no proprio trim layout for {}; lazy AD uses full state",
                    _ms_task,
                )
        if prebuilt_dataset is None:
            state_dim = int(trajectories[0]["observations"].shape[1])
            action_dim = int(trajectories[0]["actions"].shape[1])
        log.info(
            "Loaded ManiSkill data from {} file(s) (state_dim={}, action_dim={}, lazy_ad={})",
            len(traj_paths),
            state_dim,
            action_dim,
            bool(prebuilt_dataset is not None),
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

        # Pixel frames for debug MP4s / viz even when training on flattened state (use_vision=false).
        vd4rl_store_images = bool(data_cfg.use_vision) or bool(
            cfg.experiment.save_training_sample_videos
        )
        split_list = _vd4rl_split_list(data_cfg)
        cap = data_cfg.vd4rl_max_episodes
        if len(split_list) == 1:
            vd4rl_dir = (
                paths.data_root
                / data_cfg.vd4rl_suite
                / data_cfg.vd4rl_task
                / split_list[0]
                / data_cfg.vd4rl_pixel_size
            )
            trajectories, prompt_per_task = load_vd4rl_npz_trajectories(
                str(vd4rl_dir),
                max_episodes=cap,
                obs_downsample=int(data_cfg.vd4rl_obs_downsample),
                store_images=vd4rl_store_images,
                shuffle=bool(data_cfg.vd4rl_shuffle_npz_order),
                seed=int(data_cfg.seed),
            )
        else:
            merged: list = []
            for split in split_list:
                vd4rl_dir = (
                    paths.data_root
                    / data_cfg.vd4rl_suite
                    / data_cfg.vd4rl_task
                    / split
                    / data_cfg.vd4rl_pixel_size
                )
                trajs, _ = load_vd4rl_npz_trajectories(
                    str(vd4rl_dir),
                    max_episodes=None,
                    obs_downsample=int(data_cfg.vd4rl_obs_downsample),
                    store_images=vd4rl_store_images,
                    shuffle=bool(data_cfg.vd4rl_shuffle_npz_order),
                    seed=int(data_cfg.seed),
                )
                merged.extend(trajs)
                log.info(
                    "VD4RL: loaded {} trajectories from split={} ({})",
                    len(trajs),
                    split,
                    vd4rl_dir,
                )
            if cap is not None and len(merged) > int(cap):
                merged = merged[: int(cap)]
            trajectories = merged
            prompt_per_task = [sort_trajectories_by_return(merged, ascending=False)]
        if trajectories:
            state_dim = int(trajectories[0]["observations"].shape[1])
            action_dim = int(trajectories[0]["actions"].shape[1])
            log.info(
                "VD4RL: {} trajectories from {} split(s); state_dim={}, action_dim={}",
                len(trajectories),
                len(split_list),
                state_dim,
                action_dim,
            )
        else:
            log.error(
                "VD4RL: no trajectories loaded from splits {} under {}/{}/{}",
                split_list,
                paths.data_root,
                data_cfg.vd4rl_suite,
                data_cfg.vd4rl_task,
            )

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

    if in_context_result is None and prebuilt_dataset is None and not trajectories:
        log.warning("No dataset found at {}", data_dir.resolve())
        return

    if in_context_result is None:
        if prebuilt_dataset is not None:
            dataset = prebuilt_dataset
        else:
            total_plen = resolved_max_total_prompt_length(data_cfg)
            if data_cfg.max_total_prompt_length is None:
                log.info(
                    "data.max_total_prompt_length unset -> using {} "
                    "(max_episode_steps * num_context_trajectories)",
                    total_plen,
                )
            total_epi_per_task = max(1, len(trajectories) // max(1, data_cfg.num_train_tasks))
            # One shared pool (ManiSkill, single-split VD4RL, etc.) but many task_id buckets: replicate
            # so prompt_trajectories_per_task[task_id] exists for every traj_idx (see dataset._get_one_sample).
            if trajectories and prompt_per_task:
                max_task_id = (len(trajectories) - 1) // total_epi_per_task
                need_slots = max_task_id + 1
                if len(prompt_per_task) == 1 and need_slots > 1:
                    pool0 = prompt_per_task[0]
                    prompt_per_task = [pool0 for _ in range(need_slots)]
                    log.info(
                        "Replicated single prompt pool -> {} task slots (task_id 0..{}; "
                        "total_epi_per_task={})",
                        need_slots,
                        max_task_id,
                        total_epi_per_task,
                    )
            dataset = get_icl_trajectory_dataset(
                trajectories=trajectories,
                horizon=data_cfg.horizon,
                max_episode_steps=data_cfg.max_episode_steps,
                min_traj_len=int(data_cfg.min_trajectory_length),
                rtg_scale=float(data_cfg.rtg_scale),
                device=device,
                prompt_trajectories_per_task=prompt_per_task,
                context_dim=data_cfg.context_dim,
                state_dim=state_dim,
                act_dim=action_dim,
                prompt_length=data_cfg.prompt_length,
                total_epi_per_task=total_epi_per_task,
                num_context_trajectories=data_cfg.num_context_trajectories,
                randomize_num_context_trajectories=data_cfg.randomize_num_context_trajectories,
                context_sort_ascending=data_cfg.context_sort_ascending,
                context_sampling=data_cfg.context_sampling,
                max_total_prompt_length=total_plen,
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
        data_quality=(
            format_data_quality_for_log(data_cfg.data_quality) if data_cfg.data_quality else ""
        ),
        image_keys=data_cfg.image_keys or [],
        proprio_keys=data_cfg.proprio_keys or [],
        use_vision=data_cfg.use_vision,
    )

    if bool(cfg.experiment.save_training_sample_videos):
        from src.engine.training_debug_viz import save_training_sample_videos

        trs = dataset.trajectories
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
                rtg_scale=float(data_cfg.rtg_scale),
                num_clips=int(cfg.experiment.num_training_sample_videos),
                fps=int(cfg.experiment.training_sample_video_fps),
            )
        else:
            log.info(
                "experiment.save_training_sample_videos=true but no trajectory images; skipping debug MP4s."
            )

    num_instructions = len(dataset.task_instructions) if dataset.task_instructions else None
    model = build_model(
        cfg,
        state_dim,
        action_dim,
        num_instructions=num_instructions,
        trajectories=dataset.trajectories or None,
    ).to(device)
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

    ms_eval_env_cache_holder: dict[str, Any] = {}

    def eval_fn(step):
        exp_e = cfg.experiment
        num_rollouts = int(exp_e.num_eval_rollouts)
        eval_mode = exp_e.eval_context_mode
        # eval_context_k is overloaded by mode:
        # - prompt: how many context trajectories to sample (default: data.num_context_trajectories).
        # - zero_shot_adaptation: last-K env steps of the live trial in the prompt (default: cfg.model.max_length).
        # Do not use num_context_trajectories for zero-shot K — it wrongly matched e.g. 3 demos -> K=3.
        if eval_mode == "zero_shot_adaptation":
            if exp_e.eval_context_k is not None:
                eval_k = exp_e.eval_context_k
            else:
                ml = cfg.model.max_length
                eval_k = int(ml) if ml is not None else 20
        else:
            eval_k = exp_e.eval_context_k or data_cfg.num_context_trajectories
        eval_query_window = (
            int(data_cfg.query_history_length)
            if data_cfg.query_history_length is not None
            else int(data_cfg.horizon)
        )
        _ctx_st = str(data_cfg.context_style).strip().lower()
        if _ctx_st in ("algorithm_distillation", "ad", "ad_timeline"):
            _train_k = int(data_cfg.horizon)
            if int(eval_query_window) != _train_k:
                log.info(
                    "algorithm_distillation: eval query_window K={} (train K=data.horizon={}); "
                    "ensure model.max_length >= {}",
                    eval_query_window,
                    _train_k,
                    max(int(eval_query_window), _train_k),
                )
        prompt_trajectories = None
        if eval_mode == "prompt" and dataset.trajectories:
            prompt_trajectories = sample_context_trajectories(
                dataset.trajectories,
                n=eval_k,
                ascending=True,
                sampling=data_cfg.context_sampling,
            )
        task_desc = (dataset.task_instructions or [None])[0] if dataset.task_instructions else None
        eval_rollout_env = env_name
        if env_name == "VD4RL":
            override_eval = OmegaConf.select(data_cfg, "eval_env_name", default=None)
            if override_eval is not None and str(override_eval).strip():
                eval_rollout_env = str(override_eval)
            else:
                eval_rollout_env = f"VD4RL/dmc/{data_cfg.vd4rl_task}"
        minari_halfcheetah_id = None
        if "halfcheetah" in str(eval_rollout_env).lower():
            from src.envs.minari_halfcheetah_eval import resolve_minari_halfcheetah_eval_id

            minari_halfcheetah_id = resolve_minari_halfcheetah_eval_id(
                ",".join(parse_halfcheetah_data_qualities(data_cfg.data_quality))
            )
        vd4rl_px = None
        vd4rl_ds = None
        if str(eval_rollout_env).startswith("VD4RL/dmc/"):
            px_str = str(data_cfg.vd4rl_pixel_size).replace("px", "").strip()
            vd4rl_px = int(px_str) if px_str else 64
            vd4rl_ds = int(data_cfg.vd4rl_obs_downsample)
        _er_l = str(eval_rollout_env).lower()
        d4rl_score_ref = None
        if minari_halfcheetah_id is not None or (
            "halfcheetah" in _er_l and not _er_l.startswith("vd4rl/")
        ):
            from src.envs.d4rl_normalized_score import MUJOCO_HALFCHEETAH_D4RL_REF

            d4rl_score_ref = MUJOCO_HALFCHEETAH_D4RL_REF
        eval_target_returns_list = OmegaConf.select(
            cfg, "experiment.eval_target_returns", default=None
        )
        if eval_target_returns_list is not None:
            eval_target_returns_list = [float(x) for x in list(eval_target_returns_list)]
        eval_scene_seeds_list = OmegaConf.select(cfg, "experiment.eval_scene_seeds", default=None)
        if eval_scene_seeds_list is not None:
            eval_scene_seeds_list = [int(x) for x in list(eval_scene_seeds_list)]
        randomize_scene_between_trials = bool(
            OmegaConf.select(cfg, "experiment.randomize_scene_between_trials", default=False)
        )
        # Dual-camera stitch is LIBERO-only; Gymnasium (HalfCheetah, etc.) has a single render view.
        eval_render_both_views = bool(cfg.experiment.eval_render_both_views) and (
            env_name in LIBERO_SUITES
        )
        ms_sim = OmegaConf.select(data_cfg, "maniskill_sim_backend", default=None)
        ms_rm = OmegaConf.select(data_cfg, "maniskill_reward_mode", default=None)
        ms_cm = OmegaConf.select(data_cfg, "maniskill_control_mode", default=None)
        if isinstance(ms_sim, str) and not ms_sim.strip():
            ms_sim = None
        if isinstance(ms_rm, str) and not ms_rm.strip():
            ms_rm = None
        if isinstance(ms_cm, str) and not ms_cm.strip():
            ms_cm = None
        ms_state_slice = None
        if str(eval_rollout_env).startswith("ManiSkill/") and bool(data_cfg.use_vision):
            from src.data.maniskill_state_filter import (
                maniskill_task_from_env_name,
                vision_proprio_slice_for_task,
            )

            _tid = maniskill_task_from_env_name(str(eval_rollout_env))
            ms_state_slice = vision_proprio_slice_for_task(_tid)
        if str(eval_rollout_env).startswith("ManiSkill/"):
            if "cache" not in ms_eval_env_cache_holder:
                from src.envs.eval_gym import ManiSkillEvalEnvCache

                ms_eval_env_cache_holder["cache"] = ManiSkillEvalEnvCache()
            _ms_eval_env_cache = ms_eval_env_cache_holder["cache"]
        else:
            _ms_eval_env_cache = None
        # inference_mode: no autograd graph. Without this, rollout chains ~max_episode_steps forwards
        # via actions_t = cat(..., action) and retains the full graph -> CUDA OOM.
        was_training = model.training
        model.eval()
        try:
            with torch.inference_mode():
                metrics = run_rollouts_and_save_viz(
                    model=model,
                    env_name=eval_rollout_env,
                    state_mean=state_mean,
                    state_std=state_std,
                    device=device,
                    run_dir=run_dir,
                    step=step,
                    num_rollouts=num_rollouts,
                    max_episode_steps=data_cfg.max_episode_steps,
                    rtg_scale=float(data_cfg.rtg_scale),
                    save_video=exp_e.save_eval_video,
                    eval_context_mode=eval_mode,
                    prompt_trajectories=prompt_trajectories,
                    eval_num_trials=exp_e.eval_num_trials,
                    eval_context_k=eval_k,
                    eval_reward_source=exp_e.eval_reward_source,
                    eval_reward_model=exp_e.eval_reward_model,
                    total_prompt_len=dataset.total_prompt_len,
                    max_prompt_trajectory_length=dataset.max_prompt_trajectory_length,
                    context_subsample_strategy=dataset.context_subsample_strategy,
                    context_style=str(data_cfg.context_style),
                    task_description=task_desc,
                    logger=logger,
                    eval_render_both_views=eval_render_both_views,
                    wandb_defer_step_commit=use_wandb,
                    vd4rl_eval_pixel_hw=vd4rl_px,
                    vd4rl_eval_obs_downsample=vd4rl_ds,
                    vd4rl_eval_seed=int(data_cfg.seed),
                    eval_target_return=OmegaConf.select(
                        cfg, "experiment.eval_target_return", default=None
                    ),
                    eval_target_returns=eval_target_returns_list,
                    num_context_trajectories=int(data_cfg.num_context_trajectories),
                    query_window=eval_query_window,
                    minari_halfcheetah_dataset_id=minari_halfcheetah_id,
                    num_eval_rollout_videos=exp_e.num_eval_rollout_videos,
                    eval_video_max_trials=OmegaConf.select(
                        cfg, "experiment.eval_video_max_trials", default=10
                    ),
                    d4rl_score_ref=d4rl_score_ref,
                    maniskill_sim_backend=ms_sim
                    if str(eval_rollout_env).startswith("ManiSkill/")
                    else None,
                    maniskill_reward_mode=ms_rm
                    if str(eval_rollout_env).startswith("ManiSkill/")
                    else None,
                    maniskill_control_mode=ms_cm
                    if str(eval_rollout_env).startswith("ManiSkill/")
                    else None,
                    maniskill_state_obs_slice=ms_state_slice,
                    eval_scene_seeds=eval_scene_seeds_list,
                    randomize_scene_between_trials=randomize_scene_between_trials,
                    maniskill_eval_env_cache=_ms_eval_env_cache,
                )
                if cfg.experiment.run_action_compare_eval and dataset.trajectories:
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
                        scale=float(data_cfg.rtg_scale),
                        use_gt_action=True,
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
    try:
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
    finally:
        _ms_c = ms_eval_env_cache_holder.pop("cache", None)
        if _ms_c is not None:
            _ms_c.close()
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
