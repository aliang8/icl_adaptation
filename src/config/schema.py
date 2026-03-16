"""Typed config schema (OmegaConf structured configs / dataclasses)."""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """Decision transformer / in-context model."""

    state_dim: int = 27
    act_dim: int = 8
    context_dim: int = 16
    hidden_size: int = 128
    max_length: Optional[int] = 20
    max_ep_len: int = 200
    n_layer: int = 3
    n_head: int = 1
    n_inner: Optional[int] = None  # 4 * hidden_size if None
    activation_function: str = "relu"
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    n_positions: int = 1024
    action_tanh: bool = True
    # VLA-DT (vision-language-action): set true in model=vla_dt
    use_vision: bool = False
    use_language: bool = False
    num_views: int = 2
    image_embed_dim: int = 256
    # Transformer backbone: "gpt2" (custom) or "llama2" (HuggingFace pretrained)
    transformer_backbone: str = "gpt2"
    llama_model_name: Optional[str] = None  # e.g. "meta-llama/Llama-2-7b-hf"; used when transformer_backbone=llama2
    # Vision encoder type: "patch" (trainable per-patch), "crossmae" (ViT per-patch), "dinov2"/"dinov3" (1 emb per image), "paligemma" (SigLIP)
    vision_encoder_type: str = "patch"
    vision_encoder_pool: bool = True  # if True, output 1 embedding per image (for fusion); if False, per-patch
    # ICRT-style attention pooling over patch tokens (learned query) instead of mean; used for crossmae and optionally patch/dinov2
    vision_encoder_attention_pool: bool = False
    # ICRT-style: only compute action loss on the query segment (default True). If False, loss on prompt + query.
    query_loss_only: bool = True


@dataclass
class DataConfig:
    """Dataset and dataloader."""

    env_name: str = "AntDir-v0"
    data_quality: str = "medium"
    data_dir: str = "datasets"
    horizon: int = 20
    # Query = last K steps of current trajectory; K=1 = OpenVLA-style; None = use horizon
    query_history_length: Optional[int] = None
    prompt_length: int = 5
    return_scale: float = 500.0
    batch_size: int = 128
    num_workers: int = 0
    # tasks
    num_tasks: int = 50
    num_train_tasks: int = 45
    max_episode_steps: int = 200
    # context (in-context trajectories)
    context_horizon: int = 4
    context_dim: int = 16
    context_hidden_dim: int = 128
    num_context_trajectories: int = 1
    context_sort_ascending: bool = True
    context_sampling: str = "random"
    max_total_prompt_length: Optional[int] = None
    # full_trajectory only: max steps per context trajectory (each demo capped to this; then concatenated and capped by max_total_prompt_length)
    max_prompt_trajectory_length: Optional[int] = None
    # "subsampled" = prompt_length steps per context traj; "full_trajectory" = full traj per demo (capped per traj, then total)
    context_style: str = "subsampled"
    lazy_dataset: bool = True
    max_training_examples: int = 500_000
    task_instructions: Optional[List[str]] = None
    # ICRT-style: language + multi-view images (robot manipulation)
    dataset_config_json: Optional[str] = None
    use_vision: bool = False
    use_language: bool = False
    image_keys: Optional[List[str]] = None
    image_size: Optional[List[int]] = None
    proprio_keys: Optional[List[str]] = None
    action_keys: Optional[List[str]] = None
    min_trajectory_length: int = 10
    max_trajectory_length: int = 450
    # LIBERO-Cosmos
    libero_manifest: Optional[str] = None
    libero_repo_id: str = "nvidia/LIBERO-Cosmos-Policy"
    seed: int = 0


@dataclass
class OptimConfig:
    """Optimizer and scheduler."""

    lr: float = 5e-5
    weight_decay: float = 1e-4
    warmup_steps: int = 10000
    grad_clip_norm: float = 0.25
    scheduler: str = "linear_warmup"  # linear_warmup | cosine | none


@dataclass
class PathsConfig:
    """Path config: resolved to pathlib.Path at runtime. Override paths.data_root etc. from CLI."""

    data_root: str = "datasets"
    output_root: str = "outputs"
    repo_root: str = "."


@dataclass
class SystemConfig:
    """Runtime: device, seed, distributed, logging."""

    device: str = "cuda:0"
    seed: int = 412
    deterministic: bool = True
    # run directory: outputs/<project_name>/<date>/<run_name>__seed_<X>__<hash>/
    output_dir: str = "outputs"
    project_name: str = "icl_adaptation"
    # run_name set from CLI or config; used with seed and git hash for run slug
    save_dir: str = (
        "outputs/checkpoints"  # deprecated when run_dir used; ckpts live under run_dir/ckpts
    )
    # distributed (rank 0 saves)
    world_size: int = 1
    rank: int = 0
    # W&B (override with --wandb or system.use_wandb=true)
    use_wandb: bool = False
    wandb_project: str = "icl_adaptation"
    wandb_entity: str = "clvr"


@dataclass
class ExperimentConfig:
    """Training loop: steps, eval, checkpoint policy."""

    max_steps: int = 500_000
    eval_every_steps: int = 1000
    num_eval_episodes: int = 5
    num_eval_rollouts: int = 5  # number of env rollouts per eval (for real eval)
    save_eval_video: bool = False  # if True, wrap eval env with RecordVideo and save to viz/samples/step_XXX/videos/
    run_action_compare_eval: bool = False  # if True, plot predicted vs GT actions on demos to viz/action_compare/
    num_action_compare_demos: int = 3  # number of trajectories to use for action-comparison plots
    warm_train_steps: int = 70_000
    zero_shot: bool = False
    # checkpoint types
    save_latest_every_steps: Optional[int] = 5000
    save_best: bool = True
    save_periodic_every_steps: Optional[int] = 25000
    # metric for best
    best_metric_name: str = "eval/return_mean"
    best_metric_mode: str = "max"  # max or min
    # export
    export_final: bool = True


@dataclass
class AppConfig:
    """Top-level composed config."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    # run metadata (set at runtime)
    run_name: Optional[str] = None
    resume: Optional[str] = None
    eval_only: bool = False
    export_only: Optional[str] = None
