"""Typed config schema (OmegaConf structured configs / dataclasses)."""

from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class ModelConfig:
    """Decision transformer / in-context model."""

    state_dim: int = 27
    act_dim: int = 8
    context_dim: int = 16
    hidden_size: int = 128
    max_length: Optional[int] = 20
    max_ep_len: int = 6000
    n_layer: int = 3
    n_head: int = 1
    n_inner: Optional[int] = None  # 4 * hidden_size if None
    activation_function: str = "relu"
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    n_positions: int = 20000
    action_tanh: bool = True
    # VLA-DT (vision-language-action): set true in model=vla_dt
    use_vision: bool = False
    use_language: bool = False
    # If true, fuse language (instruction) into state; default false = no language conditioning.
    use_language_input: bool = False
    num_views: int = 2
    image_embed_dim: int = 256
    # Transformer backbone: "gpt2" (custom) or "llama2" (HuggingFace pretrained)
    transformer_backbone: str = "gpt2"
    llama_model_name: Optional[str] = (
        None  # e.g. "meta-llama/Llama-2-7b-hf"; used when transformer_backbone=llama2
    )
    # Vision encoder type: "patch" (trainable per-patch), "crossmae" (ViT per-patch), "dinov2"/"dinov3" (1 emb per image), "paligemma" (SigLIP)
    vision_encoder_type: str = "patch"
    vision_encoder_pool: bool = (
        True  # if True, output 1 embedding per image (for fusion); if False, per-patch
    )
    # ICRT-style attention pooling over patch tokens (learned query) instead of mean; used for crossmae and optionally patch/dinov2
    vision_encoder_attention_pool: bool = False
    # Freeze vision encoder (no gradients); only vision_proj and rest of model are trained
    freeze_vision_encoder: bool = False
    # Process vision encoder in chunks of this many images (None = no chunking). Default 8 reduces OOM for DINOv2.
    vision_encoder_chunk_size: Optional[int] = 8
    # Patch/crossmae (H, W); null -> infer from data.image_size or trajectory RGB shapes.
    vision_encoder_img_size: Optional[List[int]] = None
    # When data.use_precomputed_embeddings=true: vision encoder is not loaded at train time; set this to the embedding dim (e.g. 1536 for dinov2 2 views). Required for inference to load the encoder.
    precomputed_vision_embed_dim: Optional[int] = None
    # ICRT-style: attention over [view0, view1, proprio] (or [view0, proprio] if 1 view) to fuse vision + proprio. Default True.
    vision_proprio_attention_fusion: bool = True
    # ICRT-style: only compute action loss on the query segment (default True). If False, loss on prompt + query.
    query_loss_only: bool = True
    # Predict return-to-go and next state; default False = only predict actions.
    predict_returns: bool = False
    predict_state: bool = False
    # Condition on return-to-go in the input (RTG embedding in sequence). If False, input is (state, action) only.
    condition_rtg: bool = True
    # Per-timestep sequence layout: null -> from condition_rtg, or ``state_action_reward`` when data.context_style is algorithm_distillation.
    # ``rtg_state_action`` | ``state_action`` | ``state_action_reward`` (s,a,r tokens; no RTG at train or eval).
    sequence_token_layout: Optional[str] = None
    # If True, add learned trial-index embeddings when data.num_context_trajectories > 1 (no-op if N<=1).
    use_trial_index_embedding: bool = True
    # Embedding table size; index 0 = padding, 1+ = trial ids (clamped to [0, max_trial_embeddings - 1]).
    max_trial_embeddings: int = 64


@dataclass
class DataConfig:
    """Dataset and dataloader."""

    env_name: str = "AntDir-v0"
    # HalfCheetah multi-pool: use Hydra list [medium,medium_expert] or comma-separated string.
    data_quality: Union[str, List[str]] = "medium"
    data_dir: str = "all_datasets"
    horizon: int = 20
    # ICL / subsampled / full_trajectory: last K steps of the query trajectory; K=1 = OpenVLA-style; null = horizon.
    # algorithm_distillation: training window on the concat timeline is always ``horizon``; if ``query_history_length``
    # is set, eval rollouts use that as ``get_action`` history length (else eval uses ``horizon``).
    query_history_length: Optional[int] = None
    # Used only when context_style=subsampled (steps per context trajectory); ignored when context_style=full_trajectory
    prompt_length: int = 5
    # DT RTG tokens: cumsum(env rewards) / rtg_scale
    rtg_scale: float = 1.0
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
    # N<=0: no ICL prompts; model also uses proprio-only state embedding (ignores per-timestep batch.contexts).
    # If true, each sample uses m ~ Uniform{0..N} prior demos (N=num_context_trajectories); query window unchanged.
    randomize_num_context_trajectories: bool = True
    context_sort_ascending: bool = True
    context_sampling: str = "random"
    # Omit/null -> resolved_max_total_prompt_length (0 if num_context_trajectories<=0 else eps*n)
    max_total_prompt_length: Optional[int] = None
    # Per context demo: null = use full trajectory (within strategy); int = last N steps of each demo before concat
    max_prompt_trajectory_length: Optional[int] = None
    # How to subsample each context trajectory: "none" (full trajectory) | "last" | "uniform" | "random"
    context_subsample_strategy: str = "none"
    # "subsampled" | "full_trajectory" | "algorithm_distillation" (sorted concat timeline, H-step windows, no ICL prompt)
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
    # ManiSkill / ICL replay buffer: **required** non-empty list of shard ``.h5`` / ``.hdf5`` paths
    # (order = concat order for eager loaders). Resolved via ``paths.data_root``, cwd, or absolute.
    trajectory_hdf5_paths: Optional[List[str]] = None
    # ICRT-MT length bounds; also ICL HDF5 load + AD timeline: keep episodes with T >= this
    min_trajectory_length: int = 10
    max_trajectory_length: int = 450
    # LIBERO-Cosmos
    libero_repo_id: str = "nvidia/LIBERO-Cosmos-Policy"
    # When true, load precomputed embeddings from episodes/{id}/embeddings.npz (run precompute_libero_embeddings.py first)
    use_precomputed_embeddings: bool = False
    seed: int = 0
    # V-D4RL (https://github.com/conglu1997/v-d4rl): leaf dir data_root/suite/task/split/pixel_size with *.npz (64px) or *.hdf5 (84px)
    vd4rl_suite: str = "main"
    vd4rl_task: str = "walker_walk"
    vd4rl_split: str = "random"
    # If set (non-empty), load and merge trajectories from each split dir; ignores vd4rl_split for loading.
    vd4rl_splits: Optional[List[str]] = None
    vd4rl_pixel_size: str = "64px"
    vd4rl_max_episodes: Optional[int] = None
    vd4rl_obs_downsample: int = 16
    vd4rl_shuffle_npz_order: bool = False
    # Rollout env id when different from env_name (e.g. env_name=VD4RL for data, eval on dm_control).
    # Use VD4RL/dmc/walker_walk (see src/envs/vd4rl_eval_env.py). Null = default VD4RL/dmc/{vd4rl_task}.
    eval_env_name: Optional[str] = None


def resolved_max_total_prompt_length(data_cfg: DataConfig) -> int:
    """
    Total prompt timesteps after concatenating context demos (then pad/trim to this length).

    If ``data.max_total_prompt_length`` is set, returns that value.
    If ``num_context_trajectories <= 0`` and the override is unset, returns **0** (no prompt;
    query-only sequences use ``horizon`` / ``model.max_length`` only — no filler prompt padding).
    Otherwise returns ``max_episode_steps * num_context_trajectories`` (prompt only; query is separate).
    """
    if data_cfg.max_total_prompt_length is not None:
        return int(data_cfg.max_total_prompt_length)
    n = int(data_cfg.num_context_trajectories)
    if n <= 0:
        return 0
    eps = int(data_cfg.max_episode_steps)
    return eps * n


@dataclass
class OptimConfig:
    """Optimizer and scheduler."""

    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 10000
    grad_clip_norm: float = 0.25
    scheduler: str = "linear_warmup"  # linear_warmup | cosine | none


@dataclass
class PathsConfig:
    """Path config: resolved to pathlib.Path at runtime. Override paths.data_root etc. from CLI."""

    data_root: str = "all_datasets"
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
    run_name: Optional[str] = None  # CLI --run-name or override system.run_name=...
    save_dir: str = (
        "outputs/checkpoints"  # default; checkpoints are stored under run_dir/ckpts during training
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
    eval_every_steps: int = 5000
    # prompt / no-prompt: one env episode per rollout. zero_shot_adaptation: one adaptation *session* per rollout; trials are inside the session (eval_num_trials).
    num_eval_rollouts: int = 5
    eval_context_mode: str = "zero_shot_adaptation"  # prompt | zero_shot_adaptation
    eval_num_trials: int = 5  # zero_shot only: sequential in-session trials per rollout (ignored for prompt mode for this count)
    eval_context_k: Optional[int] = None
    # If set: prompt mode = this many context trajectories; zero_shot_adaptation = last-K env steps
    # in the live-trial prompt. If null: prompt defaults to data.num_context_trajectories; zero_shot to model.max_length.
    eval_reward_source: str = (
        "env"  # "env" | "reward_model" (for zero_shot_adaptation return used to sort context)
    )
    eval_reward_model: Optional[str] = (
        None  # e.g. "roboreward_8b" | "robometer_4b"; used when eval_reward_source=reward_model
    )
    # Target future cumulative return G (env reward units) for eval RTG init (token G/rtg_scale). None = token 1.0.
    eval_target_return: Optional[float] = None
    # Zero-shot only: explicit per-trial G; length must equal eval_num_trials. If set, overrides the
    # scalar schedule below. If null and eval_num_trials>1 with eval_target_return set, trial i gets
    # (i+1)/K * G (evenly spaced up to full G).
    eval_target_returns: Optional[List[float]] = None
    save_eval_video: bool = (
        False  # if True, wrap eval env with RecordVideo and save to viz/samples/step_XXX/videos/
    )
    # Cap how many rollouts/trials get frame capture + MP4 (None = all). Saves disk and W&B payload.
    num_eval_rollout_videos: Optional[int] = None
    # Zero-shot only: max in-session trial columns in stitched eval video (uniform subsample of trial
    # index). None = no cap. Summary plots and metrics always use every trial.
    eval_video_max_trials: Optional[int] = 10
    # LIBERO: stitch primary and wrist side-by-side in eval videos when both exist.
    eval_render_both_views: bool = True
    # Eval ``env.reset(seed=...)`` pool (Gymnasium / ManiSkill). None or empty: ``run_rollouts_and_save_viz``
    # fills a deterministic list sized to ``num_eval_rollouts`` (or × ``eval_num_trials`` when zero-shot
    # and ``randomize_scene_between_trials``). Explicit list: indices cycle if shorter than needed.
    eval_scene_seeds: Optional[List[int]] = None
    # Zero-shot with multiple in-session trials: if False (default), every trial in a session uses
    # the same reset seed so the scene matches across trials; if True, seed changes per trial.
    # Single-trial prompt/no-prompt rollouts: always one seed per rollout index from the pool (or legacy step+ep).
    randomize_scene_between_trials: bool = False
    run_action_compare_eval: bool = (
        False  # if True, plot predicted vs GT actions on demos to viz/action_compare/
    )
    num_action_compare_demos: int = 3  # number of trajectories to use for action-comparison plots
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
    # Vision / image datasets: save MP4s of query windows + RTG/mask at train start (run_dir/viz/training_sample_debug/)
    save_training_sample_videos: bool = True
    num_training_sample_videos: int = 3
    training_sample_video_fps: int = 8


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
