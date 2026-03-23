# In-Context Adaptation for Robot Trajectories

In-context learning for robot trajectories: during **training** we use context trajectories from the same task **sorted by returns** (best first); at **inference** we do **zero-shot adaptation** by providing previous rollouts **sorted in ascending order** (worst to best).

This codebase follows a single training entrypoint, compositional Hydra configs, and full resume-state checkpoints. It is built from ideas in [Meta-DT](https://github.com/NJU-RL/Meta-DT) (NeurIPS 2024) and structured for clarity as in [ICRT](https://github.com/Max-Fu/icrt) (ICRA 2025).

## Setup and quick start

**Setup** (uv, conda, D4RL/MuJoCo) and **step-by-step commands** for D4RL HalfCheetah (download → train) are in **[SETUP.md](SETUP.md)**.

TL;DR with uv: `uv venv && source .venv/bin/activate && uv sync` then `uv sync --extra d4rl` for HalfCheetah (Minari); download with `uv run python scripts/download_d4rl_halfcheetah.py --qualities medium expert medium_expert` and train with `uv run python -m src.train --wandb --override data=[base,halfcheetah]`.

## Project layout

```
project/
  pyproject.toml
  configs/
    config.yaml          # Composes model, data, optim, system, experiment
    model/
    data/
    optim/
    system/
    experiment/
  src/
    train.py             # Single entrypoint: fresh, resume, eval-only, export
    eval.py              # Evaluation and inference helpers
    models/
    data/
    engine/
      trainer.py
      checkpointing.py
      logging.py
    config/
  scripts/
  outputs/
```

## Training entrypoint

One script, four modes:

- **Start fresh**  
  `uv run python -m src.train` or `./scripts/train.sh`

- **Resume from checkpoint**  
  `uv run python -m src.train --resume outputs/checkpoints/checkpoint_latest.pt` or `./scripts/resume.sh [path]`

- **Eval only**  
  `uv run python -m src.train --eval-only --resume outputs/checkpoints/checkpoint_best.pt` or `./scripts/eval_only.sh [ckpt]`

- **Save inference artifact**  
  `uv run python -m src.train --export-only outputs/checkpoints/checkpoint_best.pt` or `./scripts/export.sh [ckpt]`

Overrides:  
`uv run python -m src.train --override experiment.max_steps=1000 --override data.batch_size=64`

## Config

Configs are **compositional** (Hydra):

- `configs/config.yaml`: `defaults: [model: transformer, data: ant_dir, optim: adamw, system: single_gpu, experiment: base]`
- Typed schema lives in `src/config/schema.py` (ModelConfig, DataConfig, OptimConfig, SystemConfig, ExperimentConfig).

## Run directory (experiment layout)

Each training run writes to a **dated, named directory** with a fixed layout:

```
outputs/<project_name>/<YYYY-MM-DD>/<run_name>__seed_<X>__<git_hash>/
  .hydra/           config.yaml, overrides.yaml
  logs/             train.log
  metrics/          history.jsonl, summary.json
  ckpts/            last/, best/, step_00050000/ ... (checkpoint.pt + metadata.json)
  artifacts/        inference/ (model_export.pt), export/
  eval/             val/, test/ (metrics.json, per_task.json)
  viz/samples/      step_*/ rollout_*.png, returns.png (eval rollout visualizations)
  code/             git.txt, diff.patch
  README.md
```

- **Experiment name**: `run_name` from `--run-name` or config; slug includes `seed` and short git hash for reproducibility.
- **Resume**: pass path to a checkpoint file under that run, e.g. `--resume outputs/icl_adaptation/2026-03-15/my_run__seed_0__a1b2c3d/ckpts/last/checkpoint.pt`; the same run directory is reused for logs and later checkpoints.
- **Eval**: at each eval step, rollout visualizations are saved under `viz/samples/step_XXXXX/` (state/action curves and return bars when an env is available).

## Checkpoints

- **Training checkpoint** (resume state): model, optimizer, scheduler, scaler, epoch, global_step, best_metric, full config, git commit, RNG state. Stored under `ckpts/last/`, `ckpts/best/`, `ckpts/step_*/` as `checkpoint.pt` plus `metadata.json`.
- **Inference artifact**: exported under `artifacts/inference/model_export.pt` at end of training, or via `--export-only` (model weights + config + state_mean/state_std). Use for deployment, not for resuming.

Loading untrusted checkpoints: use `weights_only=True` when loading (see PyTorch docs and `src/engine/checkpointing.py`).

## Data and context

- **In-context sampling**: For each training segment we **sample N trajectories** from the same task (`num_context_trajectories`), then **sort them by increasing return** and concatenate segments so the model sees **low→high progress** in context. Sampling can be `random` or `stratified` (return buckets) for diversity. Config: `data.num_context_trajectories`, `data.context_sort_ascending`, `data.context_sampling`, `data.max_total_prompt_length`.
- **Inference**: zero-shot adaptation with previous rollouts **sorted ascending** (worst to best).

**Offline RL (e.g. D4RL)**: rewards are already in the data; no extra step.

**Robotics (dense rewards)**: If trajectories come from rollouts without dense rewards, run **one offline step** with [Robometer](https://huggingface.co/robometer/Robometer-4B) to compute per-frame progress/success, then train on the scored trajectories:

```bash
uv run python scripts/score_trajectories_robometer.py --input datasets/MyRobot/trajectories.pkl --output datasets/MyRobot/trajectories_robometer.pkl --task "your task"
```

Use the output pkl as your dataset; see script docstring for trajectory format (`video_path` or `frames`).

Dataset layout:

- **HalfCheetah / D4RL-style (Minari)**: `datasets/HalfCheetah-v2/<data_quality>/trajectories.pkl` from `scripts/download_d4rl_halfcheetah.py`. Rewards are **already** in the data—no dense-reward or Robometer step. Train with `--override data=[base,halfcheetah]` (default `model=transformer`). Details: **[docs/D4RL_TRAINING.md](docs/D4RL_TRAINING.md)**; setup: [SETUP.md](SETUP.md).
- **ICRT-MT (language + multi-view images)**: `datasets/ICRT-MT/` from `uv run python scripts/download_icrt_dataset.py --output-dir datasets` ([HuggingFace](https://huggingface.co/datasets/Ravenh97/ICRT-MT)). After `uv sync --extra icrt`, visualize with `uv run python scripts/visualize_icrt_data.py`. Use `data=[base,icrt_mt] model=vla_dt` for language-conditioned, vision-based in-context policy (VLA-DT). See [docs/ICRT_MT_TRAINING.md](docs/ICRT_MT_TRAINING.md) (training only, no eval) and SETUP.md “Optional: ICRT-style”.
- **LIBERO-Cosmos-Policy**: [nvidia/LIBERO-Cosmos-Policy](https://huggingface.co/datasets/nvidia/LIBERO-Cosmos-Policy) — download with `uv run python scripts/download_libero_cosmos.py --output-dir datasets`, train with `data=[base,libero_cosmos]`, run in-distribution eval with `scripts/run_libero_eval.py`. See [docs/LIBERO_COSMOS_TRAINING.md](docs/LIBERO_COSMOS_TRAINING.md).
- **RoboArena DataDump**: [RoboArena/DataDump_08-05-2025](https://huggingface.co/datasets/RoboArena/DataDump_08-05-2025) or [DataDump_02-03-2026](https://huggingface.co/datasets/RoboArena/DataDump_02-03-2026) — download the raw dump, then convert evaluation_sessions (metadata + policy videos/npz) to task-grouped episodes and a manifest.

  **Download raw data** (requires git and [Git LFS](https://git-lfs.github.com/) for videos):

  ```bash
  git lfs install
  cd datasets
  git clone https://huggingface.co/datasets/RoboArena/DataDump_08-05-2025
  git clone https://huggingface.co/datasets/RoboArena/DataDump_02-03-2026
  ```

  **Convert to episodes + manifest**:

  ```bash
  uv run python scripts/convert_roboarena_to_dataset.py --input-dir datasets/DataDump_08-05-2025
  uv run python scripts/convert_roboarena_to_dataset.py --input-dir datasets/DataDump_02-03-2026 --output-dir datasets --symlink
  ```

  Output: `episodes/<task_slug>/<idx>/` (primary.mp4, wrist.mp4, lowdim.npz) and `manifest.parquet`. Use `--skip-existing` (default) to skip already-converted episodes; `--no-skip-existing` to overwrite.
- **AntDir / multi-task**: `datasets/<env_name>/<data_quality>/dataset_task_<id>.pkl` and `dataset_task_prompt<id>.pkl` (see Meta-DT data collection).  
If no data is found, the trainer falls back to minimal dummy data for a dry run.

## Logging

- **TensorBoard**: `outputs/logs`.
- **W&B**: `uv run python -m src.train --wandb` (and optional `--run-name <name>`). Logs train loss, lr, grad norm, GPU memory, eval metric, and full config.
- **Loguru**: progress messages (e.g. “Evaluating at step …”, “Saved best checkpoint”, “Training started: max_steps=…”) to stderr. Configure level/format via `loguru` API if needed.

## Other offline RL benchmarks (trajectories with different returns)

Besides **D4RL** (HalfCheetah, Hopper, Walker2d, Ant: medium / medium-replay / medium-expert / expert), these also provide or can be used for **mixed-return / multi-quality** data:

- **D4RL MuJoCo** (same repo): Hopper, Walker2d, Ant in the same quality tiers; good for in-context “adaptation” by return-sorted context.
- **Robomimic** ([robomimic](https://github.com/ARISE-Initiative/robomimic)): human demos and synthetic policies with different success rates; multiple datasets per task (e.g. low/med/high quality).
- **LIBERO** ([LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)): lifelong robot manipulation; multiple tasks and can define “return” by task success or proxy rewards.
- **Open X-Embodiment** (OXE) / **DROID**: large-scale robot datasets; can filter or bucket trajectories by return or success for in-context conditioning.
- **Meta-World / rand_param_envs** (as in Meta-DT): multi-task RL with shared structure; different tasks or checkpoints yield different return distributions; context = same task sorted by return.

For a minimal start we use **D4RL HalfCheetah** (medium + medium-expert) so trajectories naturally have a mix of returns; context is sorted by return for training and (at inference) previous rollouts can be passed sorted ascending.

## Citation

## Format and lint

Install dev deps then run Ruff to format and fix lint:

```bash
uv sync --extra dev
uv run ruff format src scripts
uv run ruff check src scripts --fix
```

- `ruff format` — normalize quotes, indentation, line length (100).
- `ruff check --fix` — fix auto-fixable issues (E, F, I).

If you use this code or ideas from it, please cite:

- Meta-DT: [Wang et al., NeurIPS 2024](https://github.com/NJU-RL/Meta-DT) — Offline Meta-RL as conditional sequence modeling with world model disentanglement.
- ICRT: [Fu et al., ICRA 2025](https://github.com/Max-Fu/icrt) — In-context imitation learning via next-token prediction.

## License

See repository license file.
