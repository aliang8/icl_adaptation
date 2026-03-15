# In-Context Adaptation for Robot Trajectories

In-context learning for robot trajectories: during **training** we use context trajectories from the same task **sorted by returns** (best first); at **inference** we do **zero-shot adaptation** by providing previous rollouts **sorted in ascending order** (worst to best).

This codebase follows a single training entrypoint, compositional Hydra configs, and full resume-state checkpoints. It is built from ideas in [Meta-DT](https://github.com/NJU-RL/Meta-DT) (NeurIPS 2024) and structured for clarity as in [ICRT](https://github.com/Max-Fu/icrt) (ICRA 2025).

## Setup and quick start

**Setup** (uv, conda, D4RL/MuJoCo) and **step-by-step commands** for D4RL HalfCheetah (download → train) are in **[SETUP.md](SETUP.md)**.

TL;DR with uv: `uv venv && source .venv/bin/activate && uv sync` then `uv sync --extra d4rl` for HalfCheetah (Minari); download with `uv run python scripts/download_d4rl_halfcheetah.py --qualities medium expert medium_expert` and train with `uv run python -m src.train --wandb --override data=halfcheetah`.

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

Every run saves a **resolved config** under `outputs/resolved_config.yaml` and the checkpoint stores the exact resolved config for reproducibility.

## Checkpoints

- **Training checkpoint** (resume state): model, optimizer, scheduler, scaler, epoch, global_step, best_metric, full config, git commit, RNG state.
- **Three types**: `checkpoint_latest.pt`, `checkpoint_best.pt`, and periodic `checkpoint_step_*.pt`.
- **Inference artifact**: separate export via `--export-only` (model weights + config + state_mean/state_std). Use for deployment, not for resuming training.

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

- **HalfCheetah (Minari)**: `datasets/HalfCheetah-v2/<data_quality>/trajectories.pkl` (created by `scripts/download_d4rl_halfcheetah.py` via Minari). Use `data_quality: medium_expert` (or `medium`, `expert`) in config and `--override data=halfcheetah`.
- **AntDir / multi-task**: `datasets/<env_name>/<data_quality>/dataset_task_<id>.pkl` and `dataset_task_prompt<id>.pkl` (see Meta-DT data collection).  
If no data is found, the trainer falls back to minimal dummy data for a dry run.

## Logging

- **TensorBoard**: `outputs/logs`.
- **W&B**: `python -m src.train --wandb` (and optional `--run-name <name>`). Logs train loss, lr, grad norm, GPU memory, eval metric, and full config.
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

If you use this code or ideas from it, please cite:

- Meta-DT: [Wang et al., NeurIPS 2024](https://github.com/NJU-RL/Meta-DT) — Offline Meta-RL as conditional sequence modeling with world model disentanglement.
- ICRT: [Fu et al., ICRA 2025](https://github.com/Max-Fu/icrt) — In-context imitation learning via next-token prediction.

## License

See repository license file.
