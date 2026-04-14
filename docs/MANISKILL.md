# ManiSkill integration

Episode return overview from a flat v2 `trajectories.h5` (needs `matplotlib`; from repo root): one PNG with histogram, sorted returns, cumulative success, file-order returns, and consecutive deltas.

```bash
uv run python scripts/maniskill/trajectory_returns_history.py datasets/maniskill/PickCube-v1/trajectories.h5
```

Writes `<stem>_returns_overview.png` next to the HDF5 (e.g. `trajectories_returns_overview.png`). Use `--min-traj-len L` (default 10) to keep only episodes with `len(rewards) >= L`, and `--bins` for the histogram.

This repo can train **in-context (ICL) models** on trajectories generated in [ManiSkill](https://github.com/haosulab/ManiSkill) using the upstream **PPO** recipe, extended to export **`trajectories.h5`** (HDF5, gzip-compressed episodes) compatible with `get_icl_trajectory_dataset` / `train.py`.

## Install

Follow the official [installation guide](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html): install PyTorch for your platform, then ManiSkill.

### Use a **separate virtualenv** for ManiSkill scripts

`mani-skill` **pins `gymnasium==0.29.1`**. That version does **not** provide `HalfCheetah-v5` / `gymnasium.envs.mujoco.half_cheetah_v5`, so installing it into the **same** environment as D4RL / Minari eval will break HalfCheetah rollouts (`ModuleNotFoundError` on import). The main project’s `d4rl` extra therefore pins **`gymnasium>=1.2.0`**, which **cannot** be resolved together with `mani-skill` in one `uv` lock.

**Recommended:** dedicated venv only for `scripts/maniskill/*`:

```bash
python -m venv .venv-maniskill
source .venv-maniskill/bin/activate   # or: source .venv-maniskill/Scripts/activate on Windows
pip install torch                       # your platform / CUDA build
pip install -r scripts/maniskill/requirements.txt
```

To run **`python -m src.train`** from the same venv (so **periodic eval** can build the ManiSkill simulator — `mani-skill` must be importable), add ICL/Hydra deps **without** replacing ManiSkill’s `gymnasium` pin:

```bash
pip install -r scripts/maniskill/requirements_icl_train.txt
export PYTHONPATH=/path/to/icl_adaptation
```

Then use plain `python -m src.train ...` (no `uv run`). Eval uses `import mani_skill.envs` and the task id `PickCube-v1` (the `ManiSkill/` prefix in `data.env_name` is stripped for `gym.make`). Override sim if needed: `MANISKILL_EVAL_SIM=physx_cpu` when CUDA/driver fails.

State-only ICL (`data.use_vision=false`) is supported; vision eval for ManiSkill is not wired in `eval_viz` yet.

For **PPO / `ppo_train_icldata.py` only**, you can skip `requirements_icl_train.txt` and set `PYTHONPATH` as above so `from src.data.ic_replay_buffer_hdf5 import ...` / `from src.data.maniskill_io import ...` resolve.

Alternatively, `pip install -e . --no-deps` inside `.venv-maniskill` plus the two requirement files can work; avoid plain `pip install -e .` if it pulls a conflicting Gym stack. Keep ManiSkill and D4RL (`uv sync --extra d4rl`) in **different** envs.

**Linux + NVIDIA:** install Vulkan for rendering (e.g. `sudo apt-get install libvulkan1 vulkan-tools`). See the docs for ICD JSON hints if `vulkaninfo` fails.

**Data / assets:** optional environment variables:

```bash
export MS_ASSET_DIR=/path/to/maniskill_data
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1   # optional: non-interactive asset fetch
```

## Sanity check

From the **ManiSkill** venv (`PYTHONPATH` = repo root):

```bash
python scripts/maniskill/test_maniskill.py
```

You should see a short rollout step count without errors.

The script tries **`physx_cuda`** first, then **`physx_cpu`** if CUDA fails (common when **PyTorch’s CUDA build is newer than your NVIDIA driver**). To force CPU-only: `MANISKILL_SMOKE_SIM=cpu python scripts/maniskill/test_maniskill.py`. To require GPU (no fallback): `MANISKILL_SMOKE_SIM=cuda`.

### PyTorch vs NVIDIA driver (`RuntimeError: driver ... too old`)

ManiSkill GPU sim initializes CUDA through PyTorch. If you see *“NVIDIA driver on your system is too old”* (or similar), the **PyTorch wheel’s bundled CUDA** is newer than what your **driver** allows. Fix it in one of these ways:

1. **Upgrade the NVIDIA driver** on the host (cluster admins / [NVIDIA downloads](http://www.nvidia.com/Download/index.aspx)) so it supports the CUDA generation your PyTorch build expects. See [CUDA compatibility](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html).

2. **Reinstall PyTorch** in the **ManiSkill venv** with a **lower CUDA** wheel that your current driver still supports.

In the ManiSkill venv, check what the machine advertises and what you have installed:

```bash
nvidia-smi                                    # top-right "CUDA Version" = max toolkit API your driver supports
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Replace PyTorch with a build that matches your driver (uninstall first so you do not mix wheels):

```bash
pip uninstall -y torch torchvision torchaudio
```

Then install **one** of the following (older driver → prefer **earlier** `cu*` in the list; if unsure, try `cu118` first):

```bash
# CUDA 11.8 wheels (works on many older drivers)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1 wheels
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4 wheels (needs a recent driver)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Exact combinations change over time; use the official selector at [pytorch.org](https://pytorch.org) if these indices move.

**CPU-only** (no GPU sim; fine for quick import checks, slow for training):

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Re-run `python scripts/maniskill/test_maniskill.py` (or `MANISKILL_SMOKE_SIM=cuda` once you expect GPU sim to work).

## Generate ICL data (PPO + export)

Script: `scripts/maniskill/ppo_train_icldata.py` (upstream [`examples/baselines/ppo/ppo.py`](https://github.com/haosulab/ManiSkill/tree/main/examples/baselines/ppo) plus ICL export).

Output: `datasets/maniskill/<env_id>/trajectories.h5` (override with `--icl-data-root`). The file uses **flat HDF5 v2**: a single time axis for `observations`, `actions`, `rewards`, `terminals`, plus `episode_starts` and `episode_lengths` (one entry per episode) and optional `episode_meta_json` per episode.

- **`--icl-save-rollout-buffer` (default: True)** — stitches **every** on-policy PPO rollout (all training steps) into episode dicts: **state**, **actions**, **rewards** (per-step values **as used by PPO**, i.e. after optional terminal `--success-reward-bonus` and `--reward-scale`). **No RGB** (rendering each env step during training would be far too slow). Disable with `--no-icl-save-rollout-buffer`. When ManiSkill reports `final_info` / `episode` metrics, each finished episode also gets an **`episode_meta`** dict (e.g. `success_once`, `success_at_end`, `fail_once`, `fail_at_end`, env `return`, `episode_len`).
- **`--icl-collect-episodes` (default: 0)** — optional **extra** rollouts with the **final** policy using `env.render()`, so those trajectories can include **`images`**. Per-step **`rewards`** match PPO (terminal `--success-reward-bonus` then `--reward-scale` on every step). Appended into the main `trajectories.h5` when > 0.
- **RGB during training** — `--icl-image-snapshot-every-steps N` (with `--icl-image-snapshot-episodes X`, default 8) **collects** RGB whenever total **env steps** first cross each multiple of `N` (after that iteration’s PPO update). Episodes are buffered and written only as sharded files:

  `datasets/maniskill/<env_id>/trajectories_image_shard_00000.h5`, …

  Set **`--icl-image-snapshot-shard-max-episodes K`**, or leave it at 0 and set **`--icl-shard-max-episodes K`** so RGB uses **`min(K, 1000)`** episodes per image shard (state shards can still use full `K`). Each collection adds up to `X` episodes with **state + RGB** (same schema as other ICL trajectories; **`rewards`** are PPO-scaled like the rollout buffer). At most **one** collection per training iteration (largest `N`-aligned boundary ≤ `global_step`). `train.py` discovers all flat v2 HDF5s under **`maniskill/<env_id>/`** (manifest, state shards, then monolithic `trajectories*.h5`, then image shards); override with **`data.trajectory_hdf5_paths`** if you need an explicit list.

Very long runs produce **many** episodes in the main rollout buffer; the list is held in RAM until the end of training.

### Inspect trajectory files

```bash
python scripts/maniskill/maniskill_trajectory_stats.py datasets/maniskill/PickCube-v1/trajectories.h5
python scripts/maniskill/maniskill_trajectory_stats.py datasets/maniskill/PickCube-v1/trajectories_image_shard_*.h5
```

Example (fast task, rollout buffer only — state ICL):

```bash
python scripts/maniskill/ppo_train_icldata.py \
  --env-id PushCube-v1 \
  --num-envs 256 \
  --total-timesteps 500_000 \
  --icl-data-root datasets
```

RGB snapshots every 50k env steps (8 episodes per collection), sharded at 5000 episodes per file:

```bash
python scripts/maniskill/ppo_train_icldata.py \
  --env-id PushCube-v1 \
  --num-envs 256 \
  --total-timesteps 500_000 \
  --icl-image-snapshot-every-steps 50_000 \
  --icl-image-snapshot-episodes 8 \
  --icl-image-snapshot-shard-max-episodes 5000 \
  --icl-collect-episodes 0 \
  --icl-data-root datasets
```

Export **only** from a checkpoint (no training rollouts to stitch — you must collect fresh episodes):

```bash
python scripts/maniskill/ppo_train_icldata.py \
  --env-id PushCube-v1 \
  --checkpoint runs/<run>/final_ckpt.pt \
  --icl-export-only \
  --icl-collect-episodes 100
```

## Train ICL (this repo) on ManiSkill data

**Where to run `src/train.py`**

- **Main project env** (`uv run python -m src.train`): loads `trajectories.h5` fine, but **`mani-skill` is not installed** in that lockfile. **Periodic sim eval** (`experiment.eval_every_steps` > 0) needs `mani_skill` importable in **that same interpreter** — either set **`experiment.eval_every_steps=0`** for offline-only training, install ManiSkill into the uv env, or run training with **`.venv-maniskill/bin/python -m src.train`** (not `uv run`), with **`PYTHONPATH=<repo>`** and **`pip install -r scripts/maniskill/requirements_icl_train.txt`** (note the **`-r`** flag).
- **ManiSkill venv** + **`pip install -r scripts/maniskill/requirements_icl_train.txt`** + **`PYTHONPATH=<repo>`**: eval rollouts use the same `gym.make` kwargs as PPO where configured (`data.maniskill_sim_backend`, `maniskill_reward_mode`, `maniskill_control_mode` in `maniskill_pickcube.yaml`), plus **`FlattenActionSpaceWrapper`** when the task uses a `Dict` action space.

**Config**

1. Set **`data.env_name`** to `ManiSkill/<env_id>` (same `--env-id` as PPO, e.g. `ManiSkill/PushCube-v1`).
2. Ensure **`paths.data_root`** contains `maniskill/<env_id>/trajectories.h5` (PPO writes under `--icl-data-root`, e.g. `datasets/maniskill/<env_id>/trajectories.h5`).
3. If trajectories include **`images`**, enable vision in config: **`data.use_vision: true`** and **`model.use_vision: true`** (and set `model` image / encoder fields as for other vision datasets).
4. **Full-trajectory ICL** (`data.context_style: full_trajectory`): each context trajectory is padded or truncated to **`data.max_episode_steps`** before concatenation into the prompt (masked padding), so variable-length demos align before `max_total_prompt_length` trimming.

**Run training** (example; use your experiment/model config as usual):

```bash
# from repo root; data_root defaults in configs often point at ./datasets or ./all_datasets
python src/train.py data=maniskill_pickcube paths.data_root=datasets \
  model.use_vision=true data.use_vision=true
```

Adjust `experiment=...`, `model=...`, `paths.data_root`, etc. to match your setup. The `configs/data/maniskill_pickcube.yaml` preset sets `env_name`, horizon, and RTG scale; you can copy it for other tasks by changing `env_name` and paths.

For several ManiSkill tasks, use separate `env_name` / data layouts per run. List shard paths explicitly in `data.trajectory_hdf5_paths` (see `configs/data/maniskill_pickcube.yaml`). Optional directory enumeration for tooling: `discover_ic_replay_buffer_paths` in `src/data/ic_replay_buffer_hdf5.py`.

Trajectory keys match the rest of the codebase: `observations`, `actions`, `rewards`, `terminals`, optional **`episode_meta`**, and optionally `images` as a list of one `(T,H,W,3)` `uint8` array per rollout (same convention as VD4RL / dataset collate).

## Batch PPO over several tasks

```bash
bash scripts/maniskill/train_ppo_many_envs.sh
```

By default it runs **table-top 2-finger gripper** envs that have **dense reward** in the [ManiSkill task table](https://maniskill.readthedocs.io/en/latest/tasks/table_top_gripper/index.html) (see `ENVS=(...)` in the script). To run a custom subset:

```bash
ENVS_OVERRIDE="PushCube-v1 PickCube-v1" bash scripts/maniskill/train_ppo_many_envs.sh
```

Tune `NUM_ENVS`, `STEPS`, `SNAP_EVERY`, etc. By default the script uses **`${repo}/.venv-maniskill/bin/python`** when that interpreter exists (otherwise `python` on `PATH`). Override the venv location with **`MANISKILL_VENV`** if you installed elsewhere.

### Slurm: one job per task

`scripts/maniskill/ppo_single_env.sbatch` mirrors a typical cluster batch script (nodes, CPUs, GPU, time, `MANISKILL_VENV` activation, `srun` when available). **Edit the `#SBATCH` lines** for your site (account, partition, memory, GPU syntax) or override on the CLI.

Unless you set **`MANISKILL_VENV`**, jobs use **`${REPO_ROOT}/.venv-maniskill`** (e.g. `/scr/aliang80/icl_adaptation/.venv-maniskill` when the repo lives there).

From the repo root, submit the full env list as separate jobs:

```bash
SUBMIT_SLURM=1 \
SBATCH_EXTRA="--account=YOUR_ACCOUNT --partition=gpu" \
bash scripts/maniskill/train_ppo_many_envs.sh
```

`SBATCH_EXTRA` is split on whitespace and passed before the script path (same pattern as a local `sbatch --account=…` invocation). `REPO_ROOT`, `ENV_ID`, `NUM_ENVS`, `STEPS`, `COLLECT_EP`, `SNAP_EVERY`, `SNAP_EP`, `ICL_ROOT`, and `MANISKILL_VENV` are passed into each job via `--export`.

For a single env without the wrapper script:

```bash
cd /path/to/icl_adaptation
sbatch --account=YOUR_ACCOUNT --partition=gpu \
  --export=ALL,REPO_ROOT=$PWD,ENV_ID=PushCube-v1 \
  scripts/maniskill/ppo_single_env.sbatch
```