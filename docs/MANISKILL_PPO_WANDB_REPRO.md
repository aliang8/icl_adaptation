# ManiSkill PPO (W&B) reproduction

This matches the **state** PPO runs in the public [ManiSkill PPO results report](https://wandb.ai/stonet2000/ManiSkill/reports/PPO-Results--VmlldzoxMDQzNDMzOA) (`walltime_efficient`, three seeds per task where available).

Configs are stored as **JSON** (not YAML): `scripts/maniskill/ppo_wandb_repro/configs/<EnvId>.json`. See `manifest.json` in that folder for seeds and reference W&B run ids.

## Prerequisites

- Repo root as working directory.
- **Training:** ManiSkill venv (see `scripts/maniskill/requirements.txt`), e.g. `.venv-maniskill`.
- **W&B:** `ppo_train_icldata.py` logs to W&B by default (`--track`). Use `--no-track` for fully offline training. Optional: `--wandb-entity your_team`.
- **Fetching configs:** `wandb` installed (e.g. `uv sync` from repo root). A W&B API key helps if the API rate-limits anonymous access.

Bulk loop `train_ppo_many_envs.sh` keeps W&B off unless you set `TRACK_WANDB=1` (avoids many accidental runs).

## Fetch configs from W&B (separate step)

Run whenever you want to refresh hyperparameters from the project:

```bash
cd /scr/aliang80/icl_adaptation   # your clone
uv run python scripts/maniskill/fetch_ppo_wandb_configs.py
```

Output directory: `scripts/maniskill/ppo_wandb_repro/configs/`.

Optional: `--entity-project stonet2000/ManiSkill` (default) and `--out-dir <path>`.

## Run one environment

**Option A — Python (one config file, optional single seed):**

```bash
cd /scr/aliang80/icl_adaptation
export MANISKILL_PYTHON="${PWD}/.venv-maniskill/bin/python"   # if you use the ManiSkill venv

uv run python scripts/maniskill/run_ppo_wandb_repro.py \
  --config scripts/maniskill/ppo_wandb_repro/configs/PickCube-v1.json
```

One seed only (official runs use `1788`, `4796`, `9351`):

```bash
uv run python scripts/maniskill/run_ppo_wandb_repro.py \
  --config scripts/maniskill/ppo_wandb_repro/configs/PickCube-v1.json \
  --seed 1788
```

Dry-run (print the `ppo_train_icldata.py` command without running):

```bash
uv run python scripts/maniskill/run_ppo_wandb_repro.py \
  --config scripts/maniskill/ppo_wandb_repro/configs/PickCube-v1.json \
  --dry-run
```

**Option B — Shell helper (same thing, env id as name):**

```bash
cd /scr/aliang80/icl_adaptation
ENVS_OVERRIDE="PickCube-v1" ./scripts/maniskill/run_ppo_wandb_repro.sh
```

With one seed:

```bash
SEED=1788 ENVS_OVERRIDE="PickCube-v1" ./scripts/maniskill/run_ppo_wandb_repro.sh
```

Pass extra flags to `ppo_train_icldata.py` after `--`:

```bash
ENVS_OVERRIDE="PickCube-v1" ./scripts/maniskill/run_ppo_wandb_repro.sh -- --icl-data-root datasets
```

## Related scripts

| Script | Role |
|--------|------|
| `scripts/maniskill/fetch_ppo_wandb_configs.py` | Download W&B hyperparameters → JSON |
| `scripts/maniskill/run_ppo_wandb_repro.py` | Load JSON, invoke `ppo_train_icldata.py` |
| `scripts/maniskill/run_ppo_wandb_repro.sh` | Batch default env list or `ENVS_OVERRIDE` |

## Slurm: launch all repro jobs at once

Two patterns:

1. **W&B JSON hyperparameters** (this doc): one job per `configs/*.json`.

   ```bash
   cd /scr/aliang80/icl_adaptation
   SBATCH_EXTRA=(--account=YOUR_ACCT --partition=gpu) \
     ./scripts/maniskill/submit_ppo_wandb_repro_slurm.sh
   ```

   Writes expanded batch scripts under `slurm-logs/` and submits them. Optional: `ENVS_OVERRIDE="PickCube-v1"`, `SEED=1788`, `TRACK_WANDB=0`, `WANDB_ENTITY=...`.

2. **Fixed script hyperparameters** (`train_ppo_many_envs.sh` defaults): one job per env id.

   ```bash
   SUBMIT_SLURM=1 SBATCH_EXTRA=(--account=YOUR_ACCT --partition=gpu) \
     ./scripts/maniskill/train_ppo_many_envs.sh
   ```

   Uses `scripts/maniskill/ppo_single_env.sbatch` (not the JSON files).

For general ManiSkill setup, see `docs/MANISKILL.md`.
