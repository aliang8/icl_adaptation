# ManiSkill PPO (W&B) reproduction

This matches the **state** PPO runs in the public [ManiSkill PPO results report](https://wandb.ai/stonet2000/ManiSkill/reports/PPO-Results--VmlldzoxMDQzNDMzOA) (`walltime_efficient`, three seeds per task where available).

Configs are stored as **JSON** (not YAML): `scripts/maniskill/ppo_wandb_repro/configs/<EnvId>.json`. See `manifest.json` in that folder for seeds and reference W&B run ids.

## Prerequisites

- Repo root as working directory.
- **Training:** ManiSkill venv (see `scripts/maniskill/requirements.txt`), e.g. `.venv-maniskill`.
- **W&B:** `ppo_train_icldata.py` logs to W&B by default (`--track`). Use `--no-track` for fully offline training. Optional: `--wandb-entity your_team`.
- **Fetching configs:** `wandb` installed (e.g. `uv sync` from repo root). A W&B API key helps if the API rate-limits anonymous access.

Bulk loop `train_ppo_many_envs.sh` keeps W&B off unless you set `TRACK_WANDB=1` (avoids many accidental runs).

## Example: repo on USC CARC `/project2/...`

If the clone lives under project storage (typical on CARC), point everything at that root:

```bash
export REPO_ROOT=/project2/biyik_1165/aliang80/icl_adaptation
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export MANISKILL_PYTHON="${REPO_ROOT}/.venv-maniskill/bin/python"   # if you use the ManiSkill venv there
```

Refresh JSON configs and run one task:

```bash
cd /project2/biyik_1165/aliang80/icl_adaptation
uv run python scripts/maniskill/fetch_ppo_wandb_configs.py
uv run python scripts/maniskill/run_ppo_wandb_repro.py \
  --config scripts/maniskill/ppo_wandb_repro/configs/PickCube-v1.json
```


Submit Slurm jobs from the login node (after editing account/partition in `carc_ppo_wandb_repro_single_env.sbatch` or passing them in `SBATCH_EXTRA`). `REPO_ROOT` in the job environment must be this path:

```bash
cd /project2/biyik_1165/aliang80/icl_adaptation
SBATCH_EXTRA=(--account=YOUR_ENDEAVOUR_ACCOUNT --partition=YOUR_CONDO_PARTITION) \
  ./scripts/maniskill/submit_ppo_wandb_repro_carc.sh
```

The generated jobs pass `REPO_ROOT=$PWD` at submit time, so run the submit script from this directory (or `cd` there first).

Other snippets in this doc use `/scr/aliang80/icl_adaptation` only as a placeholder; substitute `$REPO_ROOT` or the `/project2/...` path above.

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

### RGB / image observations in `trajectories.h5`

The **full stitched PPO rollout buffer** (`--icl-save-rollout-buffer`, on by default) is **state, actions, and rewards only** — it does not record pixels from training rollouts.

To **append RGB episodes at the end** (final policy, `render_mode=rgb_array`) into the same `<icl_data_root>/maniskill/<env_id>/trajectories.h5`, pass **`--icl-collect-episodes N`** (and optionally **`--icl-max-steps-per-episode`**). Those runs are **extra** on-policy rollouts after training, not a pixel dump of the entire historical buffer.

Example (W&B repro + final-policy RGB episodes):

```bash
uv run python scripts/maniskill/run_ppo_wandb_repro.py \
  --config scripts/maniskill/ppo_wandb_repro/configs/PickCube-v1.json \
  -- --icl-collect-episodes 64 --icl-max-steps-per-episode 512
```

For **periodic** RGB snapshots during training (separate files under `.../image_snapshots/`), use **`--icl-image-snapshot-every-steps`** and **`--icl-image-snapshot-episodes`** (see `ppo_train_icldata.py` module docstring).

## Related scripts

| Script | Role |
|--------|------|
| `scripts/maniskill/fetch_ppo_wandb_configs.py` | Download W&B hyperparameters → JSON |
| `scripts/maniskill/run_ppo_wandb_repro.py` | Load JSON(s), invoke `ppo_train_icldata.py`; no `--config` → default table-top list; `--all-configs` → every JSON |
| `scripts/maniskill/run_ppo_wandb_repro.sh` | Reads `default_tabletop_repro_envs.txt` or `ENVS_OVERRIDE` |

## Slurm: launch all repro jobs at once

Two patterns:

1. **W&B JSON hyperparameters** (this doc): by default one job per env in `ppo_wandb_repro/default_tabletop_repro_envs.txt` (10 table-top tasks with bundled JSON). Set `REPRO_ALL_CONFIGS=1` to submit every `configs/*.json` except `manifest.json`.

   ```bash
   cd /scr/aliang80/icl_adaptation
   SBATCH_EXTRA=(--account=YOUR_ACCT --partition=gpu) \
     ./scripts/maniskill/submit_ppo_wandb_repro_slurm.sh
   ```

   **USC CARC** (Discovery / Endeavour): use the template aligned with [carc_usage](https://github.com/kylewang1999/carc_usage) (`consolidated.md` — run `myaccount` for account/partition). Edit placeholders in `scripts/maniskill/carc_ppo_wandb_repro_single_env.sbatch`, then:

   ```bash
   SBATCH_EXTRA=(--account=YOUR_ENDEAVOUR_ACCOUNT --partition=YOUR_CONDO_PARTITION) \
     ./scripts/maniskill/submit_ppo_wandb_repro_carc.sh
   ```

   Or set `SBATCH_SCRIPT` to that file when calling `submit_ppo_wandb_repro_slurm.sh`. Override GPU type in the `.sbatch` file if needed (e.g. `#SBATCH --gpus=h200:1` per CARC Endeavour examples).

   Writes expanded batch scripts under `slurm-logs/` and submits them. Optional: `ENVS_OVERRIDE="PickCube-v1"`, `SEED=1788`, `TRACK_WANDB=0`, `WANDB_ENTITY=...`.

2. **Fixed script hyperparameters** (`train_ppo_many_envs.sh` defaults): one job per env id.

   ```bash
   SUBMIT_SLURM=1 SBATCH_EXTRA=(--account=YOUR_ACCT --partition=gpu) \
     ./scripts/maniskill/train_ppo_many_envs.sh
   ```

   Uses `scripts/maniskill/ppo_single_env.sbatch` (not the JSON files).

For general ManiSkill setup, see `docs/MANISKILL.md`.
