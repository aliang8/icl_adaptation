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
  --config scripts/maniskill/ppo_wandb_repro/configs/PickCube-v1_test.json \
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

### RGB / image observations (ICL export)

The **full stitched PPO rollout buffer** (`--icl-save-rollout-buffer`, on by default) is normally **state, actions, and rewards only**. Optional **`--icl-rollout-render-rgb`** records **`rgb_array`** every **training** rollout step into the **same** stitched episodes (`images_view_*` inside **`trajectories_shard_*.h5`** / the final export). That path is slow at large `num_envs`; use **`--reconfiguration-freq 0`**, **`physx_cuda`**, and a modest **`--icl-shard-max-episodes`** / **`--icl-rollout-rgb-shard-max-episodes`**. It **disables** **`--icl-image-snapshot-*`**.

- **Periodic RGB during training (snapshots):** when `global_step` crosses each **`--icl-image-snapshot-every-steps N`** boundary (after the policy update), the script rolls out **`--icl-image-snapshot-episodes`** short episodes with `render_mode=rgb_array`. Episodes are buffered and written as **`trajectories_image_shard_*.h5`** in **`datasets/maniskill/<env_id>/`** (same folder as state rollout shards), with RGB in **`images_view_*`**. Stored frame size is **`--icl-rgb-resize-hw`** (square H=W; **default 128**, not full 256×256 task renders). Set **`--icl-image-snapshot-shard-max-episodes K`** for episodes per file, or leave it at **`0`** to reuse **`--icl-shard-max-episodes`** when that is > 0 (you must set at least one). Training end flushes any partial buffer. Tune **`--icl-snapshot-hdf5-image-compression`** (`gzip` / `lzf` / `none`) for speed vs size.

- **RGB after training ends:** pass **`--icl-collect-episodes N`** (and optionally **`--icl-max-steps-per-episode`**) to append final-policy RGB into the **main** export path (same single `.h5`).

- **Large state-only runs:** **`--icl-shard-max-episodes M`** (`M>0`) flushes rollout episodes to `trajectories_shard_00000.h5`, … during training instead of holding everything until the end; see `icl_shards_manifest.json` in the task folder.

Example (W&B repro + **sharded** state rollout export + periodic RGB snapshots):

```bash
uv run python scripts/maniskill/run_ppo_wandb_repro.py \
  --config scripts/maniskill/ppo_wandb_repro/configs/PickCube-v1.json \
  --seed 1788 \
  -- --icl-data-root datasets \
     --icl-shard-max-episodes 50000 \
     --icl-rgb-resize-hw 128 \
     --icl-image-snapshot-shard-max-episodes 1000 \
     --icl-image-snapshot-every-steps 2000 \
     --icl-image-snapshot-episodes 5 \
     --icl-image-snapshot-max-steps 50 \
     --reward-scale 3 \
     --success-reward-bonus 5
```

**RGB embedded in the on-policy rollout shards** (no `trajectories_image_shard_*` snapshots): render every training rollout step and flush HDF5s so RAM stays bounded. There is still **one** episode list in memory; **`--icl-shard-max-episodes`** alone sets how often **`trajectories_shard_*.h5`** is written. Optional **`--icl-rollout-rgb-shard-max-episodes K`** only matters when you want a **smaller** flush than the main cap (e.g. `--icl-shard-max-episodes 50000` but flush every **800** RGB-heavy episodes); the code uses **`min`** of the two when both are positive. If `K` is **0**, only **`--icl-shard-max-episodes`** applies—no need to duplicate it.

Example ( **`--reconfiguration-freq 0`** helps batched `rgb_array` render; add **`--num-envs 64`** etc. if your JSON uses a large vector width):

```bash
uv run python scripts/maniskill/run_ppo_wandb_repro.py \
  --config scripts/maniskill/ppo_wandb_repro/configs/PickCube-v1.json \
  --seed 1788 \
  -- --icl-data-root datasets \
     --reconfiguration-freq 0 \
     --icl-shard-ram-flush-episodes 4096 \
     --icl-shard-max-episodes 50000 \
     --icl-rollout-render-rgb \
     --icl-rgb-resize-hw 128
```

**One seed:** pass **`--seed <int>`** on **`run_ppo_wandb_repro.py`** (before **`--`**). Omit **`--seed`** to run every seed listed in the JSON (official table-top runs use **`1788`**, **`4796`**, **`9351`**). Anything after **`--`** is forwarded only to **`ppo_train_icldata.py`**.

**Reward shaping (differs from strict W&B repro):** **`--reward-scale 3`** multiplies rewards in the **on-policy PPO rollout** (after the optional success add). **`--success-reward-bonus 10`** adds **10 env reward units** on the **terminal** timestep when **`final_info`** reports success, **before** `reward_scale`. Sharded **state** rollout HDF5s and **`trajectories_image_shard_*.h5`** both store **the same scaled per-step rewards** as PPO (bonus in env units, then × `reward_scale` on every step).

State rollouts flush every **50 000** episodes in this snippet; RGB snapshot shards use **1000** episodes per `trajectories_image_shard_*.h5` here (buffer fills after **200** snapshot rounds at 5 episodes each), which **lowers RAM** and writes image HDF5s sooner than inheriting a large rollout cap. **`--icl-rgb-resize-hw 128`** matches the **`ppo_train_icldata.py`** default (omit the flag to get 128×128); use **`256`** only if you need higher-res HDF5s. If you omit **`--icl-image-snapshot-shard-max-episodes`**, RGB inherits from **`--icl-shard-max-episodes`** with an internal **1000-episode cap** when the rollout cap is larger. Raise **`--icl-image-snapshot-shard-max-episodes`** only if you want fewer, bigger RGB files (more memory before flush). Use **`uv run`** from the project `.venv` or **`python ...`** from `.venv-maniskill` so **`h5py`** matches `scripts/maniskill/requirements.txt` (see `docs/MANISKILL.md`).

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
