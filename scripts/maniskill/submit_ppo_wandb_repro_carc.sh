#!/usr/bin/env bash
# Submit W&B-repro jobs on USC CARC using carc_ppo_wandb_repro_single_env.sbatch.
#
# Prereq: edit placeholders in that file OR pass account/partition on every sbatch via SBATCH_EXTRA.
# Run ``myaccount`` on Endeavour (or see CARC docs for Discovery) to get values:
#   https://github.com/kylewang1999/carc_usage/blob/main/consolidated.md
#
# Example (Endeavour login node, from repo root):
#   SBATCH_EXTRA=(--account=YOUR_ENDEAVOUR_ACCOUNT --partition=YOUR_CONDO_PARTITION) \\
#     ./scripts/maniskill/submit_ppo_wandb_repro_carc.sh
#
# Same optional env vars as submit_ppo_wandb_repro_slurm.sh (ENVS_OVERRIDE, REPRO_ALL_CONFIGS, SEED, …).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export SBATCH_SCRIPT="${ROOT}/scripts/maniskill/carc_ppo_wandb_repro_single_env.sbatch"
exec "${ROOT}/scripts/maniskill/submit_ppo_wandb_repro_slurm.sh"
