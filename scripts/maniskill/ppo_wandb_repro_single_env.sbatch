#!/bin/bash
# Slurm: one W&B-repro JSON config → run_ppo_wandb_repro.py (all seeds in that JSON, or one if SEED set).
#
# Submitted by submit_ppo_wandb_repro_slurm.sh with --export, or manually:
#   sbatch --export=ALL,REPO_ROOT=$PWD,PPO_WANDB_CONFIG=$PWD/scripts/maniskill/ppo_wandb_repro/configs/PickCube-v1.json \
#     scripts/maniskill/ppo_wandb_repro_single_env.sbatch
#
# Optional env: TRACK_WANDB=1 (default 1), ICL_ROOT, SEED, WANDB_ENTITY, MANISKILL_VENV

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=shard:10
#SBATCH --mem=50G
#SBATCH --time=24:00:00

set -euo pipefail

: "${REPO_ROOT:?Set REPO_ROOT to icl_adaptation repo root}"
: "${PPO_WANDB_CONFIG:?Set PPO_WANDB_CONFIG to absolute path to a ppo_wandb_repro/configs/<Env>.json}"

cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

TRACK_WANDB="${TRACK_WANDB:-1}"
ICL_ROOT="${ICL_ROOT:-datasets}"
MANISKILL_VENV="${MANISKILL_VENV:-${REPO_ROOT}/.venv-maniskill}"
if [ -f "${MANISKILL_VENV}/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "${MANISKILL_VENV}/bin/activate"
fi

export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-12345}"

if command -v srun >/dev/null 2>&1; then
  _run=(srun)
else
  _run=()
fi

echo "Job ${SLURM_JOB_ID:-local} $(hostname) $(date)"
echo "REPO_ROOT=${REPO_ROOT} PPO_WANDB_CONFIG=${PPO_WANDB_CONFIG} TRACK_WANDB=${TRACK_WANDB}"

cmd=("${_run[@]}" python scripts/maniskill/run_ppo_wandb_repro.py --config "${PPO_WANDB_CONFIG}")
if [ -n "${SEED:-}" ]; then
  cmd+=(--seed "${SEED}")
fi
cmd+=(--)
if [ "${TRACK_WANDB}" != "1" ]; then
  cmd+=(--no-track)
fi
cmd+=(--icl-data-root "${ICL_ROOT}")
if [ -n "${WANDB_ENTITY:-}" ]; then
  cmd+=(--wandb-entity "${WANDB_ENTITY}")
fi

"${cmd[@]}"

echo "Finished $(date)"
