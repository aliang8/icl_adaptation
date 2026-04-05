#!/bin/bash
# USC CARC (Discovery / Endeavour): one W&B-repro JSON → run_ppo_wandb_repro.py.
#
# Slurm directives follow the ``sbatch`` example in:
#   https://github.com/kylewang1999/carc_usage/blob/main/consolidated.md
# On an Endeavour login node, run ``myaccount`` and substitute the placeholders below (or pass
# ``sbatch --account=... --partition=...`` on the CLI; CLI overrides these #SBATCH lines).
#
# Submit from repo root (CARC login node):
#   sbatch --export=ALL,REPO_ROOT=$PWD,PPO_WANDB_CONFIG=$PWD/scripts/maniskill/ppo_wandb_repro/configs/PickCube-v1.json \
#     scripts/maniskill/carc_ppo_wandb_repro_single_env.sbatch
#
# Bulk submit (uses this template):
#   ./scripts/maniskill/submit_ppo_wandb_repro_carc.sh
#
# GPU: default is ``--gres=gpu:1`` (matches ``salloc`` examples in carc_usage). Some Endeavour
# condo queues want a device type instead, e.g. replace the gres line with:
#   #SBATCH --gpus=h200:1
# (see consolidated.md §5 / batch script example near the end of that file.)
#
# Discovery shared GPU: you may need ``--partition=gpu`` and a different account; ask CARC docs:
#   https://www.carc.usc.edu/user-guides/hpc-systems/discovery/getting-started-discovery
#
# Optional export env: TRACK_WANDB, ICL_ROOT, SEED, WANDB_ENTITY, MANISKILL_VENV

#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

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
