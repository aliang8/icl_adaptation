#!/bin/bash
# Slurm batch job: one ManiSkill env → ppo_train_icldata.py
#
# Submit from repo root (or pass REPO_ROOT via --export). Typical:
#   Default venv if MANISKILL_VENV unset: $REPO_ROOT/.venv-maniskill
#   sbatch --account=YOUR_ACCOUNT --partition=gpu \
#     --export=ALL,REPO_ROOT=$PWD,ENV_ID=PushCube-v1 \
#     scripts/maniskill/ppo_single_env.sbatch
#
# train_ppo_many_envs.sh sets SUBMIT_SLURM=1 and passes all needed vars via --export.
#
# Cluster differences: override #SBATCH lines on the sbatch CLI, or edit below.
# (CARC-style example: /scr/aliang80/lair-private/scripts/train.sbatch uses
#  --account, module load gcc, conda activate — add similar blocks after SBATCH headers.)

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# GPU: this cluster uses shard units (e.g. shard:25 ≈ half a GPU); one ManiSkill run uses shard:10.
#SBATCH --gres=shard:10
#SBATCH --mem=50G
#SBATCH --time=24:00:00
# Logs: train_ppo_many_envs.sh passes --output/--error under SLURM_LOG_DIR and saves a copy of this
# script per job as slurm-logs/<job-name>-<jobid>.sbatch (metadata + this body). Manual sbatch: set -o/-e yourself.

set -euo pipefail

: "${REPO_ROOT:?Set REPO_ROOT to icl_adaptation repo root}"
: "${ENV_ID:?Set ENV_ID (ManiSkill env id)}"

cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

NUM_ENVS="${NUM_ENVS:-1024}"
STEPS="${STEPS:-50_000_000}"
TRACK_WANDB="${TRACK_WANDB:-0}"
COLLECT_EP="${COLLECT_EP:-0}"
SNAP_EVERY="${SNAP_EVERY:-0}"
SNAP_EP="${SNAP_EP:-8}"
SNAP_SHARD_MAX="${SNAP_SHARD_MAX:-10000}"
ICL_ROOT="${ICL_ROOT:-datasets}"

MANISKILL_VENV="${MANISKILL_VENV:-${REPO_ROOT}/.venv-maniskill}"
if [ -f "${MANISKILL_VENV}/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "${MANISKILL_VENV}/bin/activate"
fi

# Optional: torch single-process defaults (see lair-private train.sbatch)
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-12345}"

SNAP_ARGS=()
if [ "${SNAP_EVERY}" != "0" ] && [ -n "${SNAP_EVERY}" ]; then
  SNAP_ARGS+=(--icl-image-snapshot-every-steps "${SNAP_EVERY}")
  SNAP_ARGS+=(--icl-image-snapshot-episodes "${SNAP_EP}")
  SNAP_ARGS+=(--icl-image-snapshot-shard-max-episodes "${SNAP_SHARD_MAX}")
fi
COLLECT_ARGS=()
if [ "${COLLECT_EP}" != "0" ] && [ -n "${COLLECT_EP}" ]; then
  COLLECT_ARGS+=(--icl-collect-episodes "${COLLECT_EP}")
fi
TRACK_ARGS=()
if [ "${TRACK_WANDB}" = "1" ]; then
  TRACK_ARGS+=(--track)
else
  TRACK_ARGS+=(--no-track)
fi

echo "Job ${SLURM_JOB_ID:-local} $(hostname) $(date)"
echo "REPO_ROOT=${REPO_ROOT} ENV_ID=${ENV_ID} python=$(command -v python)"

if command -v srun >/dev/null 2>&1; then
  _run=(srun)
else
  _run=()
fi

"${_run[@]}" python scripts/maniskill/ppo_train_icldata.py \
  --env-id "${ENV_ID}" \
  --num-envs "${NUM_ENVS}" \
  --total-timesteps "${STEPS}" \
  --update-epochs 8 \
  --num-minibatches 32 \
  --eval-freq 25 \
  --num-steps 20 \
  "${TRACK_ARGS[@]}" \
  "${SNAP_ARGS[@]}" \
  "${COLLECT_ARGS[@]}" \
  --icl-data-root "${ICL_ROOT}"

echo "Finished $(date)"
