#!/usr/bin/env bash
# Train PPO on several ManiSkill tasks and export ICL trajectories.h5 per task.
# Requires: ManiSkill venv (scripts/maniskill/requirements.txt) + PYTHONPATH=repo root; Vulkan/GPU per ManiSkill docs.
#
# Local (default): runs python in a loop on this machine.
# Slurm: SUBMIT_SLURM=1 materializes scripts/maniskill/ppo_single_env.sbatch + metadata into
#   SLURM_LOG_DIR (default: <repo>/slurm-logs) as <job-name>-<jobid>.sbatch, then sbatch's that file.
#   Stdout/stderr: same dir, %x-%j.out / %x-%j.err. Override account/partition via SBATCH_EXTRA.
# W&B-matched JSON hparams (separate template): scripts/maniskill/submit_ppo_wandb_repro_slurm.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# ManiSkill-only venv (default: repo-local; e.g. /scr/aliang80/icl_adaptation/.venv-maniskill).
MANISKILL_VENV="${MANISKILL_VENV:-${ROOT}/.venv-maniskill}"
if [ -x "${MANISKILL_VENV}/bin/python" ]; then
  PYTHON="${MANISKILL_VENV}/bin/python"
else
  PYTHON="${PYTHON:-python}"
fi

SUBMIT_SLURM="${SUBMIT_SLURM:-0}"
# Extra sbatch CLI args, e.g. SBATCH_EXTRA=(--account=myproj --partition=gpu)
SBATCH_EXTRA=(${SBATCH_EXTRA:-})

# Default envs: same table-top + repro list as W&B repro (ppo_wandb_repro/default_tabletop_repro_envs.txt).
# This script uses fixed CLI hparams; for JSON-matched hparams use run_ppo_wandb_repro.sh / submit_ppo_wandb_repro_slurm.sh.
#
# Override with ENVS_OVERRIDE="PushCube-v1" or any space-separated list.
ENVS_OVERRIDE="${ENVS_OVERRIDE:-}"
DEFAULT_ENV_FILE="${ROOT}/scripts/maniskill/ppo_wandb_repro/default_tabletop_repro_envs.txt"
if [ -n "${ENVS_OVERRIDE}" ]; then
  # shellcheck disable=SC2206
  ENVS=( ${ENVS_OVERRIDE} )
else
  if [ ! -f "${DEFAULT_ENV_FILE}" ]; then
    echo "Missing default env list: ${DEFAULT_ENV_FILE}" >&2
    exit 1
  fi
  mapfile -t ENVS < <(awk '!/^[[:space:]]*#/ && NF { print $1 }' "${DEFAULT_ENV_FILE}")
  if [ "${#ENVS[@]}" -eq 0 ]; then
    echo "No env ids parsed from ${DEFAULT_ENV_FILE}" >&2
    exit 1
  fi
fi

NUM_ENVS="${NUM_ENVS:-1024}"
STEPS="${STEPS:-50_000_000}"
# Set TRACK_WANDB=1 to log to W&B (default off so many-env loops do not spam runs).
TRACK_WANDB="${TRACK_WANDB:-0}"
COLLECT_EP="${COLLECT_EP:-0}"
# Set SNAP_EVERY>0 for RGB shards trajectories_image_shard_*.h5 in maniskill/<env_id>/.
# SNAP_SHARD_MAX: episodes per snapshot HDF5 (default 10000). Also set ICL_SHARD_MAX to match if you shard state rollouts.
SNAP_EVERY="${SNAP_EVERY:-0}"
SNAP_EP="${SNAP_EP:-8}"
SNAP_SHARD_MAX="${SNAP_SHARD_MAX:-10000}"
ICL_ROOT="${ICL_ROOT:-datasets}"
SBATCH_SCRIPT="${ROOT}/scripts/maniskill/ppo_single_env.sbatch"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-${ROOT}/slurm-logs}"

if [ "${SUBMIT_SLURM}" = "1" ]; then
  if ! command -v sbatch >/dev/null 2>&1; then
    echo "SUBMIT_SLURM=1 but sbatch not found." >&2
    exit 1
  fi
  mkdir -p "${SLURM_LOG_DIR}"
  n="${#ENVS[@]}"
  echo "Submitting ${n} Slurm job(s) (${SBATCH_SCRIPT}); logs -> ${SLURM_LOG_DIR}/"
  for env_id in "${ENVS[@]}"; do
    safe_name="ms_${env_id//[^A-Za-z0-9]/_}"
    safe_name="${safe_name:0:64}"
    exp="ALL,REPO_ROOT=${ROOT},ENV_ID=${env_id},NUM_ENVS=${NUM_ENVS},STEPS=${STEPS},TRACK_WANDB=${TRACK_WANDB},COLLECT_EP=${COLLECT_EP},SNAP_EVERY=${SNAP_EVERY},SNAP_EP=${SNAP_EP},SNAP_SHARD_MAX=${SNAP_SHARD_MAX},ICL_ROOT=${ICL_ROOT},MANISKILL_VENV=${MANISKILL_VENV}"
    pending="${SLURM_LOG_DIR}/.pending-${safe_name}-$$.sbatch"
    {
      head -n 1 "${SBATCH_SCRIPT}"
      echo "#"
      echo "# --- injected by train_ppo_many_envs.sh ($(date -Is)) ---"
      echo "# template: ${SBATCH_SCRIPT}"
      echo "# user: ${USER:-?} host: $(hostname 2>/dev/null || echo '?')"
      echo "#"
      echo "# Reconstructed sbatch invocation (this file was passed as the script argument):"
      printf '#   sbatch'
      for a in "${SBATCH_EXTRA[@]}"; do printf ' %q' "$a"; done
      printf ' %q' "--job-name=${safe_name}"
      printf ' %q' "--export=${exp}"
      printf ' %q' "--output=${SLURM_LOG_DIR}/%x-%j.out"
      printf ' %q' "--error=${SLURM_LOG_DIR}/%x-%j.err"
      echo "#       <this file; renamed after submit to ${safe_name}-___JOBID___.sbatch>"
      echo "# --- end injection; template continues below ---"
      echo
      tail -n +2 "${SBATCH_SCRIPT}"
    } > "${pending}"
    jobid="$(sbatch --parsable "${SBATCH_EXTRA[@]}" \
      --job-name="${safe_name}" \
      --export="${exp}" \
      --output="${SLURM_LOG_DIR}/%x-%j.out" \
      --error="${SLURM_LOG_DIR}/%x-%j.err" \
      "${pending}" | cut -d';' -f1)"
    if [ -z "${jobid}" ]; then
      echo "sbatch failed or returned empty job id for ${env_id} (see ${pending})" >&2
      exit 1
    fi
    final="${SLURM_LOG_DIR}/${safe_name}-${jobid}.sbatch"
    mv "${pending}" "${final}"
    sed -i "s/___JOBID___/${jobid}/g" "${final}"
    echo "  job ${jobid} -> ${final}"
  done
  echo "Submitted ${n} job(s). Check: squeue -u \"\${USER}\""
  echo "Copies of batch scripts + .out/.err: ${SLURM_LOG_DIR}/"
  exit 0
fi

for env_id in "${ENVS[@]}"; do
  echo "========== PPO + ICL export: ${env_id} =========="
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
  "${PYTHON}" scripts/maniskill/ppo_train_icldata.py \
    --env-id "${env_id}" \
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
done

echo "Done. State trajectories: ${ICL_ROOT}/maniskill/<env_id>/trajectories.h5"
echo "RGB snapshot shards (if SNAP_EVERY set): ${ICL_ROOT}/maniskill/<env_id>/trajectories_image_shard_*.h5"
