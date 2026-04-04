#!/usr/bin/env bash
# Submit one Slurm job per JSON in scripts/maniskill/ppo_wandb_repro/configs/ (W&B-matched PPO hparams).
# Each job runs run_ppo_wandb_repro.py for that file (three seeds by default, unless SEED is set).
#
# From repo root:
#   SBATCH_EXTRA=(--account=YOUR_ACCT --partition=gpu) ./scripts/maniskill/submit_ppo_wandb_repro_slurm.sh
#
# Optional env:
#   ENVS_OVERRIDE="PickCube-v1 PushCube-v1"   # subset (must match *.json basenames without .json)
#   TRACK_WANDB=0                             # add --no-track on workers (default 1)
#   SEED=1788                                 # one seed per job instead of all seeds in JSON
#   WANDB_ENTITY=your_team                    # forwarded to ppo_train_icldata.py
#   ICL_ROOT=/path/to/datasets
#   SLURM_LOG_DIR, MANISKILL_VENV, SBATCH_EXTRA (array)
#
# For the fixed-hparam multi-env launcher (not JSON repro), use:
#   SUBMIT_SLURM=1 ./scripts/maniskill/train_ppo_many_envs.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
shopt -s nullglob

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found." >&2
  exit 1
fi

SBATCH_SCRIPT="${ROOT}/scripts/maniskill/ppo_wandb_repro_single_env.sbatch"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-${ROOT}/slurm-logs}"
CONFIG_DIR="${ROOT}/scripts/maniskill/ppo_wandb_repro/configs"
SBATCH_EXTRA=(${SBATCH_EXTRA:-})
ENVS_OVERRIDE="${ENVS_OVERRIDE:-}"
TRACK_WANDB="${TRACK_WANDB:-1}"
ICL_ROOT="${ICL_ROOT:-datasets}"
MANISKILL_VENV="${MANISKILL_VENV:-${ROOT}/.venv-maniskill}"
SEED="${SEED:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

mkdir -p "${SLURM_LOG_DIR}"

collect_configs() {
  local -a out=()
  if [ -n "${ENVS_OVERRIDE}" ]; then
    local e
    for e in ${ENVS_OVERRIDE}; do
      local p="${CONFIG_DIR}/${e}.json"
      if [ ! -f "${p}" ]; then
        echo "Missing config: ${p}" >&2
        exit 1
      fi
      out+=("${p}")
    done
  else
    local f
    for f in "${CONFIG_DIR}"/*.json; do
      [ -f "${f}" ] || continue
      [ "$(basename "${f}")" = "manifest.json" ] && continue
      out+=("${f}")
    done
  fi
  printf '%s\n' "${out[@]}"
}

mapfile -t CONFIGS < <(collect_configs)
n="${#CONFIGS[@]}"
if [ "${n}" -eq 0 ]; then
  echo "No JSON configs under ${CONFIG_DIR} (run fetch_ppo_wandb_configs.py first)." >&2
  exit 1
fi

echo "Submitting ${n} Slurm job(s) (${SBATCH_SCRIPT}); logs -> ${SLURM_LOG_DIR}/"

for cfg in "${CONFIGS[@]}"; do
  base="$(basename "${cfg}" .json)"
  safe_name="ms_repro_${base//[^A-Za-z0-9]/_}"
  safe_name="${safe_name:0:64}"
  exp="ALL,REPO_ROOT=${ROOT},PPO_WANDB_CONFIG=${cfg},TRACK_WANDB=${TRACK_WANDB},ICL_ROOT=${ICL_ROOT},MANISKILL_VENV=${MANISKILL_VENV}"
  [ -n "${SEED}" ] && exp="${exp},SEED=${SEED}"
  [ -n "${WANDB_ENTITY}" ] && exp="${exp},WANDB_ENTITY=${WANDB_ENTITY}"

  pending="${SLURM_LOG_DIR}/.pending-${safe_name}-$$.sbatch"
  {
    head -n 1 "${SBATCH_SCRIPT}"
    echo "#"
    echo "# --- injected by submit_ppo_wandb_repro_slurm.sh ($(date -Is)) ---"
    echo "# template: ${SBATCH_SCRIPT}"
    echo "# PPO_WANDB_CONFIG=${cfg}"
    echo "#"
    printf '#   sbatch'
    for a in "${SBATCH_EXTRA[@]}"; do printf ' %q' "$a"; done
    printf ' %q' "--job-name=${safe_name}"
    printf ' %q' "--export=${exp}"
    printf ' %q' "--output=${SLURM_LOG_DIR}/%x-%j.out"
    printf ' %q' "--error=${SLURM_LOG_DIR}/%x-%j.err"
    echo
    echo "# --- end injection ---"
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
    echo "sbatch failed for ${cfg} (see ${pending})" >&2
    exit 1
  fi
  final="${SLURM_LOG_DIR}/${safe_name}-${jobid}.sbatch"
  mv "${pending}" "${final}"
  echo "  job ${jobid} (${base}) -> ${final}"
done

echo "Submitted ${n} job(s). Check: squeue -u \"\${USER}\""
