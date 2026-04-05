#!/usr/bin/env bash
# Reproduce official ManiSkill W&B PPO state results (walltime_efficient, 3 seeds per env).
# Report: https://wandb.ai/stonet2000/ManiSkill/reports/PPO-Results--VmlldzoxMDQzNDMzOA
#
# Hyperparameters live in scripts/maniskill/ppo_wandb_repro/configs/<EnvId>.json
# (refresh: uv run python scripts/maniskill/fetch_ppo_wandb_configs.py).
#
# Requires: ManiSkill venv (see scripts/maniskill/requirements.txt), repo root as cwd.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MANISKILL_VENV="${MANISKILL_VENV:-${ROOT}/.venv-maniskill}"
if [ -x "${MANISKILL_VENV}/bin/python" ]; then
  export MANISKILL_PYTHON="${MANISKILL_PYTHON:-${MANISKILL_VENV}/bin/python}"
fi

# Default batch list: scripts/maniskill/ppo_wandb_repro/default_tabletop_repro_envs.txt
# ENVS_OVERRIDE still accepts any env that has configs/<id>.json.
ENVS_OVERRIDE="${ENVS_OVERRIDE:-}"
DEFAULT_ENV_FILE="${ROOT}/scripts/maniskill/ppo_wandb_repro/default_tabletop_repro_envs.txt"

DRY=""
if [ "${DRY_RUN:-0}" != "0" ]; then
  DRY="--dry-run"
fi

if [ -n "${SEED:-}" ]; then
  SEED_ARG=(--seed "${SEED}")
else
  SEED_ARG=()
fi

if [ -n "${ENVS_OVERRIDE}" ]; then
  for e in ${ENVS_OVERRIDE}; do
    CFG="${ROOT}/scripts/maniskill/ppo_wandb_repro/configs/${e}.json"
    if [ ! -f "${CFG}" ]; then
      echo "No W&B config for ${e} (${CFG}). Refresh configs or pick an env from manifest.json." >&2
      exit 1
    fi
    uv run python "${ROOT}/scripts/maniskill/run_ppo_wandb_repro.py" \
      ${DRY} --config "${CFG}" "${SEED_ARG[@]}" "$@"
  done
else
  if [ ! -f "${DEFAULT_ENV_FILE}" ]; then
    echo "Missing default env list: ${DEFAULT_ENV_FILE}" >&2
    exit 1
  fi
  while IFS= read -r e; do
    CFG="${ROOT}/scripts/maniskill/ppo_wandb_repro/configs/${e}.json"
    if [ ! -f "${CFG}" ]; then
      echo "skip missing config: ${e}" >&2
      continue
    fi
    uv run python "${ROOT}/scripts/maniskill/run_ppo_wandb_repro.py" \
      ${DRY} --config "${CFG}" "${SEED_ARG[@]}" "$@"
  done < <(awk '!/^[[:space:]]*#/ && NF { print $1 }' "${DEFAULT_ENV_FILE}")
fi
