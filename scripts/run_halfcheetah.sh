#!/usr/bin/env bash
# 1) Download HalfCheetah via Minari (mixed expertise), 2) Train Meta-DT with W&B.
# Uses `uv run python` if uv is available, otherwise `python`.
set -e
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"
if command -v uv >/dev/null 2>&1; then
  PY="uv run python"
else
  PY="python"
fi

# Download datasets via Minari (medium, expert, and combined medium_expert)
echo "=== Downloading HalfCheetah datasets (Minari) ==="
$PY scripts/download_d4rl_halfcheetah.py \
  --output-dir "${PROJECT_ROOT}/datasets" \
  --qualities medium expert medium_expert

# Train with HalfCheetah config + W&B
echo "=== Training Meta-DT on HalfCheetah (W&B enabled) ==="
$PY -m src.train \
  --config-dir "${PROJECT_ROOT}/configs" \
  --wandb \
  --run-name "halfcheetah-medium_expert" \
  --override "data=[base,halfcheetah]" \
  "$@"
