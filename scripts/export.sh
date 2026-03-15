#!/usr/bin/env bash
# Save inference artifact from a training checkpoint.
set -e
cd "$(dirname "$0")/.."
CKPT="${1:-outputs/checkpoints/checkpoint_best.pt}"
python -m src.train \
  --config-dir "$(pwd)/configs" \
  --export-only "$CKPT"
