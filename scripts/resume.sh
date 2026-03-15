#!/usr/bin/env bash
# Resume from latest or specific checkpoint.
set -e
cd "$(dirname "$0")/.."
CKPT="${1:-outputs/checkpoints/checkpoint_latest.pt}"
python -m src.train \
  --config-dir "$(pwd)/configs" \
  --resume "$CKPT" \
  "${@:2}"
