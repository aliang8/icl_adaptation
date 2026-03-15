#!/usr/bin/env bash
# Run evaluation only (no training).
set -e
cd "$(dirname "$0")/.."
python -m src.train \
  --config-dir "$(pwd)/configs" \
  --eval-only \
  --resume "${1:-outputs/checkpoints/checkpoint_best.pt}" \
  "${@:2}"
