#!/usr/bin/env bash
# Train from scratch (single GPU). Override config from CLI.
set -e
cd "$(dirname "$0")/.."
python -m src.train \
  --config-dir "$(pwd)/configs" \
  "$@"
