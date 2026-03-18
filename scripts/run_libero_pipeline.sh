#!/usr/bin/env bash
# Run the full LIBERO data pipeline: HDF5 → episodes + manifest → sample index → precomputed embeddings.
#
# Usage:
#   ./scripts/run_libero_pipeline.sh <input_dir> [output_dir]
#   ./scripts/run_libero_pipeline.sh /path/to/LIBERO-Cosmos-Policy
#   ./scripts/run_libero_pipeline.sh /path/to/LIBERO-Cosmos-Policy /path/to/datasets
#
# Steps:
#   1. convert_libero_hdf5_to_dataset.py: HDF5 → episodes/ + manifest.parquet
#   2. build_libero_sample_index.py: manifest → sample_index.parquet
#   3. precompute_libero_embeddings.py: episodes/ → embeddings.npz per episode
#
# Then train with:
#   uv run python -m src.train data=base,libero_cosmos paths.data_root=<output_dir> data.use_vision=true data.use_precomputed_embeddings=true model=vla_dt ...
#
# Requires uv (pip install uv or https://github.com/astral-sh/uv).

set -e

INPUT_DIR="${1:?Usage: $0 <input_dir> [output_dir]}"
OUTPUT_DIR="${2:-$INPUT_DIR}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== LIBERO pipeline: input=$INPUT_DIR output=$OUTPUT_DIR ==="

echo "[1/3] Converting HDF5 → episodes + manifest..."
uv run python scripts/convert_libero_hdf5_to_dataset.py \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR"

echo "[2/3] Building sample index..."
uv run python scripts/build_libero_sample_index.py --data-dir "$OUTPUT_DIR"

echo "[3/3] Precomputing vision embeddings..."
uv run python scripts/precompute_libero_embeddings.py --data-dir "$OUTPUT_DIR"

echo "=== Done. Train with (precomputed embeddings; no vision encoder loaded): ==="
echo "  uv run python -m src.train --override data=[base,libero_cosmos] model=vla_dt paths.data_root=$OUTPUT_DIR data.use_vision=true data.use_precomputed_embeddings=true model.vision_encoder_type=dinov2 model.precomputed_vision_embed_dim=1536"
