#!/usr/bin/env bash
# Run the full RoboArena data pipeline: DataDump → episodes + manifest → dense rewards → viz → sample index → precomputed embeddings.
#
# Usage:
#   ./scripts/run_roboarena_pipeline.sh <input_dir> [output_dir]
#   ./scripts/run_roboarena_pipeline.sh /path/to/DataDump_08-05-2025
#   ./scripts/run_roboarena_pipeline.sh /path/to/DataDump_08-05-2025 /path/to/datasets
#
# Steps:
#   1. convert_roboarena_to_dataset.py: DataDump evaluation_sessions → episodes/<task_slug>/<idx>/ + manifest.parquet
#   2. compute_dense_rewards.py: episodes/ → dense reward arrays (lowdim.npz)
#   3. viz_relabeled_returns.py: histogram + example trajectories (viz_relabeled/)
#   4. build_libero_sample_index.py: manifest → sample_index.parquet (works with any manifest with episode_id, task_description, n_steps, success)
#   5. precompute_libero_embeddings.py: episodes/ → embeddings.npz per episode
#
# Then train with:
#   uv run python -m src.train data=base,<roboarena_config> paths.data_root=<output_dir> data.use_vision=true data.use_precomputed_embeddings=true model=vla_dt ...
#
# Requires uv (pip install uv or https://github.com/astral-sh/uv).

set -e

INPUT_DIR="${1:?Usage: $0 <input_dir> [output_dir]}"
OUTPUT_DIR="${2:-$INPUT_DIR}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== RoboArena pipeline: input=$INPUT_DIR output=$OUTPUT_DIR ==="

echo "[1/5] Converting DataDump → episodes + manifest..."
uv run python scripts/convert_roboarena_to_dataset.py \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR"

echo "[2/5] Computing dense reward signals (offline)..."
uv run python scripts/compute_dense_rewards.py \
  --data-dir "$OUTPUT_DIR" \
  --models robometer_4b \
  --overwrite-rewards robometer_4b \
  --cam primary

echo "[3/5] Visualizing relabeled returns (histogram + examples)..."
uv run python scripts/viz_relabeled_returns.py --data-dir "$OUTPUT_DIR"

echo "[4/5] Building sample index..."
uv run python scripts/build_libero_sample_index.py --data-dir "$OUTPUT_DIR"

echo "[5/5] Precomputing vision embeddings..."
uv run python scripts/precompute_libero_embeddings.py --data-dir "$OUTPUT_DIR"

echo "=== Done. Train with (precomputed embeddings): ==="
echo "  uv run python -m src.train --override data=[base,<roboarena>] model=vla_dt paths.data_root=$OUTPUT_DIR data.use_vision=true data.use_precomputed_embeddings=true model.vision_encoder_type=dinov2 model.precomputed_vision_embed_dim=1536"
