#!/bin/bash

set -e  # Exit on any error

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Script directory: $SCRIPT_DIR"

# Download the dataset
cd "$SCRIPT_DIR/gsplat/examples"
python "$SCRIPT_DIR/gsplat/examples/datasets/download_dataset.py"

# Run Single-GPU benchmark
bash "$SCRIPT_DIR/gsplat/examples/benchmarks/basic.sh"
BENCHMARK_EXIT_CODE=$?

# Now check the captured exit code
if [ $BENCHMARK_EXIT_CODE -eq 0 ]; then
  echo "Benchmark completed at $(date) with exit code: $BENCHMARK_EXIT_CODE"
else
  echo "Benchmark failed at $(date) with exit code: $BENCHMARK_EXIT_CODE"
fi

# Reformat output metrics to CSV and markdown tables
python "$SCRIPT_DIR/create_scene_metrics_tables.py"

# Print the two outputs from the reformat_log.py script.
cat "scene_metrics_table.md"
cat "num_gs_table.md"

echo "Job completed at $(date)"
