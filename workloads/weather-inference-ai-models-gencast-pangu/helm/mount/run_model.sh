#!/bin/bash

# Pre allocate memory fraction for XLA
# Setting to 0.3 for MI300
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.30

# Turns off mem preallocation
# export XLA_PYTHON_CLIENT_PREALLOCATE=false

# dynamic mem allocation
# export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Log level for TF, setting to 2 leaves only errors
# good to have with profiling, otherwise too many warnings
export TF_CPP_MIN_LOG_LEVEL=2

# Affects how many concurrent kernel executions or data transfers can be scheduled on the GPU.
# export GPU_MAX_HW_QUEUES=2

# Parse cmd line args
for ARGUMENT in "$@"; do
   ARG="${ARGUMENT#--}"
   KEY="${ARG%%=*}"
   VALUE="${ARG#*=}"
   export "$KEY"="$VALUE"
done

# Set default values
MODEL_NAME="${model_name:-gencast}"
LEAD_TIME="${lead_time:-24}"
VISUALIZE="${visualize:-true}"
MODEL_OUT_DIR="${model_out_dir:-/workspace/model_out}"
LOG_DIR="${log_dir:-$MODEL_OUT_DIR/logs}"
PRED_DIR="${pred_dir:-$MODEL_OUT_DIR/predictions}"
ASSETS_DIR="${assets_dir:-/workspace/assets}"
JAX_PROFILER="${jax_profiler:-false}"
NUM_ENSEMBLE_MEMBERS="${num_ensemble_members:-1}"
MOUNT_DIR="${mountdir:-/workload/mount}"
DATE="${date:-20230110}"
TIME="${time:-0000}"

# Model type & parameter validation
if [[ "$MODEL_NAME" == *gencast* ]]; then
  MODEL_TYPE="gencast"
elif [[ "$MODEL_NAME" == *panguweather* ]]; then
  MODEL_TYPE="panguweather"
else
  echo "Error: Only 'gencast' and 'panguweather' models are supported."
  exit 1
fi

if [[ "$JAX_PROFILER" == "true" && "$MODEL_TYPE" != "gencast" ]]; then
  echo "Profiling is only supported for jax based models. Disabling profiler."
  JAX_PROFILER="false"
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$LOG_DIR/${MODEL_NAME}_${TIMESTAMP}"
TRACE_DIR="$LOG_DIR/traces"
LOG_FILE="$LOG_DIR/output.log"
XLA_DUMP_PATH="$LOG_DIR/xla_dumps"

mkdir -p "$MODEL_OUT_DIR"
mkdir -p "$PRED_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$XLA_DUMP_PATH"

cp "$0" "$LOG_DIR/$(basename "$0" .sh).sh"

echo "Starting model run for $MODEL_NAME at $(date)" | tee -a "$LOG_FILE"

{
  CMD_ARGS=(--download-assets
            --assets $ASSETS_DIR/$MODEL_NAME
            --input cds
            --date "$DATE"
            --path $PRED_DIR/$MODEL_NAME.grib
            --time "$TIME"
            --lead-time "$LEAD_TIME"
            "$MODEL_NAME")


if [ "$NUM_ENSEMBLE_MEMBERS" -gt 1 ]; then
  CMD_ARGS+=(--num-ensemble-members "$NUM_ENSEMBLE_MEMBERS")
fi

  if [ "$JAX_PROFILER" = "true" ]; then
    export PROFILE_TB=1
    export XLA_FLAGS="--xla_gpu_enable_command_buffer='' --xla_dump_to=$XLA_DUMP_PATH"
    CMD_ARGS+=(--jax-profiler)
    CMD_ARGS+=(--trace-dir "$TRACE_DIR")
  fi

  echo "XLA_FLAGS: $XLA_FLAGS" | tee -a "$LOG_FILE"

  ai-models "${CMD_ARGS[@]}"

  # Conditional visualization based on the VISUALIZE variable
  if [ "$VISUALIZE" = "true" ]; then
    echo "visualizing"
    python3 $MOUNT_DIR/grib_visualizer.py --input "$PRED_DIR/$MODEL_NAME.grib" --output "$PRED_DIR/gifs_$MODEL_NAME"
  fi

  # Upload results to minio
  echo "Uploading results to minio..."
  MINIO_BUCKET="${minio_bucket:-default-bucket}"
  MINIO_PREFIX="${minio_prefix:-$MODEL_NAME/$TIMESTAMP}"

  python3 $MOUNT_DIR/minio_uploader.py "$MODEL_OUT_DIR" "$MINIO_BUCKET" "$MINIO_PREFIX"

  # Print done
  echo "Done!"
  echo "Log saved to: $LOG_FILE"
} 2>&1 | tee -a "$LOG_FILE"
