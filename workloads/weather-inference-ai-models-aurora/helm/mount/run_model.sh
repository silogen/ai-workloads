#!/bin/bash


# Parse cmd line args
for ARGUMENT in "$@"; do
   ARG="${ARGUMENT#--}"
   KEY="${ARG%%=*}"
   VALUE="${ARG#*=}"
   export "$KEY"="$VALUE"
done

# Set default values
MODEL_NAME="${model_name:-aurora}"
LEAD_TIME="${lead_time:-24}"
VISUALIZE="${visualize:-true}"
MODEL_OUT_DIR="${model_out_dir:-/workspace/model_out}"
LOG_DIR="${log_dir:-$MODEL_OUT_DIR/logs}"
PRED_DIR="${pred_dir:-$MODEL_OUT_DIR/predictions}"
ASSETS_DIR="${assets_dir:-/workspace/assets}"
MOUNT_DIR="${mountdir:-/workload/mount}"
DATE="${date:-20230110}"
TIME="${time:-0000}"

# Model type & parameter validation
if [[ "$MODEL_NAME" == *aurora* ]]; then
  MODEL_TYPE="aurora"
else
  echo "Error: Only 'aurora' model family is supported."
  exit 1
fi


TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$LOG_DIR/${MODEL_NAME}_${TIMESTAMP}"
TRACE_DIR="$LOG_DIR/traces"
LOG_FILE="$LOG_DIR/output.log"

mkdir -p "$MODEL_OUT_DIR"
mkdir -p "$PRED_DIR"
mkdir -p "$LOG_DIR"

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
