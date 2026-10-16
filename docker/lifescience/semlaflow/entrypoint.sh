#!/bin/bash

SCRIPT="$1"
OUTPUT_FILE="$2"
shift
shift

echo ${OUTPUT_FILE}
# Check if SCRIPT is one of the allowed values
if [[ "$SCRIPT" != "preprocess" && "$SCRIPT" != "train" && "$SCRIPT" != "evaluate" && "$SCRIPT" != "predict" ]]; then
  echo "Error: SCRIPT must be one of 'preprocess', 'train', 'evaluate', or 'predict'."
  exit 1
fi

python -m semlaflow."$SCRIPT" "$@" &> /output/${OUTPUT_FILE}

if [[ "$SCRIPT" == "train" ]]; then
  # Copy the model checkpoint to the output directory
  cp -r lightning_logs/version_* /output/
fi
