#!/bin/bash

CONFIG_FILE="$1"
OUTPUT_FILE="$2"

# Check if OUTPUT ends with ".log" and append it if not
if [[ ! "$OUTPUT_FILE" =~ .log$ ]]; then
    OUTPUT_FILE="${OUTPUT_FILE}.log"
fi

if [ -n "$CONFIG_FILE" ] && [ -n "$OUTPUT_FILE" ]; then
    echo "Config file provided: $CONFIG_FILE"
    echo "Output file provided: $OUTPUT_FILE"
    echo "Running software in automated mode..."

    # Set the correct permissions for the config file
    chmod +r "$CONFIG_FILE"

    exec reinvent -l "$OUTPUT_FILE" "$CONFIG_FILE"

    echo "Processing complete. Results saved in $(dirname "$OUTPUT_FILE")."
else
    echo "No config file or output file provided. Starting interactive mode..."
    exec /bin/bash
fi
