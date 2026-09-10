#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Change directory to the location of this script
cd "$(dirname "$0")"

echo "---------------------------------"
echo "Installing MinIO Client (mc)..."
echo "---------------------------------"

# Download and install MinIO client for linux-amd64
curl -L https://dl.min.io/client/mc/release/linux-amd64/mc -o /usr/local/bin/mc
chmod +x /usr/local/bin/mc

# Verify installation
echo "✓ MinIO client installed successfully"
mc --version

# Configure MinIO alias if credentials are provided
if [[ -n "$BUCKET_STORAGE_HOST" && -n "$BUCKET_STORAGE_ACCESS_KEY" && -n "$BUCKET_STORAGE_SECRET_KEY" ]]; then
    echo "Configuring MinIO alias..."
    mc alias set minio-host "$BUCKET_STORAGE_HOST" "$BUCKET_STORAGE_ACCESS_KEY" "$BUCKET_STORAGE_SECRET_KEY"
    echo "✓ MinIO connection configured"
else
    echo "MinIO credentials not provided - skipping alias configuration"
fi

echo "---------------------------------"
echo "Installing Python dependencies..."
echo "---------------------------------"

pip install --no-cache-dir -r requirements.txt

echo "✓ Python dependencies installed successfully"

# Construct the training command from environment variables
CMD_ARGS=""

# Loop through all environment variables with the prefix 'TRAINING_ARG_'
for var in $(env | grep "^TRAINING_ARG_"); do
    # Extract the original key and value
    KEY_UPPER=$(echo "$var" | sed -e 's/=.*//' -e 's/TRAINING_ARG_//')
    VALUE=$(echo "$var" | sed 's/.*=//')

    KEY_LOWER=$(echo "$KEY_UPPER" | tr '[:upper:]' '[:lower:]')

    # Handle boolean flags (e.g., --save_checkpoint)
    if [[ "$VALUE" == "true" ]]; then
        CMD_ARGS="$CMD_ARGS --$KEY_LOWER"
    # Handle key-value pairs (e.g., --batch_size=1)
    # This ignores arguments that are explicitly set to 'false' or are empty
    elif [[ "$VALUE" != "false" && -n "$VALUE" ]]; then
        CMD_ARGS="$CMD_ARGS --$KEY_LOWER=$VALUE"
    fi
done

# Build the final command
COMMAND="python main.py $CMD_ARGS"

echo "---------------------------------"
echo "Executing command:"
echo "$COMMAND"
echo "---------------------------------"

# Execute the command
eval "$COMMAND"
