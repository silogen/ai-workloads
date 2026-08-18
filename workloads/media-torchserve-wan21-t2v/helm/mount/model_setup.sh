#!/bin/bash
set -e  # Exit on any error

echo "Starting model setup ... "

# Check required environment variables
if [ -z "$BUCKET_PATH" ] || [ -z "$MODEL_ID" ]; then
    echo "ERROR: BUCKET_PATH and MODEL_ID environment variables must be set"
    exit 1
fi

# Download model.zip from MinIO
echo "Downloading model from minio-host/$BUCKET_PATH/$MODEL_ID/packaged/model.zip..."

if mc cp minio-host/$BUCKET_PATH/$MODEL_ID/packaged/model.zip model.zip 2>/dev/null; then
    echo "Model.zip downloaded successfully ($(du -h model.zip | cut -f1))"
else
    echo "ERROR: model.zip not found at minio-host/$BUCKET_PATH/$MODEL_ID/packaged/model.zip"
    echo "Please ensure the torchserve-model-packager workload has completed successfully"
    exit 1
fi

# Create model store directory
mkdir -p model_store
echo "Model store directory ready"

# Create model archive
echo "Creating model archive with torch-model-archiver..."
if torch-model-archiver --model-name wan21 --version 1.0 --export-path model_store --handler wan_handler.py --extra-files model.zip; then
    echo "Model archive created successfully"
else
    echo "ERROR: Failed to create model archive"
    exit 1
fi

# Verify the archive was created
if [ -f "model_store/wan21.mar" ]; then
    echo "Model archive verification passed: wan21.mar created ($(du -h model_store/wan21.mar | cut -f1))"
else
    echo "ERROR: Expected model archive wan21.mar not found in model_store/"
    exit 1
fi

echo "Model setup completed successfully!"
