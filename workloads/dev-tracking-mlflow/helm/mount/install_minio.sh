#!/bin/bash

echo '--------------------------------------------'
echo 'Installing MinIO Client'
echo '--------------------------------------------'

# Download MinIO client binary
apt-get update && apt-get install -y curl
curl -s https://dl.min.io/client/mc/release/linux-amd64/mc \
    --create-dirs \
    -o /usr/local/bin/mc

# Make the binary executable
chmod +x /usr/local/bin/mc

# Configure MinIO alias
mc alias set minio "${MLFLOW_S3_ENDPOINT_URL}" "${AWS_ACCESS_KEY_ID}" "${AWS_SECRET_ACCESS_KEY}"

echo '--------------------------------------------'
echo 'MinIO Client Installed!'
echo '--------------------------------------------'

# check if MLFLOW_ARTIFACTS_DESTINATION starts with s3://
if [[ "${MLFLOW_ARTIFACTS_DESTINATION}" == s3://* ]]; then
    echo '--------------------------------------------'
    echo 'Configuring MinIO for MLFLOW_ARTIFACTS_DESTINATION'
    echo '--------------------------------------------'

    # Extract bucket name from MLFLOW_ARTIFACTS_DESTINATION
    BUCKET_NAME=${MLFLOW_ARTIFACTS_DESTINATION#s3://}

    # Create the bucket if it does not exist
    if ! mc mb minio/"${BUCKET_NAME}"; then
        # Check if the failure was because bucket already exists
        if mc ls minio/"${BUCKET_NAME}" >/dev/null 2>&1; then
            echo "Bucket ${BUCKET_NAME} already exists."
        else
            echo "Error: Failed to create bucket ${BUCKET_NAME} and bucket does not exist."
            exit 1
        fi
    fi

    echo '--------------------------------------------'
    echo 'MinIO configured for MLFLOW_ARTIFACTS_DESTINATION!'
    echo '--------------------------------------------'
fi
