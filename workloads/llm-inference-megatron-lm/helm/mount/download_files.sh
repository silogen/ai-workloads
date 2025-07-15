#!/bin/bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <model_path> <tokenizer_path>"
  echo "<model_path>: ..."
  echo "<tokenizer_path>: ..."
  exit 1
fi
echo "Running resources download with the following parameters:"
echo "model_path: $1"
echo "tokenizer_path: $2"

# Parameters
model_path="$1"
tokenizer_path="$2"

echo '--------------------------------------------'
echo 'Installing minio client'
echo '--------------------------------------------'
curl https://dl.min.io/client/mc/release/linux-amd64/mc \
      --create-dirs \
      -o /minio-binaries/mc
chmod +x /minio-binaries/mc
export PATH="${PATH}:/minio-binaries/"
mc alias set minio-host ${BUCKET_STORAGE_HOST} ${BUCKET_STORAGE_ACCESS_KEY} ${BUCKET_STORAGE_SECRET_KEY}
echo '--------------------------------------------'
echo 'Downloading the model-checkpoint to the local container'
echo '--------------------------------------------'
mc cp --recursive minio-host/$model_path /workload/model/
echo '--------------------------------------------'
echo 'Downloading the tokenizer to the local container'
echo '--------------------------------------------'
mc cp minio-host/$tokenizer_path/special_tokens_map.json /workload/tokenizer/special_tokens_map.json
mc cp minio-host/$tokenizer_path/tokenizer.json /workload/tokenizer/tokenizer.json
mc cp minio-host/$tokenizer_path/tokenizer_config.json /workload/tokenizer/tokenizer_config.json
