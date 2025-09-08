#!/bin/bash

set -eu

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <tokenizer_id>"
  echo "<tokenizer_id>: Huggingface model id or s3 path that has Huggingface tokenizer files: special_tokens_map.json, tokenizer.json, tokenizer_config.json"
  echo "Example 1: $0 NousResearch/Meta-Llama-3.1-8B"
  echo "Example 2: $0 s3://default-bucket/models/meta-llama/Llama-3.1-8B"
  exit 1
fi
echo "Running data and tokenizer download with the following parameters:"
echo "tokenizer_id: $1"

# Parameters
tokenizer_id="$1"

mkdir -p /downloads/tokenizer

if [[ "$tokenizer_id" = "s3://"* ]]; then
  echo "Downloading tokenizer from S3 path: $tokenizer_id ..."
  echo "Setting up minio"
  mc alias set minio-host $BUCKET_STORAGE_HOST $BUCKET_STORAGE_ACCESS_KEY $BUCKET_STORAGE_SECRET_KEY;
  tokenizer_path="${tokenizer_id#'s3://'}"
  mc cp minio-host/$tokenizer_path/special_tokens_map.json /downloads/tokenizer/special_tokens_map.json
  mc cp minio-host/$tokenizer_path/tokenizer.json /downloads/tokenizer/tokenizer.json
  mc cp minio-host/$tokenizer_path/tokenizer_config.json /downloads/tokenizer/tokenizer_config.json
else
  echo "Downloading tokenizer from HuggingFace Hub using HuggingFace model id: $tokenizer_id ..."
  huggingface-cli download "$tokenizer_id" \
              special_tokens_map.json tokenizer.json tokenizer_config.json \
              --local-dir /downloads/tokenizer/"$tokenizer_id"
fi

echo "Downloading dataset..."
python /scripts/download_data.py --target-dir /downloads/tmp
