#!/bin/bash

set -eu

if [ "$#" -lt 5 ]; then
  echo "Usage: $0 <ds_name> <input_file_name> <output_prefix> <json_key> <tokenizer_id>"
  echo "<ds_name>: name of the dataset, will be used to create the preprocessed dataset directory"
  echo "<input_file_name>: name of the downloaded file that has to be preprocessed"
  echo "<output_prefix>: prefix of the preprocessed files, the full names of the two resulting files will be <output_prefix>_<json_key>_document.<bin|idx> with only extension differing"
  echo "<json_key>: the key in the <input_file_name> json file that contains the text to be preprocessed"
  echo "<tokenizer_id>: Huggingface model id or s3 path that has Huggingface tokenizer files: special_tokens_map.json, tokenizer.json, tokenizer_config.json"
  echo "Example: $0 fineweb-edu fineweb-edu-train.jsonl fineweb-edu-train text NousResearch/Meta-Llama-3.1-8B"
  exit 1
fi
echo "Running data preparation with the following parameters:"
echo "ds_name: $1"
echo "input_file_name: $2"
echo "output_prefix: $3"
echo "json_key: $4"
echo "tokenizer_id: $5"

# Parameters
ds_name="$1"
input_file_name="$2"
output_prefix="$3"
json_key="$4"
tokenizer_id="$5"

if [[ "$tokenizer_id" = "s3://"* ]]; then
  tokenizer_path="/downloads/tokenizer"
else
  tokenizer_path="/downloads/tokenizer/$tokenizer_id"
fi

echo "Run data pre-processing:"
python tools/preprocess_data.py \
    --input "/downloads/tmp/$input_file_name" \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model "$tokenizer_path" \
    --output-prefix "/downloads/tmp/$output_prefix" \
    --json-keys "$json_key" \
    --partitions 8 \
    --workers 8

mkdir -p /downloads/datasets/"$ds_name"
cp /downloads/tmp/"$output_prefix"_"$json_key"_document.bin /downloads/datasets/"$ds_name"/
cp /downloads/tmp/"$output_prefix"_"$json_key"_document.idx /downloads/datasets/"$ds_name"/
