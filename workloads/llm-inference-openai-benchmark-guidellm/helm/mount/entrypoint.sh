apt update && apt install -y jq
OPENAI_API_BASE_URL=${OPENAI_API_BASE_URL%/}
MODEL=$(curl -s ${OPENAI_API_BASE_URL}/models | jq -r '.data[0].id')
pip install guidellm

echo -e "Starting GuideLLM benchmarking.\n==========================>"
guidellm \
    --target $OPENAI_API_BASE_URL \
    --model $MODEL \
    --tokenizer $TOKENIZER \
    --data-type emulated \
    --data "prompt_tokens=512,generated_tokens=128" ||
    echo "GuideLLM exit!"
