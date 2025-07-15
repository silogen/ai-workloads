# OpenAI-compatible Endpoint Benchmarking

This Helm chart defines a batch job to benchmark LLM performance using vLLM's benchmarking script against OpenAI-compatible API endpoints. It follows the [best practices](https://rocm.blogs.amd.com/artificial-intelligence/LLM_Inference/README.html) for optimized inference on AMD Instinct GPUs.

## Prerequisites and Configuration

1. **Helm**: Ensure `helm` is installed. Refer to the [Helm documentation](https://helm.sh/) for installation instructions.

2. **MinIO Storage**: Required for saving benchmark results. Configure the following environment variables in `values.yaml`:
    - `BUCKET_STORAGE_HOST`
    - `BUCKET_STORAGE_ACCESS_KEY`
    - `BUCKET_STORAGE_SECRET_KEY`
    - `BUCKET_RESULT_PATH`

3. **API Endpoint**: An OpenAI-compatible API endpoint is required. Configure this in `values.yaml` as `env_vars.OPENAI_API_BASE_URL` or override using the `--set` option with Helm.

4. **Tokenizer**: Required for token calculations. Specify a HuggingFace model repository in `values.yaml` by setting `env_vars.TOKENIZER`. The default is `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`.

5. **HuggingFace Token** (optional): Set the `env_vars.HF_TOKEN` environment variable if using gated tokenizers (e.g., Mistral and Llama models) from HuggingFace.

## Benchmark Configuration

The benchmark behavior can be customized using the following environment variables:

- **`INPUT_LENGTH`** (default: `2048`): Sets the input token length for benchmark requests
- **`OUTPUT_LENGTH`** (default: `2048`): Sets the output token length for benchmark requests
- **`QPS`** (default: `inf`): Sets the queries per second rate. Can be:
  - A single value: `"10"`, `"inf"` (unlimited)
  - Multiple space-separated values: `"1 5 10 inf"` (runs tests at each rate)

The benchmark automatically tests with request concurrency levels of 1, 2, 4, 8, 16, 32, 64, 128, and 256.

## Deployment Example

To deploy the chart, run the following command in the `helm/` directory:

```bash
helm template . \
    --set env_vars.OPENAI_API_BASE_URL="http://example-open-ai-api-server.com/v1/" \
    --set env_vars.TOKENIZER="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    --set env_vars.INPUT_LENGTH="1024" \
    --set env_vars.OUTPUT_LENGTH="512" \
    --set env_vars.QPS="1 5 10 inf" | \
    kubectl apply -f -
```
