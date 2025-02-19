# LLM Inference with vLLM

This Helm Chart deploys the LLM Inference vLLM workload.

## Prerequisites

Ensure the following prerequisites are met before deploying any workloads:

1. **Helm**: Install `helm`. Refer to the [Helm documentation](https://helm.sh/) for instructions.
2. **Secrets**: Create the following secrets in the namespace:
    - `minio-credentials` with keys `minio-access-key` and `minio-secret-key`.
    - `hf-token` with key `hf-token`.

## Deploying the Workload

It is recommended to use `helm template` and pipe the result to `kubectl create`, rather than using `helm install`.

### Alternative 1: Deploy a Specific Model Configuration

To deploy a specific model along with its settings, use the following command from the `helm` directory:

```bash
helm template . -f overrides/models/tinyllama_tinyllama-1.1b-chat-v1.0.yaml | kubectl apply -f -
```

### Alternative 2: Override the Model

You can also override the model on the command line:

```bash
helm template . --set model=Qwen/Qwen2-0.5B-Instruct | kubectl apply -f -
```

### Alternative 3: Deploy a Model from Bucket Storage

If you have downloaded your model to bucket storage, use:

```bash
helm template . --set model=s3://models/Qwen/Qwen2-0.5B-Instruct | kubectl apply -f -
```

The model will be automatically downloaded before starting the inference server.

## User Input Values

Refer to the `values.yaml` file for the user input values you can provide, along with instructions.
