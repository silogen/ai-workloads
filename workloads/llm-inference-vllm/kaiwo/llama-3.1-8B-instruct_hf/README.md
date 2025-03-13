# LLM Inference with vLLM

> [!NOTE]
> This guide demonstrates how to deploy the LLM Inference vLLM workload using [Kaiwo](https://github.com/silogen/kaiwo), a Kubernetes-native AI Workload Orchestrator designed to accelerate GPU workloads, and has been tested to work with Kaiwo v0.0.5.

## Prerequisites

Ensure the following prerequisites are met before deploying the workload:

1. **Kaiwo CLI**: Install the Kaiwo CLI tool. Refer to the [Kaiwo Installation Guide](https://github.com/silogen/kaiwo) for instructions.
2. **Secret**: A secret named `hf-token` must exist in the namespace.

## Deploying the Workload

Follow these steps to deploy the LLM Inference vLLM workload using Kaiwo:

1. **Deploy with Kaiwo**:
    ```bash
    kaiwo serve --image rocm/vllm-dev:20241205-tuned --path kaiwo/llama-3.1-8B-instruct_hf --gpus 1
    ```

2. **Verify Deployment**: Check the deployment status:
    ```bash
    kubectl get deployment
    ```

3. **Port Forwarding**: Forward the port to access the service (assuming the deployment is named `ubuntu-kaiwo`):
    ```bash
    kubectl port-forward deployments/ubuntu-kaiwo 8080:8080
    ```

4. **Test the Deployment**: Send a test request to verify the service:
    ```bash
    curl http://localhost:8080/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who won the world series in 2020?"}
            ]
        }'
    ```
