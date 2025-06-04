# LLM Inference Service with Llama.cpp

This Helm chart deploys a Large Language Model (LLM) inference service using [llama.cpp](https://github.com/ggml-org/llama.cpp). The chart clones the llama.cpp source code from GitHub and compiles it into optimized binaries based on the target GPU architecture. Upon deployment, the service downloads pre-trained GGUF models from Hugging Face and exposes an OpenAI-compatible API via an HTTP endpoint

## Prerequisites

**Helm**: Install `helm`. Refer to the [Helm documentation](https://helm.sh/) for instructions.

## Deploying the Workload

Basic configurations are defined in the `values.yaml` file.

The default model is 1.73-bit quantized [DeepSeek-R1-UD-IQ1_M](https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main/DeepSeek-R1-UD-IQ1_M), which fits in one MI300X GPU (192GB VRAM) and can serve with a context length of 4K.

For example: run the following command within the `helm/` folder to deploy the service:

```bash
helm template . --set env_vars.TEMP="0.8" | kubectl apply -f -
```

**Note**: Compiling llama.cpp executables and downloading/merging the GGUF files of DeepSeek R1 (~200GB) from HuggingFace can take a significant amount of time. The deployment process may take over 30 minutes before the LLM inference service is ready.

## Interacting with the Deployed Model

### Verify Deployment

Check the deployment and service status:

```bash
kubectl get deployment
kubectl get service
```

### Port Forwarding

To access the service locally, forward the port using the following commands. This assumes the service name is `llm-inference-llamacpp`:

```bash
kubectl port-forward services/llm-inference-llamacpp 8080:80
```

You can access the Llama.cpp server's WebUI at `http://localhost:8080` using a web browser.

Additionally, an OpenAI-compatible API endpoint is available at `http://localhost:8080/v1`
