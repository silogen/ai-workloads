# Silogen-Engine Fine-Tuning Llama 3.1 8B Instruct with Open Protein Instructions Dataset

## Introduction

In this tutorial, we demonstrate LoRA fine-tuning of Llama 3.1 8B Instruct with the Open Protein Instructions dataset using the [Silogen fine-tuning engine](https://github.com/silogen/llm-finetuning) end-to-end, from downloading data to querying the fine-tuned model. Upon receiving a query containing a protein sequence as a input, the fine-tuned model attempts to provide an expert response about the protein sequence, such as its functional description. Depending on your fine-tuning configuration, the end result would be similar to [OPI-Llama](https://huggingface.co/BAAI/OPI-Llama-3.1-8B-Instruct) fine-tuned by the dataset authors.

[Base model](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | [Fine-tuning dataset](https://huggingface.co/datasets/BAAI/OPI)

Please note that the dataset is CC-BY-NC-4.0 licensed and used solely for demonstration purposes here with permission from the authors.

## Prerequisites

- MinIO cluster storage (or similar) with credentials configured in your Kubernetes namespace secrets
- Hugging Face token for data and model download in your Kubernetes namespace secrets

## Running workloads

The commands below are assumed to be run at the repository root.

### Data download and preprocessing

We will use the `workloads/download-data-to-bucket/helm/overrides/tutorial-05-opi-data.yaml` override to download the dataset from Hugging Face, convert it to the format expected by the Silogen fine-tuning engine, and persist the processed dataset to `bucketDataDir` configured in our override. Our `dataScript` will also create a sample of 1k rows for quick demonstration of the fine-tuning workflow.

```bash
helm template workloads/download-data-to-bucket/helm \
  -f workloads/download-data-to-bucket/helm/overrides/tutorial-05-opi-data.yaml \
  --name-template "download-opi-data" \
  | kubectl apply -f -
```

### Base model download

We can download the base model to MinIO without customizing the existing override for our base model. Downloading this model requires a Hugging Face token (assumed to be available in the namespace), which we specify in another override.

```bash
helm template workloads/download-huggingface-model-to-bucket/helm \
  -f workloads/download-huggingface-model-to-bucket/helm/overrides/models/meta-llama_llama-3.1-8b-instruct.yaml \
  -f workloads/llm-finetune-silogen-engine/helm/overrides/utilities/hf-token.yaml \
  --name-template "download-llama-31-8-instruct" \
  | kubectl apply -f -
```

### Fine-tuning

To start fine-tuning with the Silogen engine, we can use existing overrides for reasonable default fine-tuning parameters for the base model, and to enable TensorBoard monitoring. We can customize any parameters, such as the number of fine-tuning GPUs, with `workloads/llm-finetune-silogen-engine/helm/overrides/tutorial-05-llama-lora-opi-data.yaml`.

```bash
workloads_path="workloads/llm-finetune-silogen-engine/helm"
helm template $workloads_path \
  -f $workloads_path/overrides/models/meta-llama_llama-3.1-8b-instruct.yaml \
  -f $workloads_path/overrides/utilities/tensorboard.yaml \
  -f $workloads_path/overrides/tutorial-05-llama-lora-opi-data.yaml \
  --name-template llm-finetune-llama-opi \
  | kubectl apply -f -
```

To monitor fine-tuning progress with TensorBoard, we can forward the associated port to access with a local browser, e.g., `kubectl port-forward pods/<pod_name> 6006:6006`. Model checkpoints and logs will persist in the `checkpointsRemote` specified in our custom override file.

### Inference

#### Deploying each model

To deploy the base model using vLLM, we can use the existing override

```bash
name="llama-31-8-instruct"
helm template $name workloads/llm-inference-vllm/helm \
-f workloads/llm-inference-vllm/helm/overrides/models/meta-llama_llama-3.1-8b-instruct.yaml \
--set "vllm_engine_args.served_model_name=$name" \
| kubectl apply -f -
```

To deploy our fine-tuned model, we set the model path to our final experiment checkpoint

```bash
name="llama-31-8B-lora-opi-1k"
helm template workloads/llm-inference-vllm/helm \
  -f workloads/llm-inference-vllm/helm/overrides/models/meta-llama_llama-3.1-8b-instruct.yaml \
  --set "model=s3://default-bucket/experiments/finetuning/$name/checkpoint-final" \
  --set "vllm_engine_args.served_model_name=$name" \
  --name-template "opi-llama" \
  | kubectl apply -f -
```

#### Querying the deployed models

Forward a port for each deployment

```bash
base_model="llama-31-8-instruct"
ft_model="llama-31-8B-lora-opi-1k"
port_1=8011
port_2=8012

kubectl port-forward svc/llm-inference-vllm-$base_model $port_1:80 > /dev/null & portforwardPID=$!

kubectl port-forward svc/llm-inference-vllm-$ft_model $port_2:80 > /dev/null & portforwardPID=$!
```

Query each model to compare their outputs:

```bash
question="Can you provide the functional description of the following protein sequence? Sequence: MRWQEMGYIFYPRKLR"

# Base model
curl http://localhost:$port_1/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'$base_model'",
        "messages": [
            {"role": "user", "content": "'"$question"'"}
        ]
    }' | jq ".choices[0].message.content" --raw-output

# [Example response] Unfortunately, I can't identify the exact function of the given protein sequence. However, ...

# Fine-tuned model
curl http://localhost:$port_2/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'$ft_model'",
        "messages": [
            {"role": "user", "content": "'"$question"'"}
        ]
    }' | jq ".choices[0].message.content" --raw-output

# [Example response] This protein is a ribonucleoprotein involved in the processing of rRNA and the assembly of ribosomes.
```

### Cleaning up

We can delete our model deployments for example with kubectl

```bash
kubectl delete deployments/llm-inference-vllm-<model_name>
kubectl delete svc/llm-inference-vllm-<model_name>
```
