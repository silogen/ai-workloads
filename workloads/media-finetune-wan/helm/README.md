# Fine-Tuning Wan 2.2

## Introduction
This Helm Chart is used to deploy the media-finetune-wan workload, i.e. a Job for customizing Wan 2.2 models, by performing either full-parameter of LoRA fine-tuning, using DiffSynth framework. There are two options to run the workload:

- Interactive (e.g. for testing)
- Automated

The interactive option will simply create a container waiting for manual input, with a setup ready to test or tweak the fine-tuning workflows.
The automated workflows will run the following steps automatically and finish the job:

1. Download models and data
2. Setup and install the training environment
3. Fine-tune
4. Upload checkpoints

## Prerequisites
This workload requires the Wan2.2 model (either 5B or 14B) and a suitable dataset available in a MinIO cluster (or equivalent) bucket storage. We provide workloads to download and prepare the model and an example dataset.

### Model
We can use the workload `workloads/download-huggingface-model-to-bucket/helm` to download the Wan2.2 models (either the 5B or the 14B parameter version). This workload downloads the chosen Wan2.2 model and saves it in MinIO.

#### Wan2.2 5B
```
helm template workloads/download-huggingface-model-to-bucket/helm \
  -f workloads/download-huggingface-model-to-bucket/helm/overrides/media-finetune-wan2-2-TI2V-5B.yaml \
  --name-template download-wan2-2-ti2v-5b \
  | kubectl apply -f -
```

#### Wan2.2 14B
```
helm template workloads/download-huggingface-model-to-bucket/helm \
  -f workloads/download-huggingface-model-to-bucket/helm/overrides/media-finetune-wan2-2-T2V-A14B.yaml \
  --name-template download-wan2-2-t2b-a14b \
  | kubectl apply -f -
```

### Dataset
The workload `workloads/download-data-to-bucket/helm/overrides/media-finetune-wan-disney-dataset.yaml` prepares an example dataset to get started with fine-tuning. This workload downloads the [Steamboat Willy dataset](https://huggingface.co/datasets/Wild-Heart/Disney-VideoGeneration-Dataset), preprocesses it as required by the fine-tuning workload, and saves the dataset in MinIO.

```
helm template workloads/download-data-to-bucket/helm \
  -f workloads/download-data-to-bucket/helm/overrides/media-finetune-wan-disney-dataset.yaml \
  --name-template download-disney-dataset \
  | kubectl apply -f -
```

## Fine-tuning
### Interactive
To run the interactive workload:

```sh
helm template ./helm --name-template wan-finetune | kubectl apply -f -
```

### Automated
To run one of the automated workloads (overrides):

```sh
helm template ./helm -f ./helm/overrides/5B_lora.yaml --name-template wan-finetune | kubectl apply -f -
```

You can choose the override from any of the provided:
- `5B_full.yaml` - (5B full parameter finetuning)
- `5B_lora.yaml` - (5B LoRA finetuning)
- `14B_full_highnoise.yaml` & `14B_full_lownoise.yaml` - (14B full parameter finetuning)
- `14B_lora_highnoise.yaml` & `14B_lora_lownoise.yaml` - (14B LoRA finetuning)
