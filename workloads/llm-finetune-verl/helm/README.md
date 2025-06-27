# Finetuning with VeRL

This is a Helm Chart for running a finetuning job using [VeRL](https://github.com/volcengine/verl)

The output is saved with MinIO in the directory specified by `checkpointsRemote`.

## Configuration

Include any parameters for VeRL in the `verlConfig` parameter. See the override file [`overrides/ppo_qwen_gsm8k.yaml`](overrides/ppo_qwen_gsm8k.yaml) for an example and the [VeRL documentation](https://verl.readthedocs.io/en/latest/examples/config.html) for more details.

## Running the workload

The simplest is to run `helm template` and pipe the result to `kubectl create`.

Example command using the example override file `overrides/ppo_qwen_gsm8k.yaml`:

```bash
helm template workloads/llm-finetune-verl/helm \
  --values workloads/llm-finetune-verl/helm/overrides/ppo_qwen_gsm8k.yaml \
  --name-template ppo-qwen-gsm8k-verl \
  | kubectl create -f -
```

## Data specification

VeRL requires that the data is prepared for the policy training in a [particular way](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html).

Some example data preprocess scripts are provided, to use one of these, specify the name of data set used for training as `dataset`. Available datasets are "full_hh_rlhf", "geo3k", "gsm8k", "hellaswag", "math_dataset".

To use your own datasets from MinIO, specify the path as `datasetRemote`. It should point to a directory with files that have already been appropriately processed (`train.parquet` and `test.parquet`).

## Model specification

To use a base model from HuggingFace or other source directly supported by LLaMA-Factory, specify the model name in `modelName`.

Alternatively to use a model from MinIO, specify the path to the model in `modelRemote`.

Either `modelName` or `modelRemote` must be specified. If both are included, the model from `modelRemote` is used.

## Cleanup

After the jobs are completed, please delete the resources created. To delete the resources, you can run the same `helm template` command, only replacing `kubectl create` with `kubectl delete`, e.g.:

```bash
helm template workloads/llm-finetune-verl/helm \
  --values workloads/llm-finetune-verl/helm/overrides/ppo_qwen_gsm8k.yaml \
  --name-template ppo-qwen-gsm8k-verl \
  | kubectl delete -f -
```
