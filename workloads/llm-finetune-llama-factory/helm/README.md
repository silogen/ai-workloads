# Finetuning with LLaMA-Factory

This is a Helm Chart for running a finetuning job using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

Currently the base model and input data are assumed to be from HuggingFace, or some other source directly supported by LLaMA-Factory.
The output is saved with MinIO in the directory specified by `checkpointsRemote`.

## Configuration

Include any parameters for LLaMA-Factory in the `llamaFactoryConfig` parameter. See the override file [`overrides/finetune-lora.yaml`](overrides/finetune-lora.yaml) for an example and the [LLaMA-Factory documentation](https://llamafactory.readthedocs.io/en/latest/index.html) for more details.

## Running the workload

The simplest is to run `helm template` and pipe the result to `kubectl create`.

Example command using the example override file `overrides/finetune-lora.yaml`:

```bash
helm template workloads/llm-finetune-llama-factory/helm \
  --values workloads/llm-finetune-llama-factory/helm/overrides/finetune-lora.yaml \
  --name-template finetune-lora-llama-factory \
  | kubectl create -f -
```

## Cleanup

After the jobs are completed, please delete the resources created. In particular for multi-node ray jobs, a `PersistentVolumeClaim` is used as shared storage and persists on the cluster after the job is completed.

To delete the resources, you can run the same `helm template` command, only replacing `kubectl create` with `kubectl delete`, e.g.:

```bash
helm template workloads/llm-finetune-llama-factory/helm \
  --values workloads/llm-finetune-llama-factory/helm/overrides/finetune-lora.yaml \
  --name-template finetune-lora-llama-factory \
  | kubectl delete -f -
```

## Multi-node finetuning with ray

The chart supports multi-node jobs by setting `nodes` to an integer greater than 1. Doing so enables ray and creates a RayJob instead. An example config is provided in [`overrides/finetune-lora-ray.yaml`](overrides/finetune-lora-ray.yaml)

When configuring ray jobs, the resources you are requesting (`nodes` and `gpusPerNode`) are automatically specified for LLaMA-Factory, and do not need to be included separately in the `llamaFactoryConfig`.

## Limitations

`unsloth` and `bitsandbytes` are not installed in the currently used image, so any functionality using those libraries will not work.
