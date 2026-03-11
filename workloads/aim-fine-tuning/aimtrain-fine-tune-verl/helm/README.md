<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->


# VeRL Finetuning

This is a Helm Chart for running a fine-tuning job using [VeRL](https://github.com/volcengine/verl)

There are recipes in the `overrides/` directory, organized by fine-tuning method, and then by model. When using these recipes, be sure to set appropriate values for `checkpointsRemote` and `datasetRemote`. The output is saved with in the bucket storage path specified by `checkpointsRemote`.

Bucket storage configuration is set with the `bucketStorageHost` and `bucketCredentialsSecret` values.

See the `values.yaml` file for annotations of all configurable values.


## VeRL Configuration

Include any parameters for VeRL in the `verlConfig` parameter. This mapping is turned into a YAML for VeRL in the workload. You can check the [VeRL Documentation](https://verl.readthedocs.io/en/latest/examples/config.html) to understand the different variables that are available.


## Running the workload

From the `aim-fine-tuning` directory, run `helm template` and pipe the result to `kubectl create`.

Example command:
```bash
helm template full-2gpu-alpha aimtrain-fine-tune-verl/helm \
  --values aimtrain-fine-tune-verl/helm/overrides/sft/meta-llama_llama-3-1-8b/full-2gpu.yaml \
  | kubectl create -f -
```


## Data specification

VeRL requires that the data is prepared for the policy training in a [particular way](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html).

To use your own datasets from MinIO, specify the path as `datasetRemote`. It should point to a directory with files that have already been appropriately processed (`train.parquet` and `test.parquet`).


## Model specification

To use a base model from HuggingFace or other source directly supported by VeRL, specify the model name in `modelName`.

Alternatively to use a model from MinIO, specify the path to the model in `modelRemote`.

Either `modelName` or `modelRemote` must be specified. If both are included, the model from `modelRemote` is used.

## Cleanup

After the jobs are completed, delete the resources you created by running the same command but piping to `kubectl delete` instead:

```bash
helm template full-2gpu-alpha aimtrain-fine-tune-verl/helm \
  --values aimtrain-fine-tune-verl/helm/overrides/sft/meta-llama_llama-3-1-8b/full-2gpu.yaml \
  | kubectl delete -f -
```


## Deploying the VeRL finetuned model as an AIM

The training produces a set of trained weights in bucket storage. Beside these weights, a Kubernetes Manifest for an AIMModel is also produced. This should be applied to the inference cluster to make the finetuned model available as an AIM. This manifest looks something like:

```yaml
apiVersion: aim.eai.amd.com/v1alpha1
kind: AIMModel
metadata:
  name: llama-31-8b-fullft
  namespace: finetuning
spec:
  image: amdenterpriseai/aim-base:0.9
  modelSources:
    - modelId: finetuned/llama-31-8b-fullft
      sourceUri: s3://default-bucket/experiments/llama31-8b-mi300-fullft/checkpoint-final
  customTemplates:
    - hardware:
        gpu:
          requests: 1
  env:
    - name: AWS_ACCESS_KEY_ID
      valueFrom:
        secretKeyRef: { name: minio-credentials, key: minio-access-key }
    - name: AWS_SECRET_ACCESS_KEY
      valueFrom:
        secretKeyRef: { name: minio-credentials, key: minio-secret-key }
    - name: AWS_ENDPOINT_URL
      value: http://minio.minio-tenant-default.svc.cluster.local:80
```

Note that this manifest does not deploy the model, for that, you can use a trivial AIMService that reference the AIMModel by `spec.model.name`, for instance with the above AIMModel, this manifest could be used:
```yaml
apiVersion: aim.eai.amd.com/v1alpha1
kind: AIMService
metadata:
  name: custom-llama-31-8b-fullft
  namespace: finetuning
spec:
  model:
    name: llama-31-8b-fullft
  template:
    allowUnoptimized: true
```
Here, the `template.allowUnoptimized: true` is needed, because the default template that the AIMModel specifies inline (above) is unoptimized. For custom AIMModels with optimized templates, this is not needed.


## Using MLFlow for experiment tracking

If you have an MLFlow instance running, it can be used for experiment tracking, by creating an override file as follows. Be sure to set the MLFlow URI to yours, here is an example of an MLFlow instance running in the cluster:
```bash
echo << EOF > mlflow-enable.yaml
mlflowTrackingUri: "http://mw-dev-tracking-mlflow-1769418573-c1b7"
verlConfig:
  trainer:
    logger:
      - "console"
      - "mlflow"
EOF
```

Then, just add `--values mlflow-enable.yaml` to the helm chart template command.
