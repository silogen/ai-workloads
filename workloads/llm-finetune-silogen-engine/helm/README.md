# Fine-tuning with the SiloGen fine-tuning engine

This is a Helm Chart for fine-tuning Jobs based on the SiloGen fine-tuning engine.
The chart integrates the fine-tuning config as part of the `values.yaml` input.

See the `values.yaml` file for the general structure (more documentation coming soon).

## Running the workload
Since the `helm install` semantics are centered around on-going installs, not jobs that run once,
it's best to just run `helm template` and pipe the result to `kubectl create`.

Example command:
```bash
helm template llama-32-1b-alpha . \
  -f overrides/models/meta-llama_llama-3.2-1b-instruct.yaml \
  -f overrides/data/argilla-mistral-large-human-prompts.yaml \
  --set checkpointsRemote="default-bucket/experiments/debug-llama-32-1b-alpha" \
  | kubectl create -f -
```

## Tensorboard
Specifying `runTensorboard: true` and `finetuning_config.trainings_args.report_to: ["tensorboard"]` logs the training progress to tensorboard and serves the Tensorboard web UI from the training container.
The tensorboard logs are also uploaded to the bucket storage for later use.

To connect to the Tensorboard web UI on the container, start a port-forward:
```bash
kubectl port-forward --namespace YOUR_NAMESPACE pods/YOUR_POD_NAME 6006:6006
```
Then browse to [localhost:6006](localhost:6006).

Note that the logging frequency is set by the Hugging Face Transformers [logging options](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.logging_strategy).

## Best-known-configuration model overrides

The directory `overrides/models` hosts fine-tuning recipes for various models. The files are named according to model canonical names, which is the Hugging Face pattern of `organization/model-name` just changed into `organization_model-name`. These configurations have been shown to work well in experiments, but that does not guarantee that these exact parameters are always optimal. The best parameters still depend on the data, too.

## Running DPO

The default values have an SFT-specific field `finetuning_config.sft_args`, which must be set to null to run DPO. An example command for running DPO training:

```bash
helm template debug-dpo . \
 --values overrides/tiny-llama-dpo-full-param.yaml \
 --values overrides/data/ultrafeedback-binarized.yaml \
 --set checkpointsRemote="default-bucket/experiments/debug-dpo-tiny-llama-ultrafeedback-alpha" \
 | kubectl apply -f - -nkaiwo
```

## AIM Manifest integration

With `aimManifest.enabled: true`, the workload will create and upload an AIMModel manifest to bucket storage, and by default apply it in the cluster. This makes the successfully fine-tuned model available to deploy.

This feature needs a service account with the correct permissions. An account is created by default, but if the user wants to use an existing account instead, it's possible to provide by reference with `aimManifest.serviceAccountName`.
The service account referred to by `serviceAccountName` should have:
```yaml
rules:
- apiGroups: ["aim.eai.amd.com"]
  resources: ["aimmodels"]
  verbs: ["create", "get", "list", "patch", "update"]
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "watch"]
```

## Compatible Accelerator Metadata

The `metadata.compatibleAccelerators` section has no effect on the workloads, but overrides in `overrides/models/` use this to indicate the accelerator that the recipe is compatible with.
