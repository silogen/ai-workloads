<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# AIM Fine-tuning

AIM Fine-tuning offers tools and starting points to make fine-tuning easy to use, fast, and guaranteed to work in the context of [AIMs](https://enterprise-ai.docs.amd.com/en/latest/aims/overview.html).

Fine-tuning is the act of adapting or adjusting an already trained, capable AI model to suit a new purpose or learn a new ability.
It is an inherently experiment-based, iterative process involving three key elements:
- The model, which should have the capacity and base capabilities necessary to learn what we intend it to learn.
- The data, which should contain clear examples of the kind of behaviour that we want to teach the model.
- The learning process parameters, the variables (hyperparameters) that control how the model learns from the data.
Fine-tuning involves finding suitable choices for each of these three elements.

At the core of the fine-tuning solution is an optimized training image, wrapping the [VeRL](https://verl.readthedocs.io/en/latest/) fine-tuning framework. AIM Fine-tuning uses the official [ROCm/VeRL](https://hub.docker.com/r/rocm/verl) image. The [VeRL compatibility documentation](https://rocm.docs.amd.com/en/latest/compatibility/ml-compatibility/verl-compatibility.html) lists the features that different versions of the image offer.

The AIM fine-tuning solution is packaged as [helm charts](https://helm.sh/), a templating software for Kubernetes manifests. There are two charts:
- `aimtrain-dataprep-verl/helm` handles data preparation. The data needs to be formatted in a specific way and saved in a binary format for VeRL to read. This is a preliminary step that should be run first.
- `aimtrain-finetune-verl/helm`, handles the fine-tuning itself.


### Supervised Fine-tuning (SFT)
Supervised Fine-tuning (SFT) is a robust, straight-forward fine-tuning method, typically the first go-to approach for any task. Fine-tuning can be done with full parameter updates, which is computationally heavier, but allows the full capacity of the model and the training to be used. A computationally lighter alternative is Low-Rank Adapter (LoRA) training, which can with the right training hyperparameters lead to very good results. LoRA training produces a neural adapter, but it is by default merged back to produce a regular type of model.


## Minimum requirements
- Access to a Kubernetes cluster with an AMD GPU node and AMD GPU operator installed.
- Access to an S3-compatible bucket storage
- Secret in the cluster that holds the bucket storage credentials
- Helm >3.18

## Minimal test
Here's a minimal set of commands with some variables. Run these to e.g. verify your setup.

Edit the credential YAML in the heredoc to values that apply to your setup. The defaults here should work for a typical AMD EAI Cluster:
```bash
echo << EOF > bucket-storage-credentials.yaml
bucketStorageHost: http://minio.minio-tenant-default.svc.cluster.local:80  # URL for your bucket storage
bucketCredentialsSecret:
  name: minio-credentials  # Name of the secret that holds your bucket storage credentials
  accessKeyKey: minio-access-key  # Key in the secret that holds the access key
  secretKeyKey: minio-secret-key  # Key in the secret that holds the secret key
EOF
```

Set variables that determine where data and models are stored, and set the namespace (maps to project name in the AIRM managed clusters):
```bash
datasetRemotePath="default-bucket/aim-fine-tuning/preflight-check/data-split"
modelCheckpointPath="default-bucket/aim-fine-tuning/preflight-check/model-out"
namespace=my-project
runname=preflight-check
```

Now we're ready to run. First, data preparation:
```bash
helm template $runname aimtrain-dataprep-verl/helm \
    --values aimtrain-dataprep-verl/helm/overrides/sft/dolci-instruct-sft.yaml \
    --values bucket-storage-credentials.yaml \
    --set outputRemotePath="$datasetRemotePath" \
    | kubectl create -f - -n $namespace
```

Monitor the progress. The data preparation stage takes about a minute.
```bash
kubectl logs -f -n $namespace job/aimtrain-dataprep-verl-$runname-job
```

Once done, proceed to fine-tuning:
```bash
helm template $runname aimtrain-fine-tune-verl/helm \
    --values aimtrain-fine-tune-verl/helm/overrides/sft/meta-llama_llama-3-1-8b/full-2gpu.yaml \
    --values bucket-storage-credentials.yaml \
    --set datasetRemote="$datasetRemotePath" \
    --set checkpointsRemote="$modelCheckpointPath" \
    --set verlConfig.data.train_max_samples=4096 \
    | kubectl create -f - -n $namespace
```

Fine-tuning can be monitored with:
```bash
kubectl logs -f -n $namespace job/aimtrain-fine-tune-verl-$runname-job
```
