# LLM Pretraining Megatron LM Multinode
Workload for running Megatron based pretraining workloads on multiple nodes using ray

## Helm

To generate manifests and print them in standard output using the default `values.yaml`, run:
```bash
helm template workloads/llm-pretraining-megatron-lm-multinode/helm
```


This will generate a kubernetes manifest with a RayJob, a ConfigMap and a PersistentVolumeClaim resources in the user's active namespace.

To override the default values, a specific file can be passed using `--values` flag
```bash
helm template workloads/llm-pretraining-megatron-lm-multinode/helm --values workloads/llm-pretraining-megatron-lm-multinode/helm/overrides/values-llama-8b-2x2-4ddp.yaml
```

Note:
Anything overlapping with the default `values.yaml` file can be omitted from the specific files passed with the `--values` flag.

## Running

To run the workload, simply pipe the generated manifests to a `kubectl apply` command, like:

```bash
helm template workloads/llm-pretraining-megatron-lm-multinode/helm | kubectl apply -f -
```

## Docker image

We recommend using the `ghcr.io/silogen/megatron-lm-ray` image. Ray dependecies are  included in the `megatron-lm-ray` image.

## Assumptions

Some assumptions for running the pretraining jobs are as follows: The initial model checkpoint and data, both in Megatron format, are located in an S3-compatible storage. Additionally, it is assumed that a secret containing the S3 storage provider's HMAC credentials (access key and secret key) is present in the namespace where the jobs are executed. The defaults (as viewed from the Kubernetes manifest's perspective) are:

```
- name: BUCKET_STORAGE_ACCESS_KEY
    valueFrom:
    secretKeyRef:
        name: minio-credentials
        key: minio-access-key
- name: BUCKET_STORAGE_SECRET_KEY
    valueFrom:
    secretKeyRef:
        name: minio-credentials
        key: minio-secret-key
```

## Cleanup

Note that this chart, when run with `kubectl apply`, will create RayJob, PersistentVolumeClaim and ConfigMap objects. After the RayJob has finished, there is a 3600-second grace period to remove the RayJob object from the namespace. ConfigMap and PersistentVolumeClaim are attached to the lifecycle of the RayJob at the start of the workload and cleaned up automatically. However, if there is an issue during start up of the workload, there can be a situation, when ConfigMap and PersistentVolumeClaim are created but are not owned by the RayJob. In this case ConfigMap and PersistentVolumeClaim resources should be cleaned up manually using `kubectl delete` command.
