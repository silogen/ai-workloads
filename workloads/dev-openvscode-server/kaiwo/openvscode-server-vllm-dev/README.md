# Open VS Code Server

> [! NOTE]
> This guide demonstrates how to deploy a VS Code workload with [Kaiwo](https://github.com/silogen/kaiwo), a Kubernetes-native AI Workload Orchestrator designed to accelerate GPU workloads, and has been tested to work with Kaiwo v0.0.5.

## Prerequisites

Ensure the following prerequisites are met before deploying the workload:

1. **Kaiwo CLI**: Install the Kaiwo CLI tool. Refer to the [Kaiwo Installation Guide](https://github.com/silogen/kaiwo) for instructions.
2. **Secrets**: Secrets for accessing the bucket storage must exist in the namespace:
    - `minio-credentials` with keys `minio-access-key` and `minio-secret-key`.
    - `hf-token` with key `hf-token`

## Deploying the Workload

Follow these steps to deploy workload using Kaiwo:

1. **Deploy with Kaiwo**:

```bash
kaiwo serve -i ghcr.io/silogen/openvscode-server:vllm-dev-20250124 -g 1 -p kaiwo/openvscode-server-vllm-dev/ --name openvscode-server-vllm-dev
```

2. **Port Forwarding**: Forward the port to access the service (assuming the deployment is named `ubuntu-kaiwo`):

``bash
kubectl port-forward deployments/ubuntu-kaiwo 3000:3000 -n kaiwo
```
