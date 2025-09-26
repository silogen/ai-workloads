# Helm Template for Packaging Models for TorchServe

This helm template archives model files. The workload interacts with MiniO storage to manage model artifacts. Specifically, it writes compressed model files to the remote storage.

## Usage

To deploy the workload, use the following command:

```bash
helm template workloads/torchserve-model-packager/helm \
    --name-template job \
    | kubectl apply -f -
```

You can pass overrides to the default values as following:

```bash
helm template workloads/torchserve-model-packager/helm \
    -f workloads/torchserve-model-packager/helm/overrides/Wan2.1-VACE-1.3B-diffusers.yaml
    --name-template VACE \
    | kubectl appply -f -
```

### Note

- The `helm install` command is designed for ongoing installations, not one-time jobs. Therefore, it is recommended to use `helm template` and pipe the output to `kubectl create`. This approach is more suitable for jobs that do not require modifying existing entities.
- Use `kubectl create` instead of `kubectl apply` for this job, as it is intended to create new resources without updating existing ones.

## Configuration

Refer to the `values.yaml` file for configurable user input values. The file includes instructions to help you customize the workload as needed.
