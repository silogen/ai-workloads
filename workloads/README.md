# Workloads

This directory contains all the workloads. Each workload has its own directory, with different formats defined in subdirectories.

To create a new workload, you can either copy the template directory or duplicate an existing workload.

The definition of different instantiations of a workload depends on the format. For example:

- **Helm**: Uses `values.yaml` and its overrides.
- **Kaiwo**: Uses `config.yaml` and `env` files along with their copies.
- **K8s**: Uses `deployment.yaml`, `service.yaml`, and other Kubernetes resource files.

## Submitting to Kueue

To submit a workload to Kueue, label the workload's manifest file with the necessary specifications. For example, in the case of a Kubernetes Job:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
    labels:
        kueue.x-k8s.io/queue-name: your-queue-name
    name: your-workload-name
    namespace: your-namespace
spec:
    ...
```