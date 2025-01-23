# Workloads

This directory contains all the workloads. Each workload has its own directory, with different formats defined in subdirectories.

To create a new workload, you can duplicate an existing workload and adapt as needed.

The files needed to create different instantiations of a workload depends on the format. For reference, see the existing workloads, and the documentation of the respective formats.

## Kueue

If you are using Kueue to manage quotas and how jobs consume them, you need to add the necessary metadata labels or annotations to your manifest. For example, in the case of a Kubernetes Job, based on the [Kueue documentation](https://kueue.sigs.k8s.io/docs/tasks/run/jobs/), you would have:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
    labels:
        kueue.x-k8s.io/queue-name: your-queue-name
    name: your-workload-name
    namespace: your-namespace
spec:
    suspend: true
    ...
```