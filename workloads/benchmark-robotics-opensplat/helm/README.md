# OpenSplat by Silogen using ROCm

https://github.com/pierotofy/OpenSplat

A free and open source implementation of 3D Gaussian splatting written in C++, focused on being portable, lean and fast.

OpenSplat takes camera poses + sparse points in COLMAP, OpenSfM, ODM, OpenMVG or nerfstudio project format and computes a scene file (.ply or .splat) that can be later imported for viewing, editing and rendering in other software.

This Helm Chart deploys the OpenSplat workload for AMD GPU's.

## Prerequisites

Ensure the following prerequisites are met before deploying any workloads:

1. **Helm**: Install `helm`. Refer to the [Helm documentation](https://helm.sh/) for instructions.
2. **Kubectl** Install [kubectl](https://kubernetes.io/docs/tasks/tools/#kubectl).
3. Get access to a kubernetes cluster with AMD GPU nodes with ROCm support.

## Running the workload

It is recommended to use `helm template` and pipe the result to `kubectl apply` , rather than using `helm install`. Generally, a command looks as follows

```bash
helm template [optional-release-name] <helm-dir> -f <overrides/xyz.yaml> --set <name>=<value> | kubectl apply -f -
```

For example, to launch the workload in your default namespace use the following command:

```bash
helm template testrun workloads/benchmark-robotics-opensplat/helm | kubectl apply -f -
```

## User input values

Refer to the `values.yaml` file for the user input values you can provide, along with instructions.

## Benchmarking results

To see the output of the benchmark run, e.g.:

```bash
kubectl logs -f benchmark-robotics-opensplat-testrun-f7vfv -c benchmark-robotics-opensplat
```
The summary table will be printed to the standard output in the end of the run. Note that your specific pod name used in the above command will be different. You can check it by reading the output of `kubectl get pods` command.
