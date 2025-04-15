# JupyterLab workload

This workload deploys a basic JupyterLab instance on top of any image that has Python (pip) built-in, ideal for interactive development sessions and experimentation with other workloads.

## Configuration Parameters

The following parameters can be configured in the `values.yaml` file or overridden via command line:

| Parameter | Description                                      | Default       |
|-----------|--------------------------------------------------|---------------|
| `image`   | Container image repository and tag               | `python:3.12` |
| `gpus`    | Number of GPUs to allocate (0 for CPU-only mode) | `0`           |

## Deploying the Workload

To deploy the chart with a release name `example`, run the following command from the `helm/` directory:

```bash
helm template example . | kubectl apply -f -
```

**Note**: When setting `gpus` value greater than 0, it is important to also specify a GPU-capable image to properly utilize the allocated resources.

## Port Forwarding

To access JupyterLab locally, forward the service port to your local machine:

```bash
kubectl port-forward services/dev-jupyterlab-pipx-example 8888:80
```

Then access JupyterLab by navigating to `http://localhost:8888` in your web browser.
