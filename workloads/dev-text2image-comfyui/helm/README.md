# ComfyUI Text-to-Image/Video Workload

This Helm Chart deploys a [ComfyUI](https://github.com/comfyanonymous/ComfyUI) web app for text-to-image/video generation. ComfyUI is a powerful node-based interface for stable diffusion that provides advanced workflows for AI image and video generation.

## Features

- **Pre-configured ComfyUI Environment**: Automatically installs and configures ComfyUI with ROCm support
- **Model Management**: Support for downloading models from Hugging Face or MinIO/S3 storage
- **ComfyUI Manager**: Includes [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) for easy extension management

## Configuration Parameters

You can configure the following parameters in the `values.yaml` file or override them via the command line:

| Parameter                    | Description                                                           | Default                                    |
|------------------------------|-----------------------------------------------------------------------|--------------------------------------------|
| `image`                      | Container image repository and tag                                    | `rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.8.0` |
| `imagePullSecrets`           | List of Kubernetes secrets for pulling images from private registries | `[]`                                       |
| `gpus`                       | Number of GPUs to allocate                                            | `1`                                        |
| `model`                      | Hugging Face model path (e.g., `Comfy-Org/flux1-dev`). Set to `""` to start with no checkpoint | `Comfy-Org/stable-diffusion-v1-5-archive` |
| `tag`                        | Identifies one model's binaries (**\*tag\*safetensors**) to download. Without it the whole repository is downloaded | `v1-5-pruned-emaonly-fp16` |
| `storage.ephemeral.quantity` | Ephemeral storage size                                                | `200Gi`                                    |
| `kaiwo.enabled`              | Enable Kaiwo operator management                                      | `false`                                    |
## Using Private Container Registries

If you need to pull images from a private registry, set the `imagePullSecrets` field in your `values.yaml` or via the command line. This should be a list of Kubernetes secret names that provide credentials for your registry.

Example in `values.yaml`:

```yaml
imagePullSecrets:
  - my-registry-secret
```

Or via the command line:

```bash
helm template . --set imagePullSecrets={my-registry-secret} | kubectl apply -f -
```

The deployment will include these secrets in the pod spec, allowing Kubernetes to authenticate to your private registry.

## Environment Variables

The following environment variables are configured for MinIO/S3 integration:

| Variable                     | Description                                                           | Default                                    |
|------------------------------|-----------------------------------------------------------------------|--------------------------------------------|
| `BUCKET_STORAGE_HOST`        | MinIO/S3 endpoint URL                                                 | `http://minio.minio-tenant-default.svc.cluster.local:80` |
| `BUCKET_STORAGE_ACCESS_KEY`  | MinIO/S3 access key (from secret)                                     | From `minio-credentials` secret            |
| `BUCKET_STORAGE_SECRET_KEY`  | MinIO/S3 secret key (from secret)                                     | From `minio-credentials` secret            |
| `PIP_DEPS`                   | Additional Python packages to install via pip (space or newline separated URLs/packages) | `""`                    |
| `COMFYUI_PATH`               | ComfyUI installation path                                             | `/workload/ComfyUI`                        |
| `MODEL_BIN_URL`              | Direct URL to download an additional model checkpoint (optional)      | Not set                                    |

## Model Configuration

The default deployment pre-loads `v1-5-pruned-emaonly-fp16.safetensors` (2 GiB),
which is the checkpoint ComfyUI's stock workflow selects by name, so the
workspace can generate an image as soon as it opens. The workspace reports
itself ready only once that checkpoint is on disk.

`tag` must identify a single model's file. It is matched as `*tag*safetensors`
both to choose what to download and to decide the workspace is ready, so a
fragment shared by several models (`fp8`, say) would let a checkpoint left by a
previously configured model pass for the current one.

Other ways to get a model:

- **A different checkpoint at deploy time**, by setting `model` and `tag`, or by
  using one of the overrides in `overrides/models/`. Larger models take
  proportionally longer before the workspace becomes ready, and the stock
  workflow will need its checkpoint re-selected.
- **At runtime from the UI**, using the ComfyUI-Manager model manager. Set
  `model: ""` to skip the pre-load entirely, in which case the workspace becomes
  ready as soon as the server answers and starts with an empty checkpoint list.

### Using Hugging Face Models

Configure models from Hugging Face by setting the `model` parameter:

```yaml
# Example: FLUX.1-dev model
model: "Comfy-Org/flux1-dev"
tag: "flux1-dev-fp8"
```

The example above appears in ComfyUI as `flux1-dev-fp8.safetensors`. Note the
tag is the full `flux1-dev-fp8` rather than `fp8`, which `Comfy-Org/flux1-schnell`
also matches.

### Using S3/MinIO Models

For models stored in S3/MinIO, use the s3:// prefix:

```yaml
model: "s3://models/Comfy-Org/flux1-dev"
```

### Using Direct Download URLs

For direct model downloads, use the `MODEL_BIN_URL` environment variable:

```yaml
env_vars:
  MODEL_BIN_URL: "https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/all_in_one/lumina_2.safetensors"
```

### Pre-configured Model Overrides

The workload includes several pre-configured model overrides in the `overrides/models/` directory:

## Deploying the Workload

### Basic Deployment

To deploy the service with default settings, run the following command within the `helm` folder:

```bash
helm template . | kubectl apply -f -
```

### Deployment with Model Override

To deploy with a specific model configuration:

```bash
helm template flux . -f overrides/models/comfy-org_flux1-dev-fp8.yaml | kubectl apply -f -
```

### Custom Deployment

To deploy with custom parameters. Set `tag` whenever you change `model`, so that
it identifies a file the new repository actually contains:

```bash
helm template flux . --set model="Comfy-Org/flux1-dev" --set tag="flux1-dev-fp8" | kubectl apply -f -
```

## Accessing the Workload

### Verify Deployment

Check the deployment and service status:

```bash
kubectl get deployment
kubectl get service
```

### Port Forwarding

To access the service locally on port `8188`, forward the port of the service/deployment:

```bash
kubectl port-forward services/dev-text2image-comfyui 8188:80
```

Then open a web-browser and navigate to [http://localhost:8188](http://localhost:8188) to access ComfyUI.

### Accessing the Workload via URL

To access the workload through a URL, you can enable either an Ingress or HTTPRoute in the `values.yaml` file. The following parameters are available:

| Parameter              | Description                                                                 | Default                                                                 |
|------------------------|-----------------------------------------------------------------------------|-------------------------------------------------------------------------|
| `ingress.enabled`      | Enable Ingress resource                                                     | `false`                                                                 |
| `http_route.enabled`   | Enable HTTPRoute resource                                                   | `false`                                                                 |
| `http_route.parentRefs`| List of gateway parent references (`group`, `name`, `namespace`), takes precedence over gateway namespace keys | `[]` |
| `http_route.gatewayNamespace` | Single-gateway fallback used when `parentRefs` is empty | `envoy-gateway-system` |
| `http_route.gateway_namespace` | Deprecated alias for `gatewayNamespace` | `""` |

Example dual-gateway configuration:

```yaml
http_route:
  enabled: true
  parentRefs:
    - group: gateway.networking.k8s.io
      name: https
      namespace: envoy-gateway-system
    - group: gateway.networking.k8s.io
      name: https
      namespace: kgateway-system
```

See the corresponding template files in the `templates/` directory. For more details on configuring Ingress or HTTPRoute, refer to the [Ingress documentation](https://kubernetes.io/docs/concepts/services-networking/ingress/) and [HTTPRoute documentation](https://kubernetes-sigs.github.io/gateway-api/v0.5.0/httproute/), or documentation of the particular gateway implementation you may use, like [KGateway](https://kgateway.dev/). Check with your cluster administrator for the correct configuration for your environment.

## Health Checks and Monitoring

The workload includes comprehensive health monitoring:

- **Startup Probe**: Allows up to 10 minutes for ComfyUI to start (checks `/queue` endpoint)
- **Liveness Probe**: Monitors if ComfyUI is running properly
- **Readiness Probe**: Asks ComfyUI which checkpoints it can see (`/models/checkpoints`). When `model` is set, it requires that model's checkpoint, so the workload only receives traffic once it is usable. With no `model`, the server answering is enough

A configured checkpoint is downloaded in the background so that a large model
cannot delay the server bind past the startup probe budget. ComfyUI therefore
serves `/queue` before the model is on disk, and the pod only becomes Ready once
the checkpoint appears. A download that fails or stalls restarts the container,
which resumes the transfer, rather than leaving a modelless workload running.
