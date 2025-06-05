# ComfyUI Text-to-Image/Video Workload

This Helm Chart deploys a [ComfyUI](https://github.com/comfyanonymous/ComfyUI) web app for text-to-image/video generation. ComfyUI is a powerful node-based interface for stable diffusion that provides advanced workflows for AI image and video generation.

## Features

- **Pre-configured ComfyUI Environment**: Automatically installs and configures ComfyUI with ROCm support
- **Model Management**: Support for downloading models from HuggingFace or MinIO/S3 storage
- **ComfyUI Manager**: Includes [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) for easy extension management

## Configuration Parameters

You can configure the following parameters in the `values.yaml` file or override them via the command line:

| Parameter                    | Description                                                           | Default                                    |
|------------------------------|-----------------------------------------------------------------------|--------------------------------------------|
| `image`                      | Container image repository and tag                                    | `rocm/dev-ubuntu-22.04:6.2.4`             |
| `gpus`                       | Number of GPUs to allocate                                            | `1`                                        |
| `model`                     | HuggingFace model path (e.g., `Comfy-Org/flux1-dev`)                | Not set                                    |
| `tag`                       | Specific model binaries (**\*tag\*.safetensors**)  to download (optional)                        | Clone the repo when not set                                   |
| `storage.ephemeral.quantity` | Ephemeral storage size                                               | `200Gi`                                    |
| `kaiwo.enabled`             | Enable Kaiwo operator management                                      | `false`                                    |

## Environment Variables

The following environment variables are configured for MinIO/S3 integration:

| Variable                     | Description                                                           | Default                                    |
|------------------------------|-----------------------------------------------------------------------|--------------------------------------------|
| `BUCKET_STORAGE_HOST`        | MinIO/S3 endpoint URL                                                | `http://minio.minio-tenant-default.svc.cluster.local:80` |
| `BUCKET_STORAGE_ACCESS_KEY`  | MinIO/S3 access key (from secret)                                   | From `minio-credentials` secret            |
| `BUCKET_STORAGE_SECRET_KEY`  | MinIO/S3 secret key (from secret)                                   | From `minio-credentials` secret            |
| `PIP_DEPS`                  | Additional Python packages to install via pip (space or newline separated URLs/packages) | ROCm-compatible torchaudio wheel          |
| `COMFYUI_PATH`              | ComfyUI installation path                                            | `/workload/ComfyUI`                       |
| `MODEL_BIN_URL`             | Direct URL to download an additional model checkpoint (optional)                       | Not set                                    |

## Model Configuration

### Using HuggingFace Models

Configure models from HuggingFace by setting the `model` parameter:

```yaml
# Example: FLUX.1-dev model
model: "Comfy-Org/flux1-dev"
tag: "flux1-dev-fp8"
```

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

To deploy with custom parameters:

```bash
helm template flux . --set model="Comfy-Org/flux1-dev" | kubectl apply -f -
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
| `httproute.enabled`    | Enable HTTPRoute resource                                                   | `false`                                                                 |

See the corresponding template files in the `templates/` directory. For more details on configuring Ingress or HTTPRoute, refer to the [Ingress documentation](https://kubernetes.io/docs/concepts/services-networking/ingress/) and [HTTPRoute documentation](https://kubernetes-sigs.github.io/gateway-api/v0.5.0/httproute/), or documentation of the particular gateway implementation you may use, like [KGateway](https://kgateway.dev/). Check with your cluster administrator for the correct configuration for your environment.

## Health Checks and Monitoring

The workload includes comprehensive health monitoring:

- **Startup Probe**: Allows up to 10 minutes for ComfyUI to start (checks `/queue` endpoint)
- **Liveness Probe**: Monitors if ComfyUI is running properly
- **Readiness Probe**: Ensures ComfyUI is ready to serve requests
