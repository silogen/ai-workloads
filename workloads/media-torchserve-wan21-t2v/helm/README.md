# 🧠 TorchServe for Wan2.1 — Helm Deployment

This Helm chart deploys a custom TorchServe instance serving the Wan2.1 text-to-video generation model. It handles model setup, packaging, and serving in a fully containerized GPU environment.

## 🔧 Project Structure

```
helm/
├── Chart.yaml # Helm chart metadata
├── values.yaml # Main config values (e.g. image, ports, GPU)
├── templates/ # Helm templates for deployment, config, etc.
├── mount/ # Files mounted into the container via ConfigMap
│ ├── config.properties
│ ├── download_model.py
│ ├── entrypoint.sh
│ ├── model_setup.sh
│ ├── requirements-torchserve.txt
│ ├── torchserve_rocm62_requirements.patch
│ └── wan_handler.py
├── overrides/ # Optional values overrides
```

## 📚Prerequisits

The TorchServe workload is heavilly dependent on `torchserve-model-packager` workload, which is responsible for compressing the model into .zip and uploading it to MiniO object storage. For this workload it's important to be sure that compressed model is already available for downloading in MiniO.

You can run the workload `torchserve-model-packager` by:

```
helm template workloads/torchserve-model-packager/helm \
    -f workloads/torchserve-model-packager/helm/overrides/Wan2.1-1.3B-diffusers.yaml
    --name-template wan21 \
    | kubectl appply -f -
```

## 🚀 Quickstart

Deploy directly with:

```bash
helm template wan21 . | kubectl apply -f -
```

This sets up the pod, installs dependencies, downloads the model, creates the `.mar` archive, and launches TorchServe.

## 🌐 Access the API

Forward the API to your local machine:

`kubectl port-forward deployment/wan21-torchserve-wan21 8080:8080`

Send request:

```
curl -X POST http://localhost:8080/predictions/wan21 \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a scene of an astronaut riding a horse on mars",
    "width": 480,
    "height": 832,
    "num_frames": 81,
    "num_inference_steps": 40
  }' \
  --output "output-$(date +%Y%m%d%H%M%S).mp4"
```

## ⚙️ TorchServe Files Explained

| File                                     | Purpos                                                     |
| ---------------------------------------- | ---------------------------------------------------------- |
| `entrypoint.sh`                        | Installs dependencies, builds model, and starts TorchServe |
| `wan_handler.py`                       | Custom inference handler using HuggingFace Diffusers       |
| `config.properties`                    | Sets TorchServe ports, timeouts, etc.                      |
| `requirements-torchserve.txt`          | Dependencies for serving                                   |
| `torchserve_rocm62_requirements.patch` | Patch to adjust ROCm-specific install                      |
| `model_setup.sh`                       | Generates `.mar` file for torchserve                     |
