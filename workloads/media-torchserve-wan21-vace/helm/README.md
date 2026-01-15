# 🧠 TorchServe for VACE — Helm Deployment

This Helm chart deploys a custom TorchServe instance serving VACE (All-in-One Video Creation and Editing) model. It handles model setup, preprocessing and serving in a fully containerized GPU environment with AMD ROCm support.

## 🔧 Project Structure

```
helm/
├── Chart.yaml                 # Helm chart metadata
├── values.yaml                # Main config values (e.g. image, ports, GPU)
├── templates/                 # Helm templates for deployment, config, etc.
│   ├── _helpers.tpl
│   ├── configmap.yaml
│   ├── deployment.yaml
│   └── service.yaml
├── mount/                     # Files mounted into the container via ConfigMap
│   ├── config.properties
│   ├── entrypoint.sh
│   └── vace_handler.py
├── overrides/                 # Optional values overrides
│   └── kaiwo/
```

## 📚 Prerequisites

The TorchServe workload is heavily dependent on `torchserve-model-packager` workload, which is responsible for compressing the model into .zip and uploading it to MinIO object storage. For this workload it's important to be sure that compressed model is already available for downloading in MinIO.

You can run the workload `torchserve-model-packager` by:

```bash
helm template workloads/torchserve-model-packager/helm \
    -f workloads/torchserve-model-packager/helm/overrides/Wan2.1-VACE-1.3B-diffusers.yaml \
    --name-template vace \
    | kubectl apply -f -
```

## 🚀 Quickstart

Deploy directly with:

```bash
helm template vace . | kubectl apply -f -
```

This sets up the pod, installs dependencies, downloads the model, creates the `.mar` archive, and launches TorchServe.

## 🌐 Access the API

Forward the API to your local machine:

`kubectl port-forward svc/vace-inference-torchserve-vace 8080:80`

## 📹 Usage Examples

### Basic Video Generation

Generate a video from text prompt:

```bash
python vace_query.py --url "http://localhost:8080/predictions/Wan2.1-VACE-1.3B" --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

### Video Inpainting

Change certain objects with inpainting:

```bash
python vace_query.py --url "http://localhost:8080/predictions/Wan2.1-VACE-1.3B" \
    --prompt "The cat is wearing pink sunglasses. Everything else remains unchanged." \
    --video <path-to-input-video> \
    --task inpainting
```

### Video Outpainting

Extend an existing video with outpainting:

```bash
python vace_query.py --url "http://localhost:8080/predictions/Wan2.1-VACE-1.3B" \
    --prompt "The surfboard stretches further along the sandy beach beneath the cat, revealing more of its colorful design and sunlit contours. Above, a flock of seagulls soars through the clear blue sky, bringing a sense of movement to the peaceful coastal scene. The background remains gently blurred to keep the focus on the cat." \
    --video <path-to-input-video> \
    --task outpainting
```

## ⚙️ TorchServe Files Explained

| File                  | Purpose                                                    |
|-----------------------|------------------------------------------------------------|
| `entrypoint.sh`       | Installs dependencies, builds model, and starts TorchServe |
| `vace_handler.py`     | Custom inference handler for VACE video generation         |
| `config.properties`   | Sets TorchServe ports, timeouts, etc.                      |
