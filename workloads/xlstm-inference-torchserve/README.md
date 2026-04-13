# Serving xLstm Model with Torchserve — Helm Deployment

This Helm chart deploys a TorchServe service serving xLSTM model (developed by NXAI). This Helm chart handles model setup, preprocessing and serving in a fully containerized GPU environment with AMD ROCm support.

## 📋 Prerequisites

- MinIO or S3-compatible storage with access credentials stored in Kubernetes secrets.
- Package and upload your model to MinIO storage using `torchserve-model-packager` helm chart:

```bash
helm template workloads/torchserve-model-packager/helm \
    -f workloads/torchserve-model-packager/helm/overrides/NXAI-XLSTM-7B.yaml \
    --name-template xlstm \
    | kubectl apply -f -
```

## 🔧 Project Structure

```
helm/
├── Chart.yaml                 # Helm chart metadata
├── values.yaml                # Main config values (e.g. image, ports, GPU, storage)
├── values.schema.json         # JSON schema for values validation
├── templates/                 # Helm templates and helpers for deployment, config, etc.
│   ├── _helpers.tpl           # Helpers common to other helm charts in the repo
│   ├── configmap.yaml
│   ├── deployment.yaml        # Main deployment template
│   └── service.yaml           # Service for TorchServe endpoints
├── mount/                     # Files mounted into the container via ConfigMap
│   ├── entrypoint.sh          # Main entrypoint script
│   ├── xlstm_handler.py        # Custom TorchServe handler for xlstm model
│   └── config.properties      # Configs for TorchServe ports, timeouts, etc.
└── overrides/                 # Optional values overrides
```

## 🚀 Quickstart

Deploy directly with:

```bash
helm template xlstm ./helm  | kubectl apply -f -
```

This sets up the pod, downloads the packaged xLSTM model from MinIO, archives it into a `.mar` file, and starts the inference server.

Override with:

```bash
helm template test ./helm -f ./helm/overrides/models/xlstm-7b.yaml | kubectl apply -f -
```

## 🌐 Access the API

**Port forward to access the service:**

```bash
kubectl port-forward deployment/xlstm-inference-torchserve-test 8080:8080
```

## 📹 Usage Example

Generate a response from xLSTM with given prompt:

```bash
python xlstm_query.py --url "http://localhost:8080/predictions/NX-AI_xLSTM-7b" --prompt "there is a whale in this package that does not look like a"
```
