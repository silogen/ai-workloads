# Tutorial 06: Package and Serve Wan2.1 with TorchServe

This tutorial shows how to prepare the Wan2.1 model for TorchServe, upload it to a cluster-internal MinIO storage, and then deploy a TorchServe workload that serves the model behind an API endpoint. The process consists of two steps: **packaging the model** and **serving it**.

## 1. Setup

Follow the setup in the [tutorial pre-requisites section](./tutorial-00-prerequisites.md).

---

## 2. Package model into MinIO

Before TorchServe can serve a model, it needs a model `.zip` archive. We use the workload `torchserve-model-packager` to compress the Wan2.1 model and upload it to MinIO storage. Its user input file is:

```bash
workloads/torchserve-model-packager/helm/overrides/tutorial-06-package.yaml
```

Run:

```bash
helm template workloads/torchserve-model-packager/helm \
  --values workloads/torchserve-model-packager/helm/overrides/tutorial-06-package.yaml \
  --name-template wan21-packager \
  | kubectl create -f -
```

This job downloads Wan2.1 weights, prepares TorchServe assets, and writes the `.zip` file into MinIO.

You can follow logs and progress as described in [the monitoring section](./tutorial-00-prerequisites.md#monitoring-progress-logs-and-gpu-utilization-with-k9s).

---

## 3. Deploy TorchServe with Wan2.1

Once the model archive is available in MinIO, we can deploy TorchServe itself.

The workload for this is `media-torchserve-wan21`. Its user input file is:

```bash
workloads/media-torchserve-wan21/helm/overrides/tutorial-06-serve.yaml
```

Run:

```bash
helm template workloads/media-torchserve-wan21/helm \
  --values workloads/torchserve-wan21/helm/overrides/tutorial-06-serve.yaml \
  --name-template wan21-serve \
  | kubectl apply -f -
```

This creates a GPU-enabled TorchServe deployment, installs dependencies, mounts configuration files, downloads the `.zip` from MinIO, creates `.mar` archive, and starts serving.

---

## 4. Access the API

Forward TorchServe’s REST API to your local machine:

```bash
kubectl port-forward deployment/wan21-serve-media-torchserve-wan21 8080:8080
```

Now you can send a test request to generate video:

```bash
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

The output video will be written to the current directory.

When finished, you can stop the port-forwarding process and clean up the deployment:

```bash
kubectl delete deployment wan21-serve-media-torchserve-wan21
```

---

## Next Steps

This tutorial showed how to package Wan2.1 and serve it through TorchServe. The natural next step is to deploy additional handlers for different models.
