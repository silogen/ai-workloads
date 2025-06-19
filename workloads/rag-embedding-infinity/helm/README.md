# LLM Inference with vLLM

This Helm Chart deploys the embedding inference (via infinity) workload.


## Deploying the Workload

```bash
helm template [optional-release-name] <helm-dir> -f <overrides/xyz.yaml> --set <name>=<value> | kubectl apply -f -
```

### Example commands

Use default settings:

```bash
helm template e5-large . | kubectl apply -f -
```

Use custom model (that works with infinity):

```bash
helm template bilingual-embedding-large . --set model=Lajavaness/bilingual-embedding-large | kubectl apply -f -
```

The model will be automatically downloaded before starting the inference server.

## User Input Values

Refer to the `values.yaml` file for the user input values you can provide, along with instructions.

## Interacting with Deployed Model

### Verify Deployment

Check the deployment status:

```bash
kubectl get deployment
```

### Port Forwarding

Forward the port to access the service:

```bash
kubectl port-forward services/rag-embedding-infinity 7997:7997
```

### Test the Deployment

Infinity server UI will be accessible at http://0.0.0.0:7997/docs.

Send a test request to verify the service:

```bash
curl -X POST http://0.0.0.0:7997/embeddings \
     -H "Content-Type: application/json" \
     -d '{"model":"intfloat/multilingual-e5-large-instruct","input":["Two cute cats."]}'
```
