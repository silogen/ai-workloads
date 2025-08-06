## Pretrained LLM inference with Megatron-LM on MI300X

This Helm Chart deploys the LLM Inference Megatron-LM workload.

## Prerequisites
The following prerequisites should be met before deploying this workload::

1. **Helm**: Install `helm`. Refer to the [Helm documentation](https://helm.sh/) for instructions.
2. **Secrets**: Secret containing the S3 storage provider's HMAC credentials should be created in the namespace, where workload runs. Default secret has name `minio-credentials` with keys `minio-access-key` and `minio-secret-key`.

## Deploying the Workload

To deploy workload pipe the result of `helm template` command to `kubectl apply`. Generally, full command looks as follows

```bash
helm template [release-name] <helm-chart-dir> [-f <path/to/overrides/xyz.yaml>] [--set <name>=<value>] | kubectl apply -f - [-n namespace]
```
where
- *release-name*: is the optional name of the helm release of the deployed job.
- *helm-chart-dir*: is the path to the directory where this helm chart is located.
- *-f path/to/overrides/xyz.yaml*: optional, can be multiple and presents a path to helm values overrides.
- *--set &lt;name>=&lt;value>*: optional, can be multiple and allows overriding single entry from the default values.
- *namespace*: is an optional kubernetes namespace of the deployment, if different from the default one.

An example of the command that will use default helm values and deploys workload to a default namespoace is shown below
```bash
helm template workloads/llm-inference-megatron-lm/helm/ | kubectl apply -f -
```

## Interacting with Deployed Model

### Verify Deployment

Check the deployment status:

```bash
kubectl get deployment [-n namespace]
```
You should see a deployment with a name starting with the prefix `llm-inference-megatron-lm-` up and running.

To see service deployed by the workload run

```bash
kubectl get svc [-n namespace]
```

The service should have a name starting with `llm-inference-megatron-lm-`. Note the port exposed by the service, it is expected to be the port `80`.


### Port Forwarding

Forward the port to access the service to the local machine at e.g. port `5000`. Assuming the service is named `llm-inference-megatron-lm-20250522-1521` and the port exposed is `80`, use the following command:

```bash
kubectl port-forward [-n namespace] svc/llm-inference-megatron-lm-20250522-1521 5000:80
```

### Test the model inference service

Send a request to the service to get a reply from the model using `curl` command:

```bash
curl -X PUT -H "Content-Type: application/json" -d '{"prompts": ["This is a test prompt."], "tokens_to_generate": 50}' http://localhost:5000/api
```

Another way to interact with the inference service is to use the `test_manual.py` script located in the `llm-inference-megatron-lm/helm/mount` directory. This script prompts for multiple questions interactively. To run it, use the following command (assuming repo root is your current directory):

```bash
python workloads/llm-inference-megatron-lm/helm/mount/test_manual.py localhost:5000
```

You can also run a quick sanity check to evaluate the coherence of the model by using the `coherence.py` script located in the `llm-inference-megatron-lm/helm/mount` directory. This file contains multiple questions along with their corresponding expected answers. When the model's generated response matches the expected answer, a point is awarded. At the end, the user can assess the model's coherence performance based on the total score. To run the evaluation, use the following command:

```bash
python workloads/llm-inference-megatron-lm/helm/mount/coherence.py localhost:5000
```
