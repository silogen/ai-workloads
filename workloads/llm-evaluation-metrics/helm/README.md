## Metrics Evaluation Workloads

This helm chart implements evaluation of LLMs using the BERTscore metric, comparing generated answers to a gold standard.

The necessary Kubernetes and Helm files are stored here in `/workloads/llm-evaluation-metrics/helm`, while the docker image and associated evaluation package are stored in /docker/evaluation-metrics.

### Helm and Kubernetes files

The Helm templates are stored in `/workloads/llm-evaluation-metrics/helm/templates`, the main template workload template being `metrics_evaluation_template_with_download.yaml`. Default parameters can be found in `values.yaml`, with user-defined configurations stored in `/overrides`.

A few extra resources are defined in `templates/`.
We use a `ConfigMap` (`templates/configmap.yaml`) to mount files directly to the cluster when running the workload. Anything stored in the `mount/` directory will be mounted.

### Docker Container

We define an associated evaluation package in `/docker/evaluation-metrics`. This contains code to call the inference container, and subsequently run the BERTscore metric evaluation, writing results to MinIO storage.

This package is installed into a docker image, which can be used to run the evaluation container in the helm template. We use a Makefile to push new images to our GitHub registry. (`> make push`)

## Running

To run this evaluation workload with helm, use the template command and pipe it to kubectl apply:

```bash
cd workloads/llm-evaluation-metrics
```

```bash
helm template helm -f overrides/bertscore_llama-3.1-8B_cnn-dailymail_values.yaml | kubectl apply -f - -n <your-namespace>
```

To cancel runs in between debugging, use

```bash
kubectl delete job <your-job-name> -n <your-namespace>
```
