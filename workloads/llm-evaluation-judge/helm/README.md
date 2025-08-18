# LLM-as-a-Judge Workloads

This helm chart implements evaluation of LLMs using LLM-as-a-Judge --- having an LLM provide inferences over an evaluation dataset, and letting another LLM judge the quality of the outputs.

The necessary Kubernetes and Helm files are stored here in `/workloads/llm-evaluation-judge/helm`, while the evaluation package source code and docker image build files are stored in `/docker/llm-evaluation`.

## Helm and Kubernetes files

The Helm templates are stored in `/workloads/llm-evaluation-judge/helm/templates`, the main template workload template being `evaluation_judge_template.yaml`. Default values can be found in `values.yaml`, with user-defined configurations stored in `/overrides`. We have included a few example override files for typical use cases.

A few extra resources are defined in `templates/`.
We use a `ConfigMap` (`templates/configmap.yaml`) to mount files directly to the cluster when running the workload. Anything stored in the `mount/` directory will be mounted.

## Docker Container

We define an associated evaluation package in `/docker/llm-evaluation`. This contains code to call the inference container over the evaluation dataset, and subsequently judge the outputs using a second judge container, writing results to MinIO storage.

This package is installed into a docker image, which can be used to run the evaluation container in the helm template. We use a Makefile to push new images to our GitHub registry. (`> make push`)

## Running

To run this evaluation workload with helm, use the template command and pipe it to kubectl apply:

```bash
cd workloads/llm-evaluation-judge
```

```bash
helm template helm -f overrides/prometheus-llama_3_8b-cnn_dailymail.yaml | kubectl apply -f - -n <your-namespace>
```
