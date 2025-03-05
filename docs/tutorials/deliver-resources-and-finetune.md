# Deliver model and data to cluster MinIO, then run finetune

This walk-through shows how to download a model and some data from HuggingFace Hub to a cluster-internal MinIO storage server, and then launch finetuning Jobs that use those resources. The checkpoints are also synced into the same cluster-internal MinIO storage. Finally, an inference workload is spawned to make it possible to discuss with the newly finetuned model.

We should have a working cluster, setup by a cluster administrator. The access to that cluster is provided with a suitable Kubeconfig file. We don't require administrator permissions to run through this walk-through.

## 1: Setup for the walk-through, programs used, instructions for monitoring

### Required program installs

Programs that are used in this tutorial:

* [kubectl](https://kubernetes.io/docs/tasks/tools/#kubectl)

* [helm](https://helm.sh/docs/intro/install/)

* [k9s](https://k9scli.io/topics/install/)

* [jq](https://jqlang.org/download/)

* [curl](https://everything.curl.dev/install/index.html)

At least curl and often jq too are commonly installed in many distributions out of the box.

### Additional cluster setup

Additional cluster setup. This does the following:

* Adds a namespace, where we will conduct all our work. We will use the `silo` namespace.

* Adds an External Secret to get the credentials to access the MinIO storage from our namespace.

    - This depends on a ClusterSecretStore called `k8s-secret-store` being already setup by a cluster admin, which the cluster should have.

* Adds a LocalQueue so that our Jobs schedule intelligently.

    - This references the ClusterQueue `kaiwo` which should already be setup by a cluster admin.

We will use the helm chart in `workloads/k8s-namespace-setup/helm` and the overrides in `workloads/k8s-namespace-setup/helm/overrides/`.

```bash
kubectl create namespace "silo"
helm template workloads/k8s-namespace-setup/helm \
  --values workloads/k8s-namespace-setup/helm/overrides/tutorial-01-local-queue.yaml \
  --values workloads/k8s-namespace-setup/helm/overrides/tutorial-01-storage-access-external-secret.yaml \
  --namespace "silo" \
  | kubectl apply -f - --namespace silo
```

### Monitoring progress, logs, and GPU utilization with k9s

We're interested to see a progress bar of the finetuning training, seeing any messages that a workload logs, and we also want to verify that our GPU Jobs
are consuming our compute relatively effectively. This information can be fetched from our Kubernetes cluster in many ways, but one convenient and recommended way us using [k9s](https://k9scli.io/).
We recommend the official documentation for more thorough guidance, but this section shows some basic commands to get what we want here.

To get right to the Jobs view in the namespace we're using in this walk-through, we can run:

```bash
k9s --namespace silo --command Jobs
```

Choose a Job using `arrow keys` and `Enter` to see the Pod that it spawned, then `Enter` again to see the Container in the Pod. From here, we can do three things:

* Look at the logs by pressing `l`. The logs show any output messages produced during the workload runtime.

* Attach to the output of the container by pressing `a`. This is particularly useful to see the interactive progress bar of a finetuning run.

* Spawn a shell inside the container by pressing `s`. Inside the shell we can run `watch -n0.5 rocm-smi` to get a view of the GPU utilization that updates every 0.5s.

Return from any regular `k9s` view with `Esc` .

## 2. Run a workload to deliver data + model

We will use the helm chart in `workloads/deliver-huggingface-example-resources/helm/chart` . We will use it to deliver a Tiny-Llama 1.1B parameter model, and an Argilla single-turn response supervised finetuning dataset.

Our user input file is in `workloads/deliver-huggingface-example-resources/helm/overrides/tutorial-01-tiny-llama-argilla.yaml`, we can change the `modelID` and the `dataScript` to use different models and data.
```bash
helm template workloads/deliver-huggingface-example-resources/helm/chart \
  --values workloads/deliver-huggingface-example-resources/helm/overrides/tutorial-01-tiny-llama-argilla.yaml \
  --name-template "deliver-tiny-llama-and-argilla" \
  --namespace "silo" \
  | kubectl apply -f -
```

The [logs](#monitoring-progress-logs-and-gpu-utilization-with-k9s) will show a model staging download and upload, then data download, preprocessing, and upload.

## 3. Scaling finetuning: Hyperparameter tuning with parallel Jobs

At the hyperparameter tuning stage, we run many parallel Jobs while varying a hyperparameter to find the best configuration.
Here we are going to look for the best rank parameter `r` for [LoRA](https://arxiv.org/pdf/2106.09685).

To define the finetuning workload, we will use the helm chart in `workloads/LLM-finetune-silogen-engine/helm/chart` .
Our user input file is in `workloads/LLM-finetune-silogen-engine/overrides/tutorial-01-finetune-lora.yaml` . This also includes the finetuning hyperparameters - you can change them in the file to experiment, or use `--set` with helm templating to change an individual value.

Let's create ten different finetunes to try out different LoRA ranks:

```bash
run_id=alpha
for r in 4 6 8 10 12 16 20 24 32 64; do
  name="tiny-llama-argilla-r-sweep-$run_id-$r"
  helm template workloads/LLM-finetune-silogen-engine/helm/chart \
    --values workloads/LLM-finetune-silogen-engine/helm/overrides/tutorial-01-finetune-lora.yaml \
    --name-template $name \
    --namespace "silo" \
    --set finetuning_config.peft_conf.peft_kwargs.r=$r \
    --set "checkpointsRemote=default-bucket/experiments/$name" \
    | kubectl apply -f -
done
```

For each Job we can see logs, a progress bar, and that Job's GPU utilization following the [instructions above](#monitoring-progress-logs-and-gpu-utilization-with-k9s).
If these Jobs get relaunched, they are setup to continue from the existing checkpoints. If we instead want to re-run from scratch, we can just change the `run_id` variable that is defined before the for loop.

## 4. Scaling finetuning: Multi-GPU training

Beside parallel Jobs, we can also take advantage of multiple GPUs by using them for parallel compute. This can be helpful for more compute demanding Jobs, and necessary with larger models.

Let's launch an 8GPU run of full-parameter finetuning:

```bash
name="tiny-llama-argilla-v1"
helm template workloads/LLM-finetune-silogen-engine/helm/chart \
  --values workloads/LLM-finetune-silogen-engine/overrides/tutorial-01-finetune-full-param.yaml \
  --name-template $name \
  --namespace "silo" \
  --set "checkpointsRemote=default-bucket/experiments/$name" \
  --set "finetuningGpus=8" \
  | kubectl apply -f -
```

We can see logs, a progress bar, and the full 8-GPU compute utilization following the [instructions above](#monitoring-progress-logs-and-gpu-utilization-with-k9s).
The training steps of this multi-gpu training run take merely 75 seconds, which reflects the nature of finetuning:
fast, iterative, with a focus on flexible experimentation.

If we want to compare to an equivalent single-GPU run, we can run:

```bash
name="tiny-llama-argilla-v1-singlegpu"
helm template workloads/LLM-finetune-silogen-engine/helm/chart \
  --values workloads/LLM-finetune-silogen-engine/helm/overrides/tutorial-01-finetune-full-param.yaml \
  --name-template $name \
  --namespace "silo" \
  --set "checkpointsRemote=default-bucket/experiments/$name" \
  --set "finetuningGpus=1" \
  | kubectl apply -
```

The training steps for this single-GPU run take around 340 seconds.
Thus the full-node training yields a speedup ratio of around 0.22 (4.5x speed).
Even higher speedups are achieved in pretraining, which benefits hugely from optimizations.

## 5. Inference with a finetuned model

After training the model, we'll want to discuss with it. For this we will use the helm chart in `workloads/llm-inference-vllm/helm` .

Let's deploy the full-parameter finetuned model:

```bash
name="tiny-llama-argilla-v1"
helm template workloads/llm-inference-vllm/helm \
  --values workloads/llm-inference-vllm/helm/overrides/tutorial-01-inference.yaml \
  --set "model=s3://default-bucket/experiments/$name/checkpoint-final" \
  --set "vllm_engine_args.served_model_name=$name" \
  --name-template "$name" \
  | kubectl apply --namespace "silo" -f -
```

We can change the `name` to different experiment names to deploy other models. Note that discussing with the LoRA adapter models with these workloads requires us to merge the final adapter. This can be achieved during finetuning by adding `--set mergeAdapter=true` and additionally in the deploy command, we have to refer to the merged model, changing the path to `--set "model=s3://default-bucket/experiments/$name/checkpoint-final-merged"` .

To discuss with the model, we first need to setup a connection to it. Since this is not a public-internet deployment, we'll do this simply by starting a background port-forwarding process:

```bash
name="tiny-llama-argilla-v1"
kubectl port-forward deployments/llm-inference-vllm-$name 8080:8080 -n silo >/dev/null &
portforwardPID=$!
```

Now we can discuss with the model, using curl:

```bash
name="tiny-llama-argilla-v1"
question="What are the top five benefits of eating a large breakfast?"
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'$name'",
        "messages": [
            {"role": "user", "content": "'"$question"'"}
        ]
    }' | jq ".choices[0].message.content" --raw-output
```

We can test the limits of the model with our own questions. Since this is a model with a relatively limited capacity, its answers are often delightful nonsense.

When we want to stop port-forwarding, we can just run:

```bash
kill $portforwardPID
```
