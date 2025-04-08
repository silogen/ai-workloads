# Finetuning: adding language-capabilities to a model

Adding a new language to model usually follows at least two stages: first, continued pretraining to build understanding and basic capacity to generate that language, and second, finetuning to make the model e.g. follow instructions. This tutorial handles that second stage of instruction-tuning for [Odia language](https://en.wikipedia.org/wiki/Odia_language).

The original basemodel is [Qwen1.5-7B](https://huggingface.co/Qwen/Qwen1.5-7B). The first stage of continued pretraining to add general Odia understanding and generation abilities to Qwen have already been done by the [OdiaGenAI](https://www.odiagenai.org/) organization. That continued pretrained model is available openly [here](https://huggingface.co/OdiaGenAI-LLM/qwen_1.5_odia_7b). A relevant comparison point for this tutorial is the [chat-finetuned version](https://huggingface.co/Qwen/Qwen1.5-7B-Chat) of the Qwen1.5 basemodel, which should have the capability to follow instructions, but is not specifically meant for Odia.

Note that access to the Odia continued pretraining version of the Qwen model requires signing the request on Huggingface. This also means that for downloading the model, we'll need to use a HuggingFace access token. See instructions [here](https://huggingface.co/docs/hub/en/security-tokens). ⚠️ Note: This tutorial does not add the HF Token to the cluster yet. You need to add it yourself. ⚠️

The tutorial includes cluster setup, model and data downloads, finetuning, and finally inference.
We should start with a working cluster, setup by a cluster administrator using [Cluster-forge](https://github.com/silogen/cluster-forge). The access to that cluster is provided with a suitable Kubeconfig file.

## 1: Setup for the tutorial, programs used, instructions for monitoring

⚠️ WARNING: This tutorial does not handle adding the HF Token to the cluster yet. Coming soon. Before then, to run this tutorial, you are responsible for adding your HF Token as a secret called `hf-token` with the key `hf-token` in the `silo` namespace. ⚠️

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

    - This depends on a ClusterSecretStore called `k8s-secret-store` being already setup by a cluster admin, and the MinIO API credentials being secret there.
    The cluster should have these by default.

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

⚠️TODO: Guidance on adding the HF Token⚠️

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


## 2. Fetch data and models

First we'll fetch the model and data for the finetuning.
We will use the helm charts in `workloads/download-huggingface-model-to-bucket/helm` and `workloads/download-data-to-bucket/helm`. We will download a Qwen1.5 7B-parameter model, one English instruction dataset and five different Odia-language single-turn instruction finetuning datasets. These datasets cover slightly different areas, including open instructions, context-based question-answering, translation, and identity answers. The identity answers aim to make our model call itself Olive, and tell that it is from the OdiaGenAI project. Our user input files are in `workloads/download-huggingface-model-to-bucket/helm/overrides/tutorial-02-qwen-odia.yaml` and `workloads/download-data-to-bucket/helm/overrides/tutorial-02-odia-data.yaml`

```bash
helm template workloads/download-huggingface-model-to-bucket/helm \
  --values workloads/download-huggingface-model-to-bucket/helm/overrides/tutorial-02-qwen-odia.yaml \
  --name-template "download-odia-qwen-odia" \
  --namespace "silo" \
  | kubectl apply -f -
helm template workloads/download-data-to-bucket/helm \
  --values workloads/download-data-to-bucket/helm/overrides/tutorial-02-odia-data.yaml \
  --name-template "download-odia-data" \
  --namespace "silo" \
  | kubectl apply -f -
```

The [logs](#monitoring-progress-logs-and-gpu-utilization-with-k9s) will show a model staging download and upload, then data download, preprocessing, and upload.

## 3. Finetune an Odia model on 8GPUs
We will run our Odia language finetuning using 8 GPUs in a data parallel setup. For finetuning, we'll use the helm chart in  `workloads/llm-finetune-silogen-engine/helm`, and the user input file in `workloads/llm-finetune-silogen-engine/helm/overrides/tutorial-02-qwen-odia-instruct-v1.yaml`.

This training job takes around 13 hours, because the combination of 6 datasets is large and we go through that combination 6 times. We have found that a large number of training steps is necessary to enable the model to learn act reasonably in Odia language. We suspect this has something to do with the Odia script, which is not well covered by the Qwen tokenizer, and leads to very long sequences produced character or even diacritic at a time.

Let's launch the finetuning Job:
```bash
name="qwen-odia-instruct-v1"
helm template workloads/llm-finetune-silogen-engine/helm \
  --values workloads/llm-finetune-silogen-engine/helm/overrides/tutorial-02-qwen-odia-instruct-v1.yaml \
  --name-template $name \
  --namespace "silo" \
  --set "checkpointsRemote=default-bucket/experiments/$name" \
  --set "finetuningGpus=8" \
  | kubectl apply -f -
```
can see logs, a progress bar, and the full 8-GPU compute utilization following the [instructions above](#monitoring-progress-logs-and-gpu-utilization-with-k9s).

## 4. Compare the official Qwen1.5-7B-Chat model, the Odia continued pretraining basemodel, and the Odia-finetuned model:
The Qwen1.5-7B-Chat is a general chat-finetuned version of the same Qwen basemodel that our Odia continued pretraining started from. Thus it is a good point of comparison. Additionally we'll deploy the Odia continued pretraining basemodel, to see how the instruct-finetuning changed it.

For inference deployments, we will use the helm chart in  `workloads/llm-inference-vllm/helm/chart`.

Deploy the models with the following commands. Note that the Qwen1.5-7B-Chat model we're getting directly from HuggingFace, while the other two models we're fetching from the cluster internal bucket storage.
```bash
name="qwen-base-chat"
helm template workloads/llm-inference-vllm/helm \
  --set "model=Qwen/Qwen1.5-7B-Chat" \
  --set "vllm_engine_args.served_model_name=$name" \
  --name-template "$name" \
  | kubectl create --namespace "silo" -f -
name="qwen-odia-base"
helm template workloads/llm-inference-vllm/helm \
  --set "model=s3://default-bucket/models/OdiaGenAI/LLM_qwen_1.5_odia_7b" \
  --set "vllm_engine_args.served_model_name=$name" \
  --name-template "$name" \
  | kubectl create --namespace "silo" -f -
name="qwen-odia-instruct-v1"
helm template workloads/llm-inference-vllm/helm \
  --set "model=s3://default-bucket/experiments/$name/checkpoint-final" \
  --set "vllm_engine_args.served_model_name=$name" \
  --name-template "$name" \
  | kubectl create --namespace "silo" -f -
```

To discuss with the models, we need to setup connections to them. Since these are not public-internet deployments, we'll do this simply by starting background port-forwarding processes:
```bash
name="qwen-base-chat"
kubectl port-forward deployments/llm-inference-vllm-$name 8080:8080 --namespace silo >/dev/null &
qwenchatPID=$!
name="qwen-odia-base"
kubectl port-forward deployments/llm-inference-vllm-$name 8090:8080 --namespace silo >/dev/null &
odiabasePID=$!
name="qwen-odia-instruct-v1"
kubectl port-forward deployments/llm-inference-vllm-$name 8100:8080 --namespace silo >/dev/null &
odiainstructPID=$!
```

Now we can talk to the models. We'll ask them about the difference between physics and chemistry, in Odia:
```bash
question="ପଦାର୍ଥ ବିଜ୍ଞାନ ଏବଂ ରସାୟନ ବିଜ୍ଞାନ ମଧ୍ୟରେ କ’ଣ ପାର୍ଥକ୍ୟ ଅଛି?"
temperature=0.0
max_tokens=2048
presence_penalty=1.2
min_p=0.05
echo -e "\n\nQwen 1.5 7B Chat:"
name="qwen-base-chat"
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'$name'",
        "messages": [
            {"role": "user", "content": "'"$question"'"}
        ],
        "temperature": '$temperature',
        "max_tokens": '$max_tokens',
        "presence_penalty": '$presence_penalty',
        "min_p": '$min_p'
    }' 2>/dev/null | jq ".choices[0].message.content" --raw-output | fold -s | sed 's/^/  /'
echo -e "\n\nQwen 1.5 7B Odia-CPT Base:"
name="qwen-odia-base"
curl http://localhost:8090/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'$name'",
        "messages": [
            {"role": "user", "content": "'"$question"'"}
        ],
        "temperature": '$temperature',
        "max_tokens": '$max_tokens',
        "presence_penalty": '$presence_penalty',
        "min_p": '$min_p'
    }' 2>/dev/null | jq ".choices[0].message.content" --raw-output | fold -s | sed 's/^/  /'
echo -e "\n\nQwen 1.5 Odia-CPT Instruction-tuned model v1:"
name="qwen-odia-instruct-v1"
curl http://localhost:8100/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'$name'",
        "messages": [
            {"role": "user", "content": "'"$question"'"}
        ],
        "temperature": '$temperature',
        "max_tokens": '$max_tokens',
        "presence_penalty": '$presence_penalty',
        "min_p": '$min_p'
    }' 2>/dev/null | jq ".choices[0].message.content" --raw-output | fold -s | sed 's/^/  /'
```
We'll find that the general Qwen chat-model does not keep to Odia, it easily switches to e.g. Hindi or Chinese.
The Odia continued pretraining basemodel does answer in Odia, but since it is just a continuation model, the text it produces is not
really answer to the question, but rather rambling.
The finetuned model answers the question in Odia.

When we're done chatting with the models, we can kill the port-forwards with:
```bash
kill $qwenchatPID $odiabasePID $odiainstructPID
```
and we can shut down the inference deployments with:
```bash
for name in qwen-base-chat qwen-odia-base qwen-odia-instruct-v1; do
  kubectl delete deployment --namespace silo llm-inference-vllm-$name
done
```
