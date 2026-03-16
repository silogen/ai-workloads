<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Dataset Preprocessing with VeRL

Kubernetes Jobs convert dataset from MinIO or Hugging Face into VeRL-ready Parquet shards for SFT.

## Quick reference

- **Input**: `datasetRemote` pulls from MinIO, while `preprocess.localDatasetPath` skips the download step. Set `preprocess.mode` to `hfScript` for Hugging Face pulls and describe the dataset via `preprocess.hfScriptArgs.*`.
- **Transformation**: Point `preprocess.builtinScript` to any packaged VeRL helper or rely on the HF helper for Q&A (`questionField`/`answerField`) and multi-turn (`conversationField`) corpora. Increase `preprocess.hfScriptArgs.numProc` for faster `datasets.map` calls.
  - If `preprocess.builtinScript` is a relative path (e.g., `examples/data_preprocess/gsm8k.py`), it is automatically resolved against `preprocess.verlRootDir` (default: `/workspace/verl`).
- **Custom scripts**: Switch `preprocess.mode` to `customScript` to bake an inline Python helper (`preprocess.customScript`) into the ConfigMap.
- **Output**: `preprocess.outputDir` is the local staging folder; `outputRemotePath` mirrors the Parquet shards back to MinIO when set.
- **Access + auth**: make sure `bucketStorageHost`, `bucketCredentialsSecret`, and (if required) `hfTokenSecret` line up with your cluster. Private images go through `imagePullSecrets`.
- **Sizing**: tune `resources.requests/limits` for CPU-bound preprocessing. Jobs are namespace-agnostic; append `-n <ns>` to your kubectl commands.

`values.yaml` documents every option.

## Running the workload

Use an existing override or create a new one that sets bucket paths, auth, and selects the right `preprocess.mode` (`builtinScript`, `customScript`, or `hfScript`).

Custom scripts live directly under `preprocess.customScript`. When `preprocess.mode=customScript`, the chart renders that block into `/configs/custom_script.py` and automatically points `entrypoint.sh` at it, so the standard download/upload logic still runs before and after your custom Python.

Render the chart from the local directory and submit it to Kubernetes. This command is intended to be run from the `aim-fine-tuning` directory:
 ```bash
 helm template dolly-15k aimtrain-dataprep-verl/helm \
   --values aimtrain-dataprep-verl/helm/overrides/sft/dolly-15k.yaml \
   | kubectl create -f -
 ```


## Data format

The helper writes VeRL SFT rows (`data_source`, `messages`, `extra_info`). For details and downstream expectations, see the [VeRL data preparation guide](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html).
