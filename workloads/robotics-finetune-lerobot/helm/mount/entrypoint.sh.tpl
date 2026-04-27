#!/bin/bash
set -euo pipefail

# Install ffmpeg
apt-get update && apt-get install -y --no-install-recommends ffmpeg=7:6.1.1-3ubuntu5 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install lerobot
cd /workload
git clone --depth 1 https://github.com/huggingface/lerobot.git
cd lerobot
{{- if .Values.setup.lerobotGitRef }}
git checkout {{ .Values.setup.lerobotGitRef }}
{{- end }}
{{- if .Values.setup.lerobotExtraPackages }}
pip install -e ".[{{ .Values.setup.lerobotExtraPackages }}]"
{{- else }}
pip install -e .
{{- end }}

# Run training script and upload final checkpoint to Hugging Face Hub
lerobot-train \
    --dataset.repo_id={{ .Values.hfDatasetId }} \
    --policy.type={{ .Values.policy.type }} \
    {{- if .Values.policy.hfPretrainedModelId }}
    --policy.pretrained_path={{ .Values.policy.hfPretrainedModelId }} \
    {{- end }}
    --output_dir=/workload/outputs \
    --job_name={{ .Values.jobName }} \
    --policy.repo_id={{ required "hfFinetunedModelId is required!" .Values.hfFinetunedModelId | quote }} \
    --steps={{ .Values.training.steps }} \
    --save_freq={{ .Values.training.save_freq }} \
    --eval_freq={{ .Values.training.eval_freq }} \
    --batch_size={{ .Values.training.batch_size }}{{ if .Values.extraArgs }} \{{ end }}
    {{- range $index, $item := .Values.extraArgs }}
    {{ $item }}{{- if ne $index (sub (len $.Values.extraArgs) 1) }} \{{ end }}
    {{- end }}
