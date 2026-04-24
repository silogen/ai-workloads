#!/bin/bash
set -euo pipefail

# Install ffmpeg
apt update && apt install ffmpeg=7:6.1.1-3ubuntu5 -y

# Install lerobot
cd /workload
git clone https://github.com/huggingface/lerobot.git
cd lerobot
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
    --policy.repo_id={{ .Values.hfFinetunedModelId }} \
    --steps={{ .Values.training.steps }} \
    --save_freq={{ .Values.training.save_freq }} \
    --eval_freq={{ .Values.training.eval_freq }} \
    --batch_size={{ .Values.training.batch_size }}{{ if .Values.extraArgs }} \{{ end }}
    {{- range $index, $item := .Values.extraArgs }}
    {{ $item }}{{- if ne $index (sub (len $.Values.extraArgs) 1) }} \{{ end }}
    {{- end }}
