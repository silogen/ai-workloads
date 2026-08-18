# Helper templates

# Interactive entrypoint
{{- define "interactiveEntrypoint" -}}
sleep 12h
{{- end -}}

# Install entrypoint
{{- define "installEntrypoint" -}}
# Setup DiffSynth
echo 'Installing DiffSynth...'
cd /workspace
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
git checkout {{ .Values.diffsynth_commit | default "090074e" }}
pip install .
{{- end -}}

# Download entrypoint
{{- define "downloadEntrypoint" -}}
# Setup MinIO, Download resources:
echo 'Setting up MinIO client...';
curl -s https://dl.min.io/client/mc/release/linux-amd64/mc \
  --create-dirs \
  -o /tmp/mc
chmod +x /tmp/mc
export PATH="${PATH}:/tmp/"
mc alias set minio-host ${BUCKET_STORAGE_HOST} ${BUCKET_STORAGE_ACCESS_KEY} ${BUCKET_STORAGE_SECRET_KEY}
echo 'Downloading base model directly to DiffSynth models directory...'
echo 'Downloading base model from: {{ printf "%s/%s" (.Values.bucketBasemodelPath | trimSuffix "/") (.Values.basemodelId | trimSuffix "/") }}/'
mc ls minio-host/{{ printf "%s/%s" (.Values.bucketBasemodelPath | trimSuffix "/") (.Values.basemodelId | trimSuffix "/") }}/ || echo "Base model path not found!"
mc cp --recursive \
  minio-host/{{ printf "%s/%s" (.Values.bucketBasemodelPath | trimSuffix "/") (.Values.basemodelId | trimSuffix "/") }}/ \
  /workspace/DiffSynth-Studio/models/{{ .Values.basemodelId | trimSuffix "/" }}
echo 'Downloading training dataset...'
echo 'Downloading training dataset from: {{ printf "%s/%s" (.Values.bucketDatasetPath | trimSuffix "/") (.Values.datasetId | trimSuffix "/") }}/'
mc ls minio-host/{{ printf "%s/%s" (.Values.bucketDatasetPath | trimSuffix "/") (.Values.datasetId | trimSuffix "/") }}/ || echo "Training dataset path not found!"
mc cp --recursive \
  minio-host/{{ printf "%s/%s" (.Values.bucketDatasetPath | trimSuffix "/") (.Values.datasetId | trimSuffix "/") }}/ \
  /workspace/local_resources/data/{{ .Values.datasetId | trimSuffix "/" }}
echo 'Verifying downloaded files...'
ls -la /workspace/DiffSynth-Studio/models/{{ .Values.basemodelId | trimSuffix "/" }}/ || echo "No model files found"
ls -la /workspace/local_resources/data/{{ .Values.datasetId | trimSuffix "/" }}/ || echo "No dataset files found"
{{- end -}}

# Finetune entrypoint
{{- define "finetuneEntrypoint" -}}
# Start finetuning
echo 'Starting finetuning...'
cd /workspace/DiffSynth-Studio
accelerate launch \
  --config_file /workspace/configs/accelerate_config.yaml \
  examples/wanvideo/model_training/train.py \
  {{- with .Values.finetune_config }}
  --dataset_base_path "{{ .dataset_base_path }}" \
  --dataset_metadata_path "{{ .dataset_metadata_path }}" \
  --height {{ .height }} \
  --width {{ .width }} \
  --num_frames {{ .num_frames }} \
  --dataset_repeat {{ .dataset_repeat }} \
  --model_paths '{{ .model_paths }}' \
  --learning_rate "{{ .learning_rate }}" \
  --num_epochs {{ .num_epochs }} \
  --remove_prefix_in_ckpt "{{ .remove_prefix_in_ckpt }}" \
  --output_path "{{ .output_path }}" \
  {{- if eq .architecture "lora" }}
  --lora_base_model "{{ .lora_base_model }}" \
  --lora_target_modules "{{ .lora_target_modules }}" \
  --lora_rank {{ .lora_rank }}
  {{- else if eq .architecture "full" }}
  --trainable_models {{ .trainable_models }}
  --max_timestep_boundary {{ .max_timestep_boundary }}
  --min_timestep_boundary {{ .min_timestep_boundary }}
  {{- end }}
  {{- end }}
{{- end -}}

# Upload entrypoint
{{- define "uploadEntrypoint" -}}
# Upload checkpoints
echo 'Uploading checkpoints...'
mc cp --recursive \
  {{ .Values.finetune_config.output_path | trimSuffix "/" }}/ \
  minio-host/{{- printf "%s/%s/%s/%s" (.Values.bucketBasemodelPath | trimSuffix "/") (.Values.basemodelId | trimSuffix "/") (last (splitList "/" .Values.finetune_config.output_path) | trimSuffix "/") (now | date "20060102-150405") -}}/ \
{{- end -}}

# Container resources helper
{{- define "container.resources" -}}
requests:
  {{- if .Values.resources.cpu }}
  cpu: "{{ .Values.resources.cpu }}"
  {{- end }}
  {{- if .Values.resources.gpus }}
  amd.com/gpu: "{{ .Values.resources.gpus }}"
  {{- end }}
limits:
  {{- if .Values.resources.cpu }}
  cpu: "{{ .Values.resources.cpu }}"
  {{- end }}
  {{- if .Values.resources.gpus }}
  amd.com/gpu: "{{ .Values.resources.gpus }}"
  {{- end }}
{{- end -}}
