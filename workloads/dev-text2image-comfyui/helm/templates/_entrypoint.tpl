{{- define "minio.setup" -}}
{{- $model := trimPrefix "s3://" .Values.model | trimSuffix "/" -}}
{{- $minioModel := printf "minio-host/%s" $model -}}
{{- $localModel := printf "/workload/%s/ComfyUI/models/checkpoints/%s" (.Values.metadata.user_id) (base $model) -}}
echo '--------------------------------------------'
echo 'Installing minio client'
echo '--------------------------------------------'
curl https://dl.min.io/client/mc/release/linux-amd64/mc \
      --create-dirs \
      -o /minio-binaries/mc
chmod +x /minio-binaries/mc
export PATH="${PATH}:/minio-binaries/"
mc alias set minio-host ${BUCKET_STORAGE_HOST} ${BUCKET_STORAGE_ACCESS_KEY} ${BUCKET_STORAGE_SECRET_KEY}
echo '--------------------------------------------'
echo 'Downloading the model from S3: {{ .Values.model }}'
echo '--------------------------------------------'
mkdir -p /workload/{{ .Values.metadata.user_id }}/ComfyUI/models/checkpoints/
mc cp --recursive {{$minioModel}} {{$localModel}} || { echo "Model copy from MinIO failed"; exit 1; }
{{- end }}

{{- define "comfyui.setup" }}
echo '--------------------------------------------'
echo 'Setting up ComfyUI environment'
echo '--------------------------------------------'
apt-get update && apt-get install -y git
pip install comfy-cli huggingface_hub[hf_transfer] hiredis $PIP_DEPS
yes | comfy --workspace=$COMFYUI_PATH install --skip-torch-or-directml --amd --restore
echo '--------------------------------------------'
{{- end }}

{{- define "comfyui.download_model" }}
{{- if not (hasPrefix "s3://" .Values.model) }}
echo '--------------------------------------------'
echo 'Downloading model from HuggingFace: {{ .Values.model }}'
echo '--------------------------------------------'
huggingface-cli download {{ .Values.model }} --local-dir $COMFYUI_PATH/models/checkpoints {{- if .Values.tag }} --include *{{ .Values.tag }}*safetensors{{- end }}

{{- end }}
{{- end }}

{{- define "comfyui.start" }}
echo '--------------------------------------------'
echo 'Starting ComfyUI'
echo '--------------------------------------------'

if [ -z "$MODEL_BIN_URL" ]; then
    echo "MODEL_BIN_URL is not set, skipped downloading."
else
    echo "Downloading model from URL: $MODEL_BIN_URL"
    mkdir -p $COMFYUI_PATH/models/checkpoints/
    env -C $COMFYUI_PATH/models/checkpoints/ curl -LO ${MODEL_BIN_URL} || { echo "Model download failed"; exit 1; }
fi

comfy --workspace=$COMFYUI_PATH launch -- --listen 0.0.0.0
{{- end }}

{{ define "entrypoint" }}
{{- include "comfyui.setup" . }}

{{- if and .Values.model (ne .Values.model "") }}
{{- if (hasPrefix "s3://" (toString .Values.model)) }}
{{- include "minio.setup" . }}
{{- else }}
{{- include "comfyui.download_model" . }}
{{- end }}
{{- end }}

{{- include "comfyui.start" . }}
{{- end }}
