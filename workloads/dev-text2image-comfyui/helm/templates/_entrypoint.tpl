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
pip install huggingface_hub[hf_transfer] hiredis $PIP_DEPS

COMFYUI_VERSION="{{ default "v0.18.1" .Values.comfyui_version }}"
COMFYUI_MANAGER_VERSION="{{ default "3.39.3" .Values.comfyui_manager_version }}"
COMFYUI_MANAGER_PATH="$COMFYUI_PATH/custom_nodes/ComfyUI-Manager"

mkdir -p "$(dirname "$COMFYUI_PATH")"
if [ -d "$COMFYUI_PATH/.git" ]; then
    git -C "$COMFYUI_PATH" fetch --depth 1 origin "refs/tags/${COMFYUI_VERSION}:refs/tags/${COMFYUI_VERSION}" 2>/dev/null \
        || git -C "$COMFYUI_PATH" fetch --depth 1 origin "$COMFYUI_VERSION"
    git -C "$COMFYUI_PATH" checkout -f "$COMFYUI_VERSION"
else
    if [ -e "$COMFYUI_PATH" ] && [ -n "$(ls -A "$COMFYUI_PATH" 2>/dev/null)" ]; then
        echo "ERROR: $COMFYUI_PATH exists, is not a git repository, and is not empty; remove it or empty it before retrying." >&2
        exit 1
    fi
    git clone --branch "$COMFYUI_VERSION" --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_PATH"
fi

sed -E '/^(torch|torchvision|torchaudio)([><=!~ ]|$)/d' "$COMFYUI_PATH/requirements.txt" > /tmp/requirements-no-torch.txt
pip install -r /tmp/requirements-no-torch.txt

mkdir -p "$COMFYUI_PATH/custom_nodes"
if [ -d "$COMFYUI_MANAGER_PATH/.git" ]; then
    git -C "$COMFYUI_MANAGER_PATH" fetch --depth 1 origin "refs/tags/${COMFYUI_MANAGER_VERSION}:refs/tags/${COMFYUI_MANAGER_VERSION}" 2>/dev/null \
        || git -C "$COMFYUI_MANAGER_PATH" fetch --depth 1 origin "$COMFYUI_MANAGER_VERSION"
    git -C "$COMFYUI_MANAGER_PATH" checkout -f "$COMFYUI_MANAGER_VERSION"
else
    if [ -e "$COMFYUI_MANAGER_PATH" ] && [ -n "$(ls -A "$COMFYUI_MANAGER_PATH" 2>/dev/null)" ]; then
        echo "ERROR: $COMFYUI_MANAGER_PATH exists, is not a git repository, and is not empty; remove it or empty it before retrying." >&2
        exit 1
    fi
    git clone --branch "$COMFYUI_MANAGER_VERSION" --depth 1 https://github.com/ltdrdata/ComfyUI-Manager.git "$COMFYUI_MANAGER_PATH"
fi
pip install -r "$COMFYUI_MANAGER_PATH/requirements.txt"

echo '--------------------------------------------'
echo 'Verifying ROCm PyTorch'
echo '--------------------------------------------'
python -c "import torch; print(f'torch {torch.__version__}, HIP available: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"
echo '--------------------------------------------'
{{- end }}

{{- define "comfyui.download_model" }}
{{- if not (hasPrefix "s3://" .Values.model) }}
echo '--------------------------------------------'
echo 'Downloading model from HuggingFace: {{ .Values.model }}'
echo '--------------------------------------------'
hf download {{ .Values.model }} --local-dir "$COMFYUI_PATH/models/checkpoints" {{- if .Values.tag }} --include *{{ .Values.tag }}*safetensors{{- end }}

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
    mkdir -p "$COMFYUI_PATH/models/checkpoints/"
    env -C "$COMFYUI_PATH/models/checkpoints/" curl -LO "${MODEL_BIN_URL}" || { echo "Model download failed"; exit 1; }
fi

cd "$COMFYUI_PATH" && python main.py --listen 0.0.0.0
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
