{{- define "minio.setup" -}}
{{- $model := trimPrefix "s3://" .Values.model | trimSuffix "/" -}}
{{- $minioModel := printf "minio-host/%s" $model -}}
{{- $localModel := printf "/workload/%s" $model -}}
echo '--------------------------------------------'
echo 'Installing minio client'
echo '--------------------------------------------'
curl https://dl.min.io/client/mc/release/linux-amd64/mc \
      --create-dirs \
      -o /minio-binaries/mc
chmod +x /minio-binaries/mc
export PATH="${PATH}:/minio-binaries/"
if ! command -v mc >/dev/null 2>&1; then
  echo "ERROR: MinIO client (mc) is not available after installation."
  exit 1
fi
mc alias set minio-host ${BUCKET_STORAGE_HOST} ${BUCKET_STORAGE_ACCESS_KEY} ${BUCKET_STORAGE_SECRET_KEY} || { echo "ERROR: Failed to set MinIO alias."; exit 1; }
echo '--------------------------------------------'
echo 'Checking if model exists in MinIO storage'
echo '--------------------------------------------'
echo "Looking for files at: {{$minioModel -}}/"
FILE_COUNT=$(mc find {{$minioModel -}}/ 2>/dev/null | wc -l)
if [ "$FILE_COUNT" -eq 0 ]; then
  echo "ERROR: No files found at '{{$minioModel -}}/' or path does not exist."
  echo "DEBUG: mc find command output:"
  mc find {{$minioModel -}}/ 2>&1
  exit 1
fi
echo "✓ Found $FILE_COUNT files at {{$minioModel -}}/"
echo '--------------------------------------------'
echo 'Downloading the model to the local container'
echo '--------------------------------------------'
if ! mc mirror {{$minioModel -}}/ {{$localModel -}}/ 2> /tmp/mc_mirror_error.log; then
  echo "ERROR: Failed to download model from '{{$minioModel -}}/'."
  echo "DEBUG: mc mirror command stderr output:"
  cat /tmp/mc_mirror_error.log
  exit 1
fi
{{- end -}}

{{- define "vllm.start" -}}
{{- $modelPath := .Values.model -}}
{{- if (hasPrefix "s3://" .Values.model) -}}
  {{- $modelPath = printf "/workload/%s" (trimPrefix "s3://" .Values.model | trimSuffix "/") -}}
{{- end -}}
echo '--------------------------------------------'
echo 'Starting vLLM'
echo '--------------------------------------------'
pip install huggingface_hub[hf_xet]
python3 -m vllm.entrypoints.openai.api_server \
{{- range $key, $value := .Values.vllm_engine_args }}
{{- if eq $value nil }}
--{{ $key }} \
{{- else }}
--{{ $key }}={{ tpl $value $ | quote }} \
{{- end }}
{{- end }}
--model={{ $modelPath }} \
--tensor-parallel-size={{ .Values.gpus }} \
--host="0.0.0.0" \
--port=8080
{{- end -}}

{{ define "entrypoint" -}}
{{- if (hasPrefix "s3://" .Values.model) -}}
{{- include "minio.setup" . -}}
{{- end }}
{{ include "vllm.start" . -}}
{{- end -}}
