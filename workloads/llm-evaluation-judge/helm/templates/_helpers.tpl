{/* ####################################################################################################################################################### */}}
{{- define "modelDownload" -}}
# Download the model depending on whether it is stored locally or on HuggingFace
echo 'Copying model to container...'
{{- $modelDownloadPath := .model_download_path -}}
{{- if (hasPrefix "s3://" .model_download_path) -}} # S3 Compatible Storage
{{- $modelDownloadPath := (trimPrefix "s3://" .model_download_path | trimSuffix "/") }}
echo 'S3 model download path.'
echo '--------------------------------------------'
echo 'Installing and setting up minio client'
echo '--------------------------------------------'
curl https://dl.min.io/client/mc/release/linux-amd64/mc \
      --create-dirs \
      -o /minio-binaries/mc
chmod +x /minio-binaries/mc
export PATH="${PATH}:/minio-binaries/"
{{/* We add http protocol to the host here, because other services expect the bare hostname (for example, the python minio client specifies http with secure=False instead) */}}
if [ "$(printf '%s\n' "$BUCKET_STORAGE_HOST" | cut -c1-7)" != "http://" ]; then
    export BUCKET_STORAGE_HOST="http://$BUCKET_STORAGE_HOST"
fi
mc alias set minio-host $${BUCKET_STORAGE_HOST} $${BUCKET_STORAGE_ACCESS_KEY} $${BUCKET_STORAGE_SECRET_KEY}
echo Model directory minio-host/{{ $modelDownloadPath | trimSuffix "/" }} with contents:
mc ls minio-host/{{ $modelDownloadPath | trimSuffix "/" }}
echo '--------------------------------------------'
echo 'Downloading the model to the local container'
echo '--------------------------------------------'
mc --debug cp --recursive \
  minio-host/{{ $modelDownloadPath | trimSuffix "/" }}/ \
  /local_models/judge_model
{{- else if hasPrefix "hf://" .model_download_path }} # HuggingFace
{{- $modelDownloadPath := (trimPrefix "hf://" .model_download_path | trimSuffix "/") }}
echo 'HuggingFace model path. Downloading from HuggingFace...'
huggingface-cli download {{ $modelDownloadPath }} --local-dir {{ .local_dir_path }};
{{- else }}
{{- fail (printf "Unknown model_path prefix in '%s'. Must start with 's3://' or 'hf://'." .model_download_path) }}
{{- end }}
{{- end }}
