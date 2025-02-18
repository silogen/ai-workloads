{{- define "release.name" -}}
  {{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "release.fullname" -}}
  {{- $currentTime := now | date "20060102-1504" -}}
  {{- if .Values.fullnameOverride -}}
    {{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
  {{- else -}}
    {{- if ne .Release.Name "release-name" -}}
      {{- include "release.name" . }}-{{ .Release.Name | trunc 63 | trimSuffix "-" -}}
    {{- else -}}
      {{- include "release.name" . }}-{{ $currentTime | lower | trunc 63 | trimSuffix "-" -}}
    {{- end -}}
  {{- end -}}
{{- end -}}

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
  mc alias set minio-host ${BUCKET_STORAGE_HOST} ${BUCKET_STORAGE_ACCESS_KEY} ${BUCKET_STORAGE_SECRET_KEY}
  echo '--------------------------------------------'
  echo 'Downloading the model to the local container'
  echo '--------------------------------------------'
  mc cp --recursive {{$minioModel -}}/ {{$localModel -}}/
{{- end -}}

{{- define "vllm.start" -}}
  {{- $modelPath := .Values.model -}}
  {{- if (hasPrefix "s3://" .Values.model) -}}
    {{- $modelPath := printf "/workload/%s" (trimPrefix "s3://" .Values.model | trimSuffix "/") -}}
  {{- end -}}
  echo '--------------------------------------------'
  echo 'Starting vLLM'
  echo '--------------------------------------------'
  python3 -m vllm.entrypoints.openai.api_server \
  {{- range $key, $value := .Values.vllm_engine_args }}
  --{{ $key }}={{ tpl $value $ | quote }} \
  {{- end }}
  --model={{ $modelPath }} \
  --tensor-parallel-size={{ .Values.gpus }} \
  --host="0.0.0.0" \
  --port=8080
{{- end -}}

{{- define "entrypoint" -}}
  {{- if (hasPrefix "s3://" .Values.model) -}}
    {{- include "minio.setup" . -}}
  {{- end }}
  {{ include "vllm.start" . -}}
{{- end -}}