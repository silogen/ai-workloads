{{/*
bucketStorageClient selects the S3 client baked into the container script.
"mc" keeps the MinIO client, including `mc mirror --watch`.
"aws" uses the AWS CLI. Watch-style uploads are a loop of `aws s3 sync`.
*/}}
{{- define "bucketStorageClientName" -}}
{{- $client := default "mc" .Values.bucketStorageClient -}}
{{- if not (or (eq $client "mc") (eq $client "aws")) -}}
{{- fail (printf "bucketStorageClient must be \"mc\" or \"aws\", got %q" $client) -}}
{{- end -}}
{{- $client -}}
{{- end -}}

{{/*
Remote directory URI for the selected client.
dict: client, path, suffix (optional path appended after the directory slash).
mc keeps the historical minio-host/'path'/ form so existing scripts stay stable.
*/}}
{{- define "bucketDir" -}}
{{- $path := index . "path" | trimSuffix "/" | replace "'" "'\\''" -}}
{{- $suffix := default "" (index . "suffix") -}}
{{- if eq (index . "client") "aws" -}}
's3://{{ $path }}/{{ $suffix }}'
{{- else -}}
minio-host/'{{ $path }}'/{{ $suffix }}
{{- end -}}
{{- end -}}

{{/* Remote object URI. dict: client, path. */}}
{{- define "bucketObject" -}}
{{- $path := index . "path" | replace "'" "'\\''" -}}
{{- if eq (index . "client") "aws" -}}
's3://{{ $path }}'
{{- else -}}
minio-host/'{{ $path }}'
{{- end -}}
{{- end -}}

{{/*
AWS CLI setup. Credentials stay on the aws process and are not exported into the
training process. `$$` is required so the kubelet reduces it to a single `$`
before bash runs.

SeaweedFS verifies SigV4 against a host with no default port, and it does not
implement the flexible checksums or HTTP payload signing the AWS CLI adds on
http:// endpoints. Checksums stay off, payload signing stays off, and :80/:443
are removed from the endpoint so the signed Host matches what SeaweedFS checks.
*/}}
{{- define "awsS3ClientSetup" -}}
aws_s3() {
  local endpoint="$${BUCKET_STORAGE_HOST}"
  case "$$endpoint" in
    http://*:80) endpoint="$${endpoint%:80}" ;;
    https://*:443) endpoint="$${endpoint%:443}" ;;
  esac
  AWS_ACCESS_KEY_ID="$${BUCKET_STORAGE_ACCESS_KEY}" \
  AWS_SECRET_ACCESS_KEY="$${BUCKET_STORAGE_SECRET_KEY}" \
  AWS_DEFAULT_REGION="$${AWS_DEFAULT_REGION:-us-east-1}" \
  AWS_REQUEST_CHECKSUM_CALCULATION="$${AWS_REQUEST_CHECKSUM_CALCULATION:-when_required}" \
  AWS_RESPONSE_CHECKSUM_VALIDATION="$${AWS_RESPONSE_CHECKSUM_VALIDATION:-when_required}" \
  AWS_EC2_METADATA_DISABLED=true \
  AWS_CONFIG_FILE=/tmp/bucket-storage-aws-config \
  aws --endpoint-url "$$endpoint" s3 "$@"
}
# aws s3 has no mirror --watch. Re-sync until this process is killed.
# Sleep runs in the background so SIGTERM during the wait stops the loop promptly;
# an in-flight sync is allowed to finish before the trap exits.
bucket_sync_watch() {
  local src="$1"
  local dst="$2"
  local sleeper=0
  trap 'if [ "$sleeper" -ne 0 ]; then kill "$sleeper" 2>/dev/null || true; fi; exit 0' TERM INT
  while true; do
    sleep "$${BUCKET_STORAGE_SYNC_INTERVAL:-30}" &
    sleeper=$!
    wait "$sleeper" || exit 0
    sleeper=0
    aws_s3 sync --only-show-errors "$src" "$dst" || echo "WARNING: bucket sync of $src failed; retrying"
  done
}
if ! command -v aws >/dev/null 2>&1; then
  echo "ERROR: bucketStorageClient is aws, but the aws CLI is not installed in this image"
  exit 1
fi
mkdir -p /tmp
printf '%s\n' \
  '[default]' \
  "region = $${AWS_DEFAULT_REGION:-us-east-1}" \
  'request_checksum_calculation = when_required' \
  'response_checksum_validation = when_required' \
  's3 =' \
  '    addressing_style = path' \
  '    payload_signing_enabled = false' \
  > /tmp/bucket-storage-aws-config
{{ end }}

{/* ####################################################################################################################################################### */}}
{{- define "downloadEntrypoint" -}}
{{- $client := include "bucketStorageClientName" . | trim -}}
{{- if eq $client "aws" -}}
# Setup AWS CLI and download resources:
{{- else -}}
# Setup MinIO, Download resources:
{{- end }}
echo 'Copying resources to container...';
{{- if eq $client "aws" }}
{{ include "awsS3ClientSetup" . }}
{{- else }}
mc alias set minio-host $${BUCKET_STORAGE_HOST} $${BUCKET_STORAGE_ACCESS_KEY} $${BUCKET_STORAGE_SECRET_KEY}
{{- end }}
{{- if (hasPrefix "hf://" .Values.basemodel) }}
hf_binary=$(which hf 2>/dev/null || which huggingface-cli 2>/dev/null)
$hf_binary download '{{ trimPrefix "hf://" .Values.basemodel }}' \
  {{- if .Values.hfDownloadExcludeGlob }}
  --exclude '{{ .Values.hfDownloadExcludeGlob }}' \
  {{- end }}
  --local-dir /local_resources/basemodel
{{- else }}
{{- if eq $client "aws" }}
aws_s3 cp --recursive \
  {{ include "bucketDir" (dict "client" "aws" "path" .Values.basemodel "suffix" "") }} \
  /local_resources/basemodel
{{- else }}
mc cp --recursive \
  minio-host/'{{ .Values.basemodel | trimSuffix "/" }}'/ \
  /local_resources/basemodel
{{- end }}
{{- end }}
{{- if $.Values.trainingData }}
{{- if eq $client "aws" }}
aws_s3 cp \
  {{ include "bucketObject" (dict "client" "aws" "path" $.Values.trainingData) }} \
  /local_resources/'{{ $.Values.trainingData | replace  "'" "'\\''" | replace "/" "_" }}'
{{- else }}
mc cp \
  minio-host/'{{ $.Values.trainingData | replace  "'" "'\\''" }}' \
  /local_resources/'{{ $.Values.trainingData | replace  "'" "'\\''" | replace "/" "_" }}'
{{- end }}
{{- else }}
{{- range .Values.finetuning_config.data_conf.training_data.datasets }}
{{- if eq $client "aws" }}
aws_s3 cp \
  {{ include "bucketObject" (dict "client" "aws" "path" .path) }} \
  /local_resources/'{{ .path | replace  "'" "'\\''" | replace "/" "_" }}'
{{- else }}
mc cp \
  minio-host/'{{ .path | replace  "'" "'\\''" }}' \
  /local_resources/'{{ .path | replace  "'" "'\\''" | replace "/" "_" }}'
{{- end }}
{{- end }}
{{- if (or (eq .Values.finetuning_config.data_conf.validation_data.type "AUTO_SPLIT" ) (eq .Values.finetuning_config.data_conf.validation_data.type "NONE")) }}
{{- range .Values.finetuning_config.data_conf.validation_data.datasets }}
{{- if eq $client "aws" }}
aws_s3 cp \
  {{ include "bucketObject" (dict "client" "aws" "path" .path) }} \
  /local_resources/'{{ .path | replace  "'" "'\\''" | replace "/" "_" }}'
{{- else }}
mc cp \
  minio-host/'{{ .path | replace  "'" "'\\''" }}' \
  /local_resources/'{{ .path | replace  "'" "'\\''" | replace "/" "_" }}'
{{- end }}
{{- end }}
{{- end }}
{{- end }}
# Sync checkpoints from remote to local
{{- if not .Values.debug.skip_checkpoint_download }}
{{- $checkpointsRemotePath := include "bucketDir" (dict "client" $client "path" .Values.checkpointsRemote "suffix" "") }}
{{- if eq $client "aws" }}
if aws_s3 sync {{ $checkpointsRemotePath }} /workdir/checkpoints 2>/dev/null; then
{{- else }}
if mc mirror {{ $checkpointsRemotePath }} /workdir/checkpoints 2>/dev/null; then
{{- end }}
  echo 'Downloaded checkpoints from {{ .Values.checkpointsRemote | trimSuffix "/" | replace  "'" "'\\''" }} to /workdir/checkpoints'
  ls -lah /workdir/checkpoints
else
  echo 'No checkpoints found yet'
fi
{{- end }}
{{- end }}

{/* ####################################################################################################################################################### */}}
{{- define "finetuningAndUploadEntrypoint" -}}
# quote paths with single quotes to avoid issues with special characters in paths, and replace any existing single quote with escaped single quote
{{- $client := include "bucketStorageClientName" . | trim -}}
{{- $checkpointsRemote := .Values.checkpointsRemote | trimSuffix "/" -}}
{{- $logsRemote := (default (printf "%s/logs" $checkpointsRemote) .Values.logsRemote) | trimSuffix "/" -}}
{{- $checkpointsRemotePath := include "bucketDir" (dict "client" $client "path" $checkpointsRemote "suffix" "") -}}
{{- $logsRemotePath := include "bucketDir" (dict "client" $client "path" $logsRemote "suffix" "") -}}
{{- $adapterRemotePath := include "bucketDir" (dict "client" $client "path" $checkpointsRemote "suffix" "checkpoint-final-adapter/") -}}
# Print GPU Info:
rocm-smi
{{- if not .Values.debug.skip_checkpoint_upload }}
echo "Starting checkpoint sync process"
{{- if eq $client "aws" }}
if ! aws_s3 sync /workdir/checkpoints {{ $checkpointsRemotePath }}; then
  echo "ERROR: Sync process failed to start"
  exit 1
fi
bucket_sync_watch /workdir/checkpoints {{ $checkpointsRemotePath }} &
{{- else }}
mc mirror \
  --watch \
  /workdir/checkpoints \
  {{ $checkpointsRemotePath }} &
{{- end }}
uploadPID=$!
sleep 1 # Give some time for the process to start
# Check if the sync process started successfully
if ! ps -p $uploadPID > /dev/null; then
  echo "ERROR: Sync process failed to start"
  exit 1
fi
{{- end }}
# # Run training:
{{- if .Values.runTensorboard }}
tensorboard --logdir /workdir/logs --port 6006 &
echo "Serving tensorboard on port 6006. Port-forward to access training logs during the training process lifetime."
echo "Also starting logs upload process, uploading to {{ $logsRemotePath }}"
{{- if eq $client "aws" }}
mkdir -p /workdir/logs
if ! aws_s3 sync /workdir/logs {{ $logsRemotePath }}; then
  echo "ERROR: Logs sync process failed to start"
  exit 1
fi
bucket_sync_watch /workdir/logs {{ $logsRemotePath }} &
{{- else }}
mc mirror \
  --watch \
  /workdir/logs \
  {{ $logsRemotePath }} &
{{- end }}
logsPID=$!
sleep 1
if ! ps -p $logsPID > /dev/null; then
  echo "ERROR: Logs sync process failed to start"
  exit 1
fi
{{- end }}
echo "Starting training process"
accelerate launch \
  --config_file /configs/accelerate_config.yaml \
  --no-python \
  {{ .Values.mainBinaryName }} --num-preprocess-workers 4 {{ .Values.finetuning_config.method }} /configs/finetuning_config.yaml
{{- if not .Values.debug.skip_checkpoint_upload }}
echo "Training done, stop the upload process"
kill $uploadPID
wait $uploadPID || true
{{- end }}
{{- if .Values.runTensorboard }}
kill $logsPID
wait $logsPID || true
{{- end }}
{{- if (not (eq .Values.finetuning_config.peft_conf.peft_type "NO_PEFT")) }}
echo "Running adapter post-processing:"
create_vllm_compatible_adapter --training-config /configs/finetuning_config.yaml ./checkpoints/checkpoint-final-adapter
{{- if not .Values.debug.skip_checkpoint_upload }}
# Sync the final checkpoint with overwrite to carry over vLLM-compatibility changes
{{- if eq $client "aws" }}
# aws s3 sync can skip a rewritten file when size and mtime still match, so copy every object.
aws_s3 cp --recursive \
  /workdir/checkpoints/checkpoint-final-adapter \
  {{ $adapterRemotePath }}
{{- else }}
mc mirror \
  --overwrite \
  /workdir/checkpoints/checkpoint-final-adapter \
  {{ $adapterRemotePath }}
{{- end }}
{{- end }}
{{- if .Values.mergeAdapter }}
merge_base=/local_resources/basemodel
if [ -d ./checkpoints/checkpoint-new-basemodel ]; then
  merge_base=./checkpoints/checkpoint-new-basemodel
fi
merge_adapter $merge_base ./checkpoints/checkpoint-final-adapter ./checkpoints/checkpoint-final
{{- end }}
{{- end }}
{{- if .Values.aimManifest.modelManifest.enabled }}
# Copy AIMModel manifest to checkpoint directory for archival
echo 'Copying AIMModel manifest to checkpoint directory...'
cp /configs/aim-model-manifest.yaml /workdir/checkpoints/aim-model-manifest.yaml
{{- end }}
{{- if .Values.aimManifest.adapterManifest.enabled }}
# Copy AIMArtifact manifest to checkpoint directory for archival
echo 'Copying AIMArtifact manifest to checkpoint directory...'
cp /configs/adapter-manifest.yaml /workdir/checkpoints/checkpoint-final-adapter/adapter-manifest.yaml
cp /configs/adapter-cr-manifest.yaml /workdir/checkpoints/adapter-cr-manifest.yaml
{{- end }}
# Copy preprocessor config into final checkpoint if present in downloaded base model.
# This keeps deployment artifacts self-contained without a separate processor job.
if [ -f /local_resources/basemodel/preprocessor_config.json ]; then
  mkdir -p /workdir/checkpoints/checkpoint-final
  cp /local_resources/basemodel/preprocessor_config.json /workdir/checkpoints/checkpoint-final/preprocessor_config.json
  echo 'Copied preprocessor_config.json to checkpoint-final'
else
  echo 'No preprocessor_config.json found in basemodel, skipping copy'
fi
{{- if not .Values.debug.skip_checkpoint_upload }}
# Once more to ensure everything gets uploaded
echo 'Training done, syncing once more...'
{{- if eq $client "aws" }}
aws_s3 sync \
  /workdir/checkpoints \
  {{ $checkpointsRemotePath }}
{{- else }}
mc mirror \
  /workdir/checkpoints \
  {{ $checkpointsRemotePath }}
{{- end }}
{{- end }}
{{- if .Values.runTensorboard }}
{{- if eq $client "aws" }}
aws_s3 sync \
  /workdir/logs \
  {{ $logsRemotePath }}
{{- else }}
mc mirror \
  /workdir/logs \
  {{ $logsRemotePath }}
{{- end }}
{{- end }}
echo 'All done, exiting'
{{- end }}

{{/*
Validate a resource name based on DNS label names (RFC 1123).
https://kubernetes.io/docs/concepts/overview/working-with-objects/names/#dns-label-names
1. contain at most 63 characters
2. contain only lowercase alphanumeric characters or '-'
3. start with an alphanumeric character
4. end with an alphanumeric character
*/}}
{{- define "validateResourceName" -}}
{{- $name := . -}}
{{- if gt (len $name) 63 }}
{{- fail (printf "Resource name '%s' is too long (max 63 characters)" $name) -}}
{{- end -}}
{{- if regexMatch "[A-Z]" $name -}}
{{- fail (printf "Resource name '%s' contains uppercase letters. It must be lowercase." $name) -}}
{{- end -}}
{{- if not (regexMatch "^[a-z0-9]([-a-z0-9]*[a-z0-9])?$" $name) -}}
{{- fail (printf "Resource name '%s' is invalid. It must consist of lowercase alphanumeric characters (ASCII only) or '-', and must start and end with an alphanumeric character (regex: '^[a-z0-0]([-a-z0-0]*[a-z0-0])?$')" $name) -}}
{{- end -}}
{{- end -}}
