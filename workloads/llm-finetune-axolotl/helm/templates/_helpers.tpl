{/* ####################################################################################################################################################### */}}
{{- define "downloadEntrypoint" -}}
# Setup MinIO
mc alias set minio-host $${BUCKET_STORAGE_HOST} $${BUCKET_STORAGE_ACCESS_KEY} $${BUCKET_STORAGE_SECRET_KEY}
# Sync checkpoints from remote to local
{{- $checkpointsRemotePath := printf "minio-host/'%s'/" (.Values.checkpointsRemote | trimSuffix "/" | replace  "'" "'\\''") }}
{{- if .Values.checkpointsRemote }}
if mc mirror {{ $checkpointsRemotePath }} /workdir/checkpoints 2>/dev/null; then
  echo 'Downloaded checkpoints from {{ .Values.checkpointsRemote | trimSuffix "/" | replace  "'" "'\\''"}} to /workdir/checkpoints'
  ls -lah /workdir/checkpoints
else
  echo 'No checkpoints found yet'
fi
{{- end }}
{{- end }}

{/* ####################################################################################################################################################### */}}
{{- define "finetuningAndUploadEntrypoint" -}}
# Print GPU Info:
rocm-smi
{{- $checkpointsRemotePath := printf "minio-host/'%s'/" (.Values.checkpointsRemote | trimSuffix "/" | replace  "'" "'\\''") }}
{{- if .Values.checkpointsRemote }}
echo "Starting checkpoint sync process"
mc mirror \
  --watch \
  /workdir/checkpoints \
  {{ $checkpointsRemotePath }} &
uploadPID=$!
{{- end }}
# Run training:
echo "Starting training process"
axolotl train --output-dir /workdir/checkpoints /configs/{{ .Values.configFile }}
{{- if .Values.checkpointsRemote }}
echo "Training done, stop the upload process"
kill $uploadPID
wait $uploadPID || true
# Once more to ensure everything gets uploaded
echo 'Training done, syncing once more...'
mc mirror \
  /workdir/checkpoints \
  {{ $checkpointsRemotePath }}
{{- end }}
echo 'All done, exiting'
{{- end }}
