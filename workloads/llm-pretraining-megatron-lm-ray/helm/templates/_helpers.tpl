# Release name helper
{{- define "release.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

# Release fullname helper
{{- define "release.fullname" -}}
{{- $releaseName := include "release.name" . -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- if ne .Release.Name "release-name" -}}
{{- printf "%s-job-%s" $releaseName .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- if not (index .Values "releaseSuffix") -}}
{{- /* ... store random release suffix under the .Values.releaseSuffix */ -}}
{{-   $_ := set .Values "releaseSuffix" (randAlphaNum 5 | lower) -}}
{{- end -}}
{{- printf "%s-job-%s" $releaseName .Values.releaseSuffix | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

# Container environment variables helper
{{- define "logistics.container.env" -}}
{{- range $key, $value := .Values.logistics.envVars }}
{{- if (typeIs "string" $value) }}
- name: {{ $key }}
  value: {{ $value | quote }}
{{- else }}
- name: {{ $key }}
  valueFrom:
    secretKeyRef:
      name: {{ $value.name }}
      key: {{ $value.key }}
{{- end }}
{{- end }}
{{- end -}}

# Container environment variables helper
{{- define "worker.container.env" -}}
{{- range $key, $value := .Values.workerEnvVars }}
{{- if (typeIs "string" $value) }}
- name: {{ $key }}
  value: {{ $value | quote }}
{{- else }}
- name: {{ $key }}
  valueFrom:
    secretKeyRef:
      name: {{ $value.name }}
      key: {{ $value.key }}
{{- end }}
{{- end }}
{{- end -}}

# Container volumes helper
{{- define "container.volumes" -}}
{{- if .Values.storage.ephemeral.storageClassName -}}
- name: ephemeral-storage
  persistentVolumeClaim:
    claimName: {{ include "release.fullname" . }}
{{- end }}
- emptyDir:
    medium: Memory
    sizeLimit: {{ .Values.storage.dshm.sizeLimit }}
  name: dshm
- configMap:
    name: {{ include "release.fullname" . }}
  name: workload-mount
{{- end -}}

# Entrypoint helper
{{- define "entrypoint" -}}
    {{- $extraArgs := list -}}
    {{- $isDataPath := false -}}
    {{- $isLoad := false -}}
    {{- range $index, $arg := .Values.megatronArguments -}}
      {{- if hasPrefix "--data-path" $arg -}}
        {{- $isDataPath = true -}}
      {{- else if hasPrefix "--load " $arg -}}
        {{- $isLoad = true -}}
      {{- end -}}
    {{- end -}}
    {{- /* Add --data-path if not set */ -}}
    {{- if and .Values.remoteDataDirPath .Values.remoteDataNamePrefix -}}
      {{- if not $isDataPath -}}
        {{- $extraArgs = append $extraArgs (printf "--data-path /local_resources/data/%s" .Values.remoteDataNamePrefix) -}}
      {{- end -}}
    {{- else if not (has "--mock-data" .Values.megatronArguments) -}}
      {{- $extraArgs = append $extraArgs "--mock-data" -}}
    {{- end -}}
    {{- /* Add --load if not set */ -}}
    {{- if .Values.remoteBaseModelPath -}}
      {{- if not $isLoad -}}
        {{- $extraArgs = append $extraArgs "--load /local_resources/basemodel" -}}
      {{- end -}}
    {{- else }}
      {{- if $isLoad -}}
        {{- fail ".Values.remoteBaseModelPath should be set when using --load" }}
      {{- end -}}
    {{- end -}}
    {{- range $index, $arg := .Values.megatronArguments -}}
      {{- if hasPrefix "--data-path" $arg -}}
        {{- if and $.Values.remoteDataDirPath $.Values.remoteDataNamePrefix -}}
          {{- $extraArgs = append $extraArgs (tpl $arg $) -}}
        {{- end -}}
      {{- else if hasPrefix "--load " $arg -}}
        {{- if $.Values.remoteBaseModelPath -}}
          {{- $extraArgs = append $extraArgs $arg -}}
        {{- end -}}
      {{- /* --use-checkpoint-args requires --load argument */ -}}
      {{- else if hasPrefix "--use-checkpoint-args" $arg -}}
        {{- if $isLoad -}}
          {{- $extraArgs = append $extraArgs $arg -}}
        {{- end -}}
      {{- else }}
        {{- $extraArgs = append $extraArgs $arg -}}
      {{- end -}}
    {{- end -}}
    python /local_resources/mount/ray_entrypoint.py --num-nodes {{ .Values.workers.replicas }} --gpus-per-node {{ .Values.workers.resources.gpu }}
    {{- range $index, $arg := $extraArgs }}
      {{- $arg | nindent 0 }}
    {{- end }}
{{- end -}}
