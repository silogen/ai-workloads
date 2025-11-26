# Release name helper
{{- define "release.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

# Release fullname helper
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

# Container resources helper
{{- define "container.resources" -}}
requests:
  {{- if .Values.gpus }}
  amd.com/gpu: "{{ .Values.gpus }}"
  {{- end }}
  {{- if .Values.ephemeral_storage }}
  ephemeral-storage: "{{ .Values.ephemeral_storage }}"
  {{- end }}
  {{- with .Values.job.resources.cpu }}
  {{- if .requests }}
  cpu: {{ .requests | quote }}
  {{- end }}
  {{- end }}
  {{- with .Values.job.resources.memory }}
  {{- if .requests }}
  memory: {{ .requests | quote }}
  {{- end }}
  {{- end }}
limits:
  {{- if .Values.gpus }}
  amd.com/gpu: "{{ .Values.gpus }}"
  {{- end }}
  {{- if .Values.ephemeral_storage }}
  ephemeral-storage: "{{ .Values.ephemeral_storage }}"
  {{- end }}
  {{- with .Values.job.resources.cpu }}
  {{- if .limits }}
  cpu: {{ .limits | quote }}
  {{- end }}
  {{- end }}
  {{- with .Values.job.resources.memory }}
  {{- if .limits }}
  memory: {{ .limits | quote }}
  {{- end }}
  {{- end }}
{{- end -}}

# Container environment variables helper
{{- define "container.env" -}}
{{- range $key, $value := .Values.env_vars }}
{{- if (kindIs "map" $value) }}
- name: {{ $key }}
  valueFrom:
    secretKeyRef:
      name: {{ $value.name }}
      key: {{ $value.key }}
{{- else }}
- name: {{ $key }}
  value: {{ $value | quote }}
{{- end }}
{{- end }}
{{- if .Values.training.args }}
{{- range $key, $value := .Values.training.args }}
- name: "TRAINING_ARG_{{ $key | upper | replace "-" "_" }}"
  value: {{ $value | quote }}
{{- end }}
{{- end }}
{{- end -}}

# Container volume mounts helper
{{- define "container.volumeMounts" -}}
- mountPath: /workload/mount
  name: workload-mount
- name: workload-outputs
  mountPath: /workload_outputs
- mountPath: /dev/shm
  name: dshm
{{- range .Values.volumes }}
- name: {{ .name }}
  mountPath: {{ .mountPath }}
  readOnly: true
{{- end }}
{{- end -}}

# Container volumes helper
{{- define "container.volumes" -}}
- name: dshm
  emptyDir:
    medium: Memory
    sizeLimit: {{ .Values.storage.dshm.sizeLimit }}
- configMap:
    name: {{ include "release.fullname" . }}
    defaultMode: 0755
  name: workload-mount
- name: workload-outputs
  emptyDir:
    sizeLimit: {{ .Values.storage.workload_outputs.sizeLimit }}

{{- range .Values.volumes }}
- name: {{ .name }}
  {{- if .secret }}
  secret:
    secretName: {{ .secret.secretName }}
  {{- end }}
{{- end }}
{{- end -}}
