{{/* Release name helper */}}
{{- define "release.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* Full release name with timestamp fallback */}}
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

{{/* Container resources helper (GPU, ephemeral storage) */}}
# Container resources helper
{{- define "container.resources" -}}
requests:
  {{- if .Values.resources.gpus }}
  amd.com/gpu: "{{ .Values.resources.gpus }}"
  {{- end }}
  cpu: "{{ max (mul .Values.resources.gpus .Values.resources.cpuPerGpu) 1 }}"
  memory: "{{ max (mul .Values.resources.gpus .Values.resources.memoryPerGpu) 4 }}Gi"
limits:
  {{- if .Values.resources.gpus }}
  amd.com/gpu: "{{ .Values.resources.gpus }}"
  {{- end }}
  cpu: "{{ max (mul .Values.resources.gpus .Values.resources.cpuPerGpu) 1 }}"
  memory: "{{ max (mul .Values.resources.gpus .Values.resources.memoryPerGpu) 4 }}Gi"
{{- end -}}

{{/* Optional container environment variables from values.env_vars */}}
{{- define "container.env" -}}
{{- range $key, $value := .Values.env_vars }}
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

{{/* Volume mounts (config + shm) */}}
{{- define "container.volumeMounts" -}}
- name: workload-mount
  mountPath: /workload/mount
- name: workspace-volume
  mountPath: /workspace
- name: dshm
  mountPath: /dev/shm
{{- end -}}

{{/* Volumes (configmap + memory disk) */}}
{{- define "container.volumes" -}}
- name: dshm
  emptyDir:
    medium: Memory
    sizeLimit: {{ .Values.storage.dshm.sizeLimit }}
- name: workload-mount
  configMap:
    name: {{ include "release.fullname" . }}
- name: workspace-volume
  emptyDir: {}
{{- end -}}
