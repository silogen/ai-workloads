# Helper templates.

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

# Image pull secrets helper.
{{- define "image.pull_secrets" -}}
{{- if .Values.imagePullSecrets }}
imagePullSecrets:
  {{- range $index, $map:= .Values.imagePullSecrets }}
  {{- range $key, $value:= $map }}
  - {{ $key }}: {{ $value }}
  {{- end -}}
  {{- end -}}
{{- end -}}
{{- end -}}

# Container resources helper.
{{- define "container.resources" -}}
requests:
  {{- if .Values.resources.cpu }}
  cpu: "{{ .Values.resources.cpu }}"
  {{- end }}
  {{- if .Values.resources.memory }}
  memory: "{{ .Values.resources.memory }}"
  {{- end }}
  {{- if .Values.resources.gpus }}
  amd.com/gpu: "{{ .Values.resources.gpus }}"
  {{- end }}
  {{- if .Values.resources.ephemeral_storage }}
  ephemeral-storage: "{{ .Values.resources.ephemeral_storage.request }}"
  {{- end }}
limits:
  {{- if .Values.resources.cpu }}
  cpu: "{{ .Values.resources.cpu }}"
  {{- end }}
  {{- if .Values.resources.memory }}
  memory: "{{ .Values.resources.memory }}"
  {{- end }}
  {{- if .Values.resources.gpus }}
  amd.com/gpu: "{{ .Values.resources.gpus }}"
  {{- end }}
  {{- if .Values.resources.ephemeral_storage }}
  ephemeral-storage: "{{ .Values.resources.ephemeral_storage.limit }}"
  {{- end }}
{{- end }}


# Container environment variables helper.
{{- define "container.env" -}}
{{- range $key, $value := .Values.env_vars }}
- name: {{ $key }}
  value: {{ $value | quote }}
{{- end -}}
{{- end -}}

# Container secrets helper.
{{- define "container.env_secrets" -}}
{{- range $key, $value := .Values.env_secrets }}
- name: {{ $key }}
  valueFrom:
    secretKeyRef:
      name: {{ $value.name }}
      key: {{ $value.key }}
{{- end -}}
{{- end -}}
