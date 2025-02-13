{{- define "dev-openvscode-server.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "dev-openvscode-server.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- if ne .Release.Name "release-name" -}}
{{- include "dev-openvscode-server.name" . }}-{{ .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- include "dev-openvscode-server.name" . }}-{{ randAlphaNum 5 | lower | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}
