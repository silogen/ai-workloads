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
{{- range $key, $value := .Values.logistics.env_vars }}
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
{{- range $key, $value := .Values.worker_env_vars }}
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
