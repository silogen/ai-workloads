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

# Container environment variables helper
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

# Container volume mounts helper
{{- define "container.volumeMounts" -}}
- mountPath: {{ .Values.settings.mountdir }}
  name: workload-mount
- mountPath: /dev/shm
  name: dshm
- mountPath: /workload
  name: ephemeral-storage
{{- end -}}

# Container volumes helper
{{- define "container.volumes" -}}
{{- if .Values.storage.ephemeral.storageClassName -}}
- ephemeral:
    volumeClaimTemplate:
      spec:
        {{- if .Values.storage.ephemeral.accessModes }}
        accessModes: {{ .Values.storage.ephemeral.accessModes }}
        {{- else }}
        accessModes:
          - ReadWriteOnce
        {{- end }}
        resources:
          requests:
            storage: {{ .Values.storage.ephemeral.quantity }}
        storageClassName: {{ .Values.storage.ephemeral.storageClassName }}
  name: ephemeral-storage
{{- else }}
- emptyDir: {}
  name: ephemeral-storage
  sizeLimit: {{ .Values.storage.ephemeral.quantity }}
{{- end }}
- emptyDir:
    medium: Memory
    sizeLimit: {{ .Values.storage.dshm.sizeLimit }}
  name: dshm
- configMap:
    name: {{ include "release.fullname" . }}
  name: workload-mount
{{- end -}}
