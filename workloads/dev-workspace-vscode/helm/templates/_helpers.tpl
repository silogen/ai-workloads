# Base URL helper
{{- define "httpRoute.baseUrl" -}}
{{- $projectId := default "project_id" .Values.metadata.project_id -}}
{{- $userId := default "user_id" .Values.metadata.user_id -}}
{{- $workloadId := default (include "release.fullname" .) .Values.metadata.workload_id -}}
{{- printf "/%s/%s/%s" $projectId $userId $workloadId }}
{{- end -}}

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
{{- /*
  When gpus=0 (CPU-only workspace), multiplying by gpus would produce 0, causing
  the max() floor to always win and ignoring the configured per-gpu values.
  Instead, use the per-gpu values directly as the flat allocation for CPU-only workloads.
*/ -}}
{{- $memory := ternary .Values.memory_per_gpu (mul .Values.gpus .Values.memory_per_gpu) (eq (int .Values.gpus) 0) -}}
{{- $cpu := ternary .Values.cpu_per_gpu (mul .Values.gpus .Values.cpu_per_gpu) (eq (int .Values.gpus) 0) -}}
requests:
  memory: "{{ max $memory 4 }}Gi"
  cpu: "{{ max $cpu 1 }}"
  {{- if .Values.gpus }}
  amd.com/gpu: "{{ .Values.gpus }}"
  {{- end }}
limits:
  memory: "{{ max $memory 4 }}Gi"
  cpu: "{{ max $cpu 1 }}"
  {{- if .Values.gpus }}
  amd.com/gpu: "{{ .Values.gpus }}"
  {{- end }}
{{- end -}}

# Container environment variables helper
{{- define "container.env" -}}
{{- range $key, $value := .Values.env_vars }}
{{- if (typeIs "string" $value) }}
- name: {{ $key }}
  value: {{ tpl $value $ | quote }}
{{- else }}
- name: {{ $key }}
  valueFrom:
    secretKeyRef:
      name: {{ $value.name }}
      key: {{ $value.key }}
{{- end }}
{{- end }}
- name: BASE_URL
  value: {{ include "httpRoute.baseUrl" . | quote }}
{{- end -}}

# Container volume mounts helper
{{- define "container.volumeMounts" -}}
- mountPath: /workload
  name: ephemeral-storage
- mountPath: /workload/mount
  name: workload-mount
- mountPath: /dev/shm
  name: dshm
{{- if .Values.persistent_storage.enabled }}
{{- range $key, $value := .Values.persistent_storage.volumes }}
- mountPath: {{ tpl $value.mount_path $ }}
  name: {{ $key }}
{{- end }}
{{- end }}
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
{{- if .Values.persistent_storage.enabled }}
{{- range $key, $value := .Values.persistent_storage.volumes }}
- persistentVolumeClaim:
    claimName: {{ tpl $value.pvc_name $ }}
  name: {{ $key }}
{{- end }}
{{- end }}
{{- end -}}
