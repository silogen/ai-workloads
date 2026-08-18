# Container resources helper
{{- define "container.resources" -}}
requests:
  memory: "{{ max (mul .Values.gpus .Values.memoryPerGpu) 4 }}Gi"
  cpu: "{{ max (mul .Values.gpus .Values.cpusPerGpu) 1 }}"
  {{- if .Values.gpus }}
  amd.com/gpu: "{{ .Values.gpus }}"
  {{- end }}
limits:
  memory: "{{ max (mul .Values.gpus .Values.memoryPerGpu) 4 }}Gi"
  cpu: "{{ max (mul .Values.gpus .Values.cpusPerGpu) 1 }}"
  {{- if .Values.gpus }}
  amd.com/gpu: "{{ .Values.gpus }}"
  {{- end }}
{{- end -}}
