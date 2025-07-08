{{- define "podspec" -}}
spec:
  restartPolicy: Never
  {{- if .Values.imagePullSecrets }}
  imagePullSecrets:
    {{- range .Values.imagePullSecrets }}
    - name: {{ . }}
    {{- end }}
  {{- end }}
  containers:
  - image: {{ .Values.finetuningImage | quote}}
    imagePullPolicy: Always
    name: {{ .Name }}
    env:
      {{- if .Values.hfTokenSecret }}
      - name: HF_TOKEN
        valueFrom:
          secretKeyRef:
            name: {{ .Values.hfTokenSecret.name }}
            key: {{ .Values.hfTokenSecret.key }}
      {{- end }}
      # storage
      - name: BUCKET_STORAGE_HOST
        value: {{ .Values.bucketStorageHost }}
      - name: BUCKET_STORAGE_ACCESS_KEY
        valueFrom:
          secretKeyRef:
            name: {{ .Values.bucketCredentialsSecret.name }}
            key: {{ .Values.bucketCredentialsSecret.accessKeyKey }}
      - name: BUCKET_STORAGE_SECRET_KEY
        valueFrom:
          secretKeyRef:
            name: {{ .Values.bucketCredentialsSecret.name }}
            key: {{ .Values.bucketCredentialsSecret.secretKeyKey }}
    {{- if .Command }}
    command: {{ .Command }}
    {{- end }}
    {{- if .Args }}
    args: {{ .Args }}
    {{- end }}
    resources:
      limits:
        cpu: {{ .Cpu }}
        memory: {{ .Memory }}
        amd.com/gpu: {{ .Gpu }}
      requests:
        cpu: {{ .Cpu }}
        memory: {{ .Memory }}
        amd.com/gpu: {{ .Gpu }}
    volumeMounts:
      - name: dshm # Increase SHM size for the container by mounting /dev/shm, for Pytorch parallel processing
        mountPath: /dev/shm
      - name: checkpoints
        mountPath: /workdir
        readOnly: false
      - name: configs
        mountPath: /configs
        readOnly: true
  volumes:
    - name: dshm
      emptyDir:
        medium: Memory # equivalent to `docker run --shm-size=(total_memory/2)`
    {{- if ne (int $.Values.nodes) 1 }}
    - name: checkpoints
      persistentVolumeClaim:
        claimName: "{{ .Release.Name }}-pvc"
    {{- else if .Values.storageClass }}
    - name: checkpoints
      ephemeral:
        volumeClaimTemplate:
          spec:
            accessModes: [ "ReadWriteOnce" ]
            storageClassName: {{ .Values.storageClass }}
            resources:
              requests:
                storage: "{{ .Values.checkpointsReservedSize }}"
    {{- else }}
    - name: checkpoints
      emptyDir:
        sizeLimit: "{{ .Values.checkpointsReservedSize }}"
    {{- end }}
    - name: configs
      configMap:
        name: "{{ .Release.Name }}-configs"
        items:
        - key: entrypoint.sh
          path: entrypoint.sh
          mode: 0777
        - key: llama_factory_config.yaml
          path: llama_factory_config.yaml
        {{- if .Values.datasetInfo }}
        - key: remote_dataset_info.json
          path: remote_dataset_info.json
        {{- end }}
{{- end }}
