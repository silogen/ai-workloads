{{- define "namespace-setup.roles" }}
{{- if .Values.roles.enabled }}
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: default-reader
rules:
- apiGroups: [""]
  resources: ["pods", "services"]
  verbs: ["get", "list"]
- apiGroups: ["batch"]
  resources: ["jobs"]
  verbs: ["create", "delete", "get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: default-reader-binding
subjects:
- kind: ServiceAccount
  name: default
roleRef:
  kind: Role
  name: default-reader
  apiGroup: rbac.authorization.k8s.io
---
{{- end }}
{{- end }}
