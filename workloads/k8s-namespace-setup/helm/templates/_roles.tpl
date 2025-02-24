{{- define "namespace-setup.roles" }}
{{- if .Values.roles.enabled }}
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: default-role
rules:
- apiGroups: ["", "apps", "batch"]
  resources: ["*"]
  verbs: ["*"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: default-role-binding
subjects:
- kind: ServiceAccount
  name: default
roleRef:
  kind: Role
  name: default-role
  apiGroup: rbac.authorization.k8s.io
---
{{- end }}
{{- end }}
