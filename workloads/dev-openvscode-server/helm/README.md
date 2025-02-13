# dev-openvscode-server Helm Chart

This Helm chart is used to deploy the `dev-openvscode-server` application.

## Installation

To install the chart with the release name `my-release`:

```sh
helm template my-release ./helm | kubectl apply -f -
```

## Configuration

The following table lists the configurable parameters of the `dev-openvscode-server` chart and their default values.

| Parameter                          | Description                                            | Default                                                    |
|------------------------------------|--------------------------------------------------------|------------------------------------------------------------|
| `image.repository`                 | Image repository                                       | `ghcr.io/silogen/openvscode-server`                        |
| `image.tag`                        | Image tag                                              | `vllm-dev-20250124`                                        |
| `image.pullPolicy`                 | Image pull policy                                      | `Always`                                                   |
| `namespace`                        | Namespace for the service                              | `kaiwo`                                                    |
| `labels`                           | Labels for the deployment                              | `[ { label: "kaiwo.silogen.ai/managed", value: "true" } ]` |
| `envVars.bucket_storage_host`      | Bucket storage host URL                                | `https://default-minio-tenant-hl.minio-tenant-default.svc.cluster.local:9000` |
| `envVars.bucket_storage_access_key_env_spec` | Bucket storage access key environment variable specification | `{ valueFrom: { secretKeyRef: { name: "minio-credentials", key: "minio-access-key" } } }` |
| `envVars.bucket_storage_secret_key_env_spec` | Bucket storage secret key environment variable specification | `{ valueFrom: { secretKeyRef: { name: "minio-credentials", key: "minio-secret-key" } } }` |
| `envVars.hf_token_env_spec`        | Hugging Face token environment variable specification  | `{ valueFrom: { secretKeyRef: { name: "hf-token", key: "hf-token" } } }` |
| `storage.quantity`                 | Storage quantity                                       | `100Gi`                                                    |
| `storage.storageClassName`         | Storage class name                                     | `longhorn`                                                 |
| `dshm.sizeLimit`                   | Size limit for /dev/shm                                | `200Gi`                                                    |

## Example

To deploy the chart with custom values file `values_override.yaml`:

```sh
helm template my-release ./helm -f values_override.yaml | kubectl apply -f -
```
