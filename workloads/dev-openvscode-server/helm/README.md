# dev-openvscode-server Helm Chart

This Helm chart is used to deploy the `dev-openvscode-server` application.

## Installation

To install the chart with the release name `my-release`:

```sh
helm template my-release ./helm | kubectl apply -f -
```

## Configuration

The following table lists the configurable parameters of the `dev-openvscode-server` chart and their default values.

| Parameter                                      | Description                                                                 | Default                                                                                       |
|------------------------------------------------|-----------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| `image`                                        | Image repository and tag                                                    | `ghcr.io/silogen/openvscode-server:vllm-dev-20250124`                                         |
| `imagePullPolicy`                              | Image pull policy                                                           | `Always`                                                                                      |
| `env_vars.BUCKET_STORAGE_HOST`                 | Bucket storage host URL                                                     | `https://default-minio-tenant-hl.minio-tenant-default.svc.cluster.local:9000`                 |
| `env_vars.BUCKET_STORAGE_ACCESS_KEY.name`      | Name of the secret containing the bucket storage access key                 | `minio-credentials`                                                                           |
| `env_vars.BUCKET_STORAGE_ACCESS_KEY.key`       | Key of the bucket storage access key in the secret                          | `minio-access-key`                                                                            |
| `env_vars.BUCKET_STORAGE_SECRET_KEY.name`      | Name of the secret containing the bucket storage secret key                 | `minio-credentials`                                                                           |
| `env_vars.BUCKET_STORAGE_SECRET_KEY.key`       | Key of the bucket storage secret key in the secret                          | `minio-secret-key`                                                                            |
| `storage.ephemeral.quantity`                   | Ephemeral storage quantity                                                  | `100Gi`                                                                                       |
| `storage.ephemeral.storageClassName`           | Ephemeral storage class name                                                | `mlstorage`                                                                                   |
| `storage.ephemeral.accessModes`                | Ephemeral storage access modes                                              | `[ "ReadWriteOnce" ]`                                                                         |
| `storage.dshm.sizeLimit`                       | Size limit for /dev/shm                                                     | `32Gi`                                                                                        |
| `deployment.port`                              | Port for the service                                                        | `3000`                                                                                        |
| `gpus`                                         | Number of GPUs                                                              | `1`                                                                                           |
| `memory_per_gpu`                               | Memory per GPU                                                              | `64Gi`                                                                                        |
| `cpu_per_gpu`                                  | CPU per GPU                                                                 | `4`                                                                                           |
| `entrypoint`                                   | Entrypoint script                                                           | `bash /workload/mount/install_kubectl.sh \ bash /workload/mount/install_helm.sh \ mc alias set minio-host ${BUCKET_STORAGE_HOST} ${BUCKET_STORAGE_ACCESS_KEY} ${BUCKET_STORAGE_SECRET_KEY} \ exec ${OPENVSCODE_SERVER_ROOT}/bin/openvscode-server --host 0.0.0.0 --without-connection-token` |

For more details see `values.yaml` file.

## Example

To deploy the chart with custom values file `values_override.yaml`:

```sh
helm template my-release ./helm -f values_override.yaml | kubectl apply -f -
```
