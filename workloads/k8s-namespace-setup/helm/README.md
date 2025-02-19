# k8s-namespace-setup

A Helm chart for Kubernetes namespace setup.

## Installation

Given a particular configuration for your namespace, e.g. `overrides/rename-secret-names.yaml` , you can apply it to active namespace using command

```sh
helm template . -f overrides/rename-secret-names.yaml | kubectl apply -f -
```

Optionally, you can specify another namespace on the command-line:
Given a particular configuration for your namespace, you can apply it using command

```sh
helm template . -f overrides/rename-secret-names.yaml | kubectl apply -n <NAMESPACE> -f -
```

## Configuration

The following table lists the configurable parameters of the `k8s-namespace-setup` chart and their default values.

### External Secret

| Parameter                                      | Description                                      | Default                        |
|------------------------------------------------|--------------------------------------------------|--------------------------------|
| `external_secret.enabled` | Enable external secret                           | `false` |
| `external_secret.src.secret_store_name` | Secret store name                                | `k8s-secret-store` |
| `external_secret.src.remote_secret_name` | Remote secret name                               | `default-user` |
| `external_secret.src.access_key_name` | Access key name                                  | `minio-access-key` |
| `external_secret.src.secret_key_name` | Secret key name                                  | `minio-secret-key` |
| `external_secret.dest.k8s_secret_name` | Kubernetes secret name                           | `minio-credentials` |
| `external_secret.dest.access_key_name` | Kubernetes access key name                       | `minio-access-key` |
| `external_secret.dest.secret_key_name` | Kubernetes secret key name                       | `minio-secret-key` |

### Kueue

| Parameter                                      | Description                                      | Default                        |
|------------------------------------------------|--------------------------------------------------|--------------------------------|
| `kueue.enabled` | Enable kueue                                     | `false` |
| `kueue.local_queue_name` | Local queue name                                 | `kaiwo` |
