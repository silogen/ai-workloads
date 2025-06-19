# MLflow Tracking Server

This Helm Chart deploys an MLflow tracking server for experiment tracking and model management. The server provides a centralized location for logging metrics, parameters, and artifacts from machine learning experiments.

## Prerequisites

Ensure the following prerequisites are met before deploying this workload:

1. **Helm**: Install `helm`. Refer to the [Helm documentation](https://helm.sh/) for instructions.
2. **MinIO Storage** (recommended): Create the following secret in the namespace for artifact storage:
   - `minio-credentials` with keys `minio-access-key` and `minio-secret-key`

## Configuration Parameters

You can configure the following parameters in the `values.yaml` file or override them via the command line:

### Backend Store Configuration

| Parameter                              | Description                                                    | Default                                    |
|---------------------------------------|----------------------------------------------------------------|--------------------------------------------|
| `backendStore.type`                   | Backend storage type (sqlite, postgres, mysql)               | `sqlite`                                   |
| `backendStore.dbpath`                 | SQLite database file path                                     | `/workload/mlflow.db`                     |
| `backendStore.host`                   | Database host (for postgres/mysql)                           | `""`                                       |
| `backendStore.port`                   | Database port (for postgres/mysql)                           | `""`                                       |
| `backendStore.database`               | Database name (for postgres/mysql)                           | `""`                                       |
| `backendStore.driver`                 | Database driver (for mysql)                                  | `""`                                       |
| `backendStore.secret.name`            | Secret name for database credentials                          | `mlflow-db-credentials`                    |
| `backendStore.secret.userKey`         | Key in secret for username                                    | `username`                                 |
| `backendStore.secret.passwordKey`     | Key in secret for password                                    | `password`                                 |

### Artifact Storage Configuration

| Parameter                              | Description                                                    | Default                                    |
|---------------------------------------|----------------------------------------------------------------|--------------------------------------------|
| `env_vars.MLFLOW_S3_ENDPOINT_URL`     | S3-compatible storage endpoint for artifacts                 | MinIO service URL                          |
| `env_vars.MLFLOW_ARTIFACTS_DESTINATION` | Artifact storage destination                               | `s3://mlflow/mlartifacts`                 |
| `env_vars.AWS_ACCESS_KEY_ID`          | AWS access key ID configuration from secret                  | `minio-credentials` secret                 |
| `env_vars.AWS_SECRET_ACCESS_KEY`      | AWS secret access key configuration from secret              | `minio-credentials` secret                 |

For more details see the `values.yaml` file.

## Deploying the Workload

It is recommended to use `helm template` and pipe the result to `kubectl apply`, rather than using `helm install`.

To deploy the chart with the release name `mlflow-server`, run the following command from the `helm/` directory:

```bash
helm template mlflow-server . | kubectl apply -f -
```

### Custom Configuration

You can override configuration values using command line parameters:

#### Database Backend Configuration

```bash
helm template mlflow-server . \
  --set backendStore.type="postgres" \
  --set backendStore.host="postgres.example.com" \
  --set backendStore.port="5432" \
  --set backendStore.database="mlflow" \
  --set backendStore.secret.name="my-db-credentials" | kubectl apply -f -
```

#### Using Local Storage Override

```bash
helm template mlflow-server . \
  -f overrides/backends/local_artifacts.yaml | kubectl apply -f -
```

#### Custom Local Storage Path

```bash
helm template mlflow-server . \
  --set env_vars.MLFLOW_ARTIFACTS_DESTINATION="/workload/custom/artifacts" \
  --set storage.ephemeral.quantity="1Ti" | kubectl apply -f -
```

## Accessing the MLflow Web UI

### Local Access via Port Forwarding

To access the MLflow UI from your local machine:

1. **Forward the service port** to your local machine:

   ```bash
   kubectl port-forward services/mlflow-server 8080:80
   ```

2. **Open the MLflow UI** in your browser:

- When using HTTPRoute or Ingress, the URL path prefix (`<project_id>/[<user_id>/]<workload_id>/`) is automatically handled by the routing layer. The `user_id` segment is omitted from the path when it's not specified, i.e. a project-wise deployment.
- For direct access via port-forward, no path prefix is needed since you're connecting directly to the MLflow service

### Verify Deployment

Check the deployment status:

```bash
kubectl get deployment
kubectl get service
```

### View Logs

To view the MLflow server logs:

```bash
kubectl logs -f deployment/mlflow-server
```

## Storage Configuration

### SQLite Backend (Default)

By default, MLflow uses SQLite for the backend store, with the database file stored in ephemeral storage.

### PostgreSQL/MySQL Backend

For production deployments, configure a PostgreSQL or MySQL backend:

```yaml
backendStore:
  type: postgres  # or mysql
  host: "your-db-host"
  port: "5432"
  database: "mlflow"
  driver: ""  # required for mysql, e.g., "pymysql"
  secret:
    name: "mlflow-db-credentials"
    userKey: "username"
    passwordKey: "password"
```

Create the required secret for database credentials:

```bash
kubectl create secret generic mlflow-db-credentials \
  --from-literal=username=myuser \
  --from-literal=password=mypassword
```

### Artifact Storage

Artifacts are stored in S3-compatible storage (MinIO) by default. The configuration supports:

- Local filesystem storage
- S3-compatible object storage (MinIO, AWS S3, etc.)

### Persistent Storage

The chart supports optional persistent storage volumes:

```yaml
persistent_storage:
  enabled: true
  volumes:
    pvc-user:
      pvc_name: "pvc-user-{{ .Values.metadata.user_id }}"
      mount_path: "/workload/{{ .Values.metadata.user_id }}"
```

When enabled, this creates persistent volumes that can be shared across workload restarts and used for storing user data or models.

## Health Checks

The deployment includes comprehensive health checks:

- **Startup Probe**: Checks if the container has successfully started. It disables liveness and readiness probes until it succeeds, useful for slow-starting applications.
- **Liveness Probe**: Checks if the container is still alive. If it fails, Kubernetes restarts the container to recover from failure.
- **Readiness Probe**: Checks if the container is ready to serve traffic. If it fails, the container is removed from the service's endpoints but remains running.

All probes use the `/health` endpoint on the HTTP port. The startup probe has a higher failure threshold (20) to accommodate longer startup times.

## Kaiwo Integration

The chart supports integration with Kaiwo for advanced workload management:

```yaml
kaiwo:
  enabled: true
```

When enabled, this uses Kaiwo CRDs to have the Kaiwo operator manage the workload lifecycle.

## Using MLflow in Your ML Projects

Once deployed, you can use this MLflow tracking server in your machine learning experiments:

### Basic Usage

```python
import mlflow

# Set the tracking server URI, assuming the release name as "mlflow-server-service"
mlflow.set_tracking_uri("http://mlflow-server-service/")

# Start an experiment
mlflow.set_experiment("my-experiment")

# Log parameters, metrics, and artifacts
with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_artifact("model.pkl")
```

## Accessing via URL

To access the workload through a URL, you can enable either an Ingress or HTTPRoute in the `values.yaml` file by setting `ingress.enabled: true` or `http_route.enabled: true`.

### Access URLs

The MLflow tracking server can be accessed via different methods depending on your deployment (assuming the release name as "mlflow-server-service"):

- **Port Forward**: `http://localhost:8080` (after running `kubectl port-forward services/mlflow-server-service 8080:80`)
- **Ingress/HTTPRoute**: `https://your-domain.com/<project_id>/[<user_id>/]<workload_id>/` (when ingress is enabled)
- **Internal Cluster Access**:
  - Within the same namespace: `http://mlflow-server-service`
  - From different namespaces: `http://mlflow-server-service.<namespace>.svc.cluster.local:80`
  - Used for service-to-service communication within the Kubernetes cluster
  - Example usage in application code:
    ```python
    mlflow.set_tracking_uri("http://mlflow-server-service")
    ```
