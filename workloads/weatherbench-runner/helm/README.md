# Weatherbench runner

This workload downloads preprocessed European Centre for Medium-Range Weather
Forecasts (ECMWF) reference data and preprocessed weather model forecast data,
runs Google's Weatherbench on these datasets, and finally uploads the resulting
metrics file to Minio.

## 🔧 Project Structure

```
helm/
├── Chart.yaml # Helm chart metadata.
├── values.yaml # Main config values.
├── templates/ # Helm templates and helpers for deployment, config, etc.
│ ├── _helpers.tpl # Helpers common to other helm charts in the repo.
│ ├── job.yaml
├── overrides/ # Optional values overrides.
```

## 🚀 Quickstart

Deploy directly with:

```bash
helm template ./helm -f ./helm/overrides/my_override.yaml --name-template my-job-name | kubectl apply -f -

```

This sets up the pod, downloads the data, preprocesses it and uploads the result back into Minio.

## Script params

| **Parameter**           | **Description**                               | **Default**                         |
|-------------------------|-----------------------------------------------|-------------------------------------|
| `minio_bucket`          | Target bucket in Minio.                       | `your-bucket-name`                  |
| `minio_prefix`          | Prefix for uploads in Minio.                  | `your-prefix`                       |
| `start_date`            | Start of calculation (inclusive, ISO format). | `2023-01-01`                        |
| `end_date`              | End of calculation (exclusive, ISO format).   | `2023-01-03`                        |
| `forecast_period`       | Lead time period to use (Pandas timedelta).   | `12h`                               |
| `forecast_steps`        | Number of lead time steps.                    | `4`                                 |
| `cds_file`              | Location of reference data.                   | `ecmwf_data/reference.nc`           |
| `models`                | Model data as `name` `file` pairs.            | `[gencast, model_data/gencast.nc]`  |
| `wb_metrics`            | Metrics to run (array).                       | `[rmse, mae]`                       |
| `wb_variables`          | Variables to run the metrics over (array).    | `[geopotential, temperature]`       |


## Environment Variables

| **Parameter**              | **Description**                     | **Default**                                                  |
|----------------------------|-------------------------------------|--------------------------------------------------------------|
| `MINIO_ACCESS_KEY`         | Minio access key.                   | *(from secret: minio-credentials / minio_access_key_id)*     |
| `MINIO_SECRET_ACCESS_KEY`  | Minio secret key.                   | *(from secret: minio-credentials / minio_secret_access_key)* |
| `SSL_CERT_FILE`            | SSL certificate bundle path.        | *not defined (SSL not used)*                                 |
| `MINIO_ENDPOINT`           | Minio or S3-compatible endpoint URI | `your-minio-endpoint`                                        |


## 🏗 Target Platform

This deployment is configured for the  **OSSCI cluster**.
