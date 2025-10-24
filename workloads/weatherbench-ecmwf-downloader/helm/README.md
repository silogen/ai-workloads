# ECMWF data downloader and preprocessor

This workload downloads data from the Climate Data Storage (CDS) operated by the
European Centre for Medium-Range Weather Forecasts (ECMWF). The data is
preprocessed to be compatible with Google's Weatherbench and then uploaded to
Minio.

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

This sets up the pod, downloads the data, preprocesses it and uploads it into Minio.

## Script params

| **Parameter**           | **Description**                          | **Default**                      |
|-------------------------|------------------------------------------|-------------------------------   |
| `minio_bucket`          | Target bucket in Minio.                  | `your-bucket-name`               |
| `minio_prefix`          | Prefix for uploads in Minio.             | `your-prefix`                    |
| `start_date`            | Starting date for data.                  | `2023-01-01`                     |
| `end_date`              | Ending date for data.                    | `2023-01-31`                     |
| `tods`                  | Times of day for data.                   | `00:00 12:00`                    |
| `surface_variables`     | Surface variables to download.           | Weatherbench headline variables. |
| `vertical_variables`    | Vertical variables to download.          | Weatherbench headline variables. |
| `pressure_levels`       | Pressure levels to download.             | Weatherbench defaults.           |


## Environment Variables

| **Parameter**              | **Description**                     | **Default**                                                  |
|----------------------------|-------------------------------------|--------------------------------------------------------------|
| `MINIO_ACCESS_KEY`         | Minio access key.                   | *(from secret: minio-credentials / minio_access_key_id)*     |
| `MINIO_SECRET_ACCESS_KEY`  | Minio secret key.                   | *(from secret: minio-credentials / minio_secret_access_key)* |
| `SSL_CERT_FILE`            | SSL certificate bundle path.        | *not defined (SSL not used)*                                 |
| `MINIO_ENDPOINT`           | MinIO or S3-compatible endpoint URI | `your-minio-endpoint`                                        |
| `CDSAPI_URL`               | URL for the CDS API                 | *(from secret: cds-api-secret / url)*                        |
| `CDSAPI_KEY`               | Key for the CDS API                 | *(from secret: cds-api-secret / key)*                        |
