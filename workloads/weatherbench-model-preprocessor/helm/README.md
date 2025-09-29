# ECMWF ai-models output preprocessor

This workload downloads, preprocesses and combines weather prediction model
outputs produced by the `ai-models` tool developed by the European Centre for
Medium-Range Weather Forecasts (ECMWF).

The `ai-models` tool presents an unified front-end for running various
state-of-the-art machine learning weather prediction models. However, the model
outputs must be preprocessed for compatibility with the Weatherbench weather
prediction model evaluation suite, developed by Google.

This workload downloads the given `ai-models` output files for a particular
model. It then preprocesses them for Weatherbench compatibility and combines
them into a single file. This file is then uploaded back into Minio.

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

| **Parameter**           | **Description**                          | **Default**                   |
|-------------------------|------------------------------------------|-------------------------------|
| `minio_bucket`          | Target bucket in Minio.                  | `your-bucket-name`            |
| `minio_prefix`          | Prefix for uploads in Minio.             | `your-prefix`                 |
| `model_name`            | Indicates which preprocessor to use.     | `gencast`                     |
| `output`                | Filename for the output.                 | `gencast.nc`                  |
| `inputs`                | The model input files.                   | `[input1.grib, input2.grib]`  |


## Environment Variables

| **Parameter**              | **Description**                     | **Default**                                                  |
|----------------------------|-------------------------------------|--------------------------------------------------------------|
| `MINIO_ACCESS_KEY`         | Minio access key.                   | *(from secret: minio-credentials / minio_access_key_id)*     |
| `MINIO_SECRET_ACCESS_KEY`  | Minio secret key.                   | *(from secret: minio-credentials / minio_secret_access_key)* |
| `SSL_CERT_FILE`            | SSL certificate bundle path.        | *not defined (SSL not used)*                                 |
| `MINIO_ENDPOINT`           | Minio or S3-compatible endpoint URI | `your-minio-endpoint`                                        |
