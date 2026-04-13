# ECMWF ai-models — Helm Deployment

This Helm chart deploys aurora using the `ai-models` library by ECMWF. It handles model setup, packaging, and inference in a fully containerized GPU environment.

First you need to get API key from **The Climate Data Store (CDS)** in order to download ERA5 dataset. Instructions for generating the key are available on the [CDSAPI setup](https://cds.climate.copernicus.eu/how-to-api) page. This CDS API Key should be added as a Kubernetes secret (<code>cds-api-secret</code> in our case).

**NOTE:** This workload requires the namespace to have this secret. Please work with your cluster administrator to make it available according to your cluster's best practices

## 🔧 Project Structure

```
helm/
├── Chart.yaml # Helm chart metadata
├── values.yaml # Main config values (e.g. image, GPU)
├── templates/ # Helm templates and helpers for deployment, config, etc.
│ ├── _helpers.tpl # Helpers common to other helm charts in the repo
│ ├── configmap.yaml
│ ├── inference-job.yaml
├── mount/ # Files mounted into the container via ConfigMap
│ ├── grib_visualizer.py
│ ├── README.md
│ ├── setup_env.sh
│ ├── run_model.sh
├── overrides/ # Optional values overrides
```

## 🚀 Quickstart

Deploy directly with:

```bash
helm template aurora ./helm | kubectl apply -f -

```

This sets up the pod, installs dependencies, downloads the model, runs inference and writes outputs to the `.grib` file. Then visualizes the `.grib` files and writes everything to minio.

## Script parameters

| **Parameter**           | **Description**                          | **Default**                   |
|-------------------------|------------------------------------------|-------------------------------|
| `model_name`            | model name (aurora, aurora-0.1-finetuned, aurora-2.5-finetuned, aurora-2.5-pretrained)       | `aurora`            |
| `lead_time`             | Forecast lead time in hours              | `48`                          |
| `visualize`             | Visualization (gifs)                     | `true`                        |
| `model_out_dir`         | Directory for model outputs              | `/workspace/model_out`        |
| `log_dir`               | Log file directory                       | `$MODEL_OUT_DIR/logs`         |
| `pred_dir`              | Prediction output directory              | `$MODEL_OUT_DIR/predictions`  |
| `assets_dir`            | Auxiliary files directory                | `/workspace/assets`           |
| `mountdir`              | Mount directory for additional scripts   | `/workload/mount`             |
| `date`                  | Date in YYYYMMDD format                  | `20230110`                    |
| `time`                  | Time in HHMM format                      | `0000`                        |
| `MINIO_BUCKET`          | MinIO bucket name for output storage     | `your-bucket-name`            |
| `MINIO_PREFIX`          | Subdirectory/prefix added to upload paths| `$MODEL_NAME/$TIMESTAMP`      |


## Environment Variables

| **Parameter**              | **Description**                     | **Default**                                              |
|----------------------------|-------------------------------------|----------------------------------------------------------|
| `MINIO_ACCESS_KEY`        | MinIO access key ID                   | *(from secret: minio-credentials / minio-access-key)*     |
| `MINIO_SECRET_KEY`    | MinIO secret key                      | *(from secret: minio-credentials / minio-secret-key)* |
| `MINIO_ENDPOINT`           | MinIO or S3-compatible endpoint URI | `your-minio-endpoint`                                    |
| `CDSAPI_URL`               | URL for the CDS API                 | *(from secret: cds-api-secret / url)*                    |
| `CDSAPI_KEY`               | Key for the CDS API                 | *(from secret: cds-api-secret / key)*                    |
