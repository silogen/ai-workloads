# Download a model from Weights and Biases to bucket storage

This is an workload which downloads a model from weights and biases and uploads it to bucket storage.

Run example:
```bash
helm template "dl-from-wandb" workloads/download-wandb-model-to-bucket/helm \
    -f workloads/download-wandb-model-to-bucket/helm/overrides/example-model-to-minio.yaml \
    | kubectl create -f -
```

## User input values

See the `values.yaml` file for the user input values that you can provide, with instructions.
