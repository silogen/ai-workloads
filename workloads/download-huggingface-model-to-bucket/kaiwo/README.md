# kaiwo workload template to download a model to bucket storage

This is an workload which downloads a model and uploads it to GCP.
- `values.yaml` specifies the model and target paths
- `env` file contains bucket access secret specification, and potentially huggingface hub token secret spec

Example command to submit via kaiwo
```bash
kaiwo submit \
    --template download-huggingface-model.yaml.tmpl \
    --custom-config path-to-your-custom-config.yaml \
    --path path-to-your-workload-directory-containing-env \
```

## --custom-config YAML file structure

```yaml
modelID: # Huggingface model id, which is in the format of organization/model-name
bucketPath: # Path in the bucket storage where this model should be stored. In the format bucket-name/path/separated/by/slashes/name-for-resulting-directory
```

## required variables in env file:
```yaml
envVars:
  - name: BUCKET_STORAGE_HOST
    value: # The URL of the bucket storage host
  # The bucket storage HMAC access key should be available as a secret in the cluster:
  - fromSecret:
      name: "BUCKET_STORAGE_ACCESS_KEY"
      secret: # The name of the secret in the cluster
      key: # The key inside the secret
  - fromSecret:
      name: "BUCKET_STORAGE_SECRET_KEY"
      secret: # The name of the secret in the cluster
      key: # The key inside the secret
  # If the model is private/gated, you need to make the huggingface token available as a secret in the cluster and specify it:
  # - fromSecret:
  #     name: "HF_TOKEN"
  #     secret: # The name of the secret in the cluster
  #     key: # The key inside the secret
```