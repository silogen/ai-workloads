# kaiwo workload template to download a model to bucket storage

This is a workload which downloads a model and uploads it to bucket storage. Two approaches of doing this are included:

## kaiwo version

This has been tested with [kaiwo v0.0.5](https://github.com/silogen/kaiwo/releases/tag/v.0.0.5).

## Approach 1: with custom template

The first approach leverages custom templates:

* `download-huggingface-model.yaml.tmpl` specifies a custom template
* `values.yaml` specifies the model and target paths
* `env` file contains bucket access secret specification, and potentially huggingface hub token secret spec

Example command to submit via kaiwo

```bash
kaiwo submit \
    --template download-huggingface-model.yaml.tmpl \
    --custom-config path-to-your-custom-config.yaml \
    --path path-to-your-workload-directory-containing-env
```

### --custom-config YAML file structure

```yaml
modelID: # Huggingface model id, which is in the format of organization/model-name
bucketPath: # Path in the bucket storage where this model should be stored. In the format bucket-name/path/separated/by/slashes/name-for-resulting-directory
modelRevision: # Optional string that specifies which revision of the model should be downloaded.
downloadExcludeGlob: # Optional string that specifies which files in the huggingface model repository should be excluded from the download, default: 'original/*'
allowOverwrite: # Optionally set to true to allow overwriting existing files in the bucket, default false
```

### required variables in env file:

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

## Approach 2: without custom template

Alternatively, you can use kaiwo's built-in templates. These are the examples named `*-no-custom-template`
* `env` file contains bucket access secret specification, and potentially huggingface hub token secret spec and other environment variables.
* `entrypoint` specifies the code to execute when the container starts; parametrized by environment variables.

Example command to submit via kaiwo

```bash
kaiwo submit \
    --path path-to-your-workload-directory-ending-in-no-custom-template
```
