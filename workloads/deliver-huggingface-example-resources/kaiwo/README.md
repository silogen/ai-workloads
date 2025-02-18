# kaiwo workload template for example materials

This workload is meant to setup a model and some data for a client, so that we can run further workloads like training.
This downloads a model and some data from huggingface, and uploads them to bucket storage.
The inputs required:
- `--custom-config` needs to point to a YAML file that specifies the model and target paths.
- `--path` needs to point to a workload inputfiles directory that contains:
    - `download_data.py`, a script which downloads and preprocesses the data, saving it into the local directory local_datasets/
    - `env` file contains bucket access secret specification, and potentially huggingface hub token secret spec

## kaiwo version
This has been tested with [kaiwo v0.0.5](https://github.com/silogen/kaiwo/releases/tag/v.0.0.5).

## --custom-config file format

```yaml
modelID: # Huggingface model ID in the organization/model-name format
bucketModelPath: # Path where to upload the model, bucket-name/path/in/bucket/ending/in/desired-model-name
bucketDataDir: # Directory path where all the data files from local_datasets/ are uploaded to
ephemeralStorageRequest:  # Optionally specify a value of ephemeral storage that you reserve for this job. Can be useful to request enough space if you're downloading a very large model.
modelRevision: # Optional string that specifies which revision of the model should be downloaded.
downloadExcludeGlob: # Optional string that specifies which files in the huggingface model repository should be excluded from the download, default: 'original/*'
allowOverwrite: # Optionally set to true to allow overiwriting existing files in the bucket, default false
```

## llama-3.1-tiny-random-and-argilla-human-prompts
This is an example of the inputfiles required for this workload. These input files download a tiny Llama 3.1 -like random initialized model - the model is not trained, it's tiny, it's meant for debugging and showing that training executes. The data is an openly available chat finetuning dataset.

Example command to submit via kaiwo:
```bash
kaiwo submit \
    --template deliver-example-materials.yaml.tmpl \
    --custom-config llama-3.1-tiny-random-and-argilla-human-prompts/values.yaml \
    --path llama-3.1-tiny-random-and-argilla-human-prompts
```
