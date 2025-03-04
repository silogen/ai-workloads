# Engine agnostic finetuning workload / Silogen HuggingFace finetuning as default

This kaiwo template comes with a default entrypoint which runs LLM SFT
built on using HuggingFace libraries and distributed using accelerate,
and more specifically, we recommend the Deepspeed integration in accelerate.

## kaiwo version
Currently tested with kaiwo version 0.0.5

Example of command to run:
```bash
kaiwo submit \
  --template finetuning.yaml.tmpl \
  --custom-config ../examples/meta-llama-3.1-8b-argilla-sft/values.yaml \
  --path ../examples/meta-llama-3.1-8b-argilla-sft/ \
  --gpus 1
```

Note that the --image for kaiwo is ignored with this template.

## Format of --custom-config

```yaml
mainImage: # Use this to specify your own image, or don't specify to use the Silogen Huggingface finetuning engine.
baseModel: # Directory path to your basemodel in bucket storage, downloaded to /local_resources/basemodel
dataDownloads: # List of data to download, in the following format:
  - bucketPath: # Path to file or directory in the bucket
    localName: # The output goes to  /local_resources/<localName>
  - # Note that you can specify as many data files as you need
checkpointsBucketPath: # Path in the bucket where checkpoints get uploaded
downloadsSizeLimit: # Hard disk space requrest for the downloads, defaults to 128Gi
checkpointsSizeLimit: # Hard disk space request for the checkpoints, defaults 256Gi
method: # Optional parameter, if using the Silogen finetuning engine, to specify the finetuning method, defaults to "sft", "dpo" is also supported.
```

## Input files in the workload directory
If not specifying a custom entrypoint, the following additional files are required:
- finetuning\_config.yaml
    The Silogen finetuning engine config YAML file. Note that this needs to reference
    the data and the basemodel with downloads paths under /local\_resources/
- accelerate\_config.yaml
    The HuggingFace accelerate config YAML file. Note that this should specify the
    correct number of processes (usually matching the number of GPUs you request).

## Env file
You must specify the following envVars in the env file: BUCKET\_STORAGE\_HOST,
BUCKET\_STORAGE\_ACCESS\_KEY, BUCKET\_STORAGE\_SECRET\_KEY. For example:

```yaml
- name: BUCKET_STORAGE_HOST
  value: https://storage.googleapis.com
- fromSecret:
    name: BUCKET_STORAGE_ACCESS_KEY
    secret: bucket-storage-credentials
    key: access_key
- fromSecret:
    name: BUCKET_STORAGE_SECRET_KEY
    secret: bucket-storage-credentials
    key: secret_key
```

## Other finetuning engines

Other finetuning engines may be used with this template by setting `mainImage: ...` in the `--custom-config`,
and providing a matching entrypoint file in the workload inputfiles path.

To use the download and upload containers take note of the following:
- A volume is mounted to `/workdir/checkpoints` in your container, checkpoints are first synced there and then anything there is continuously synced to the bucket. So you must make sure your program outputs checkpoints in that directory.
- The path given as baseModel in the `--custom-config` is downloaded to /local_resources/basemodel.
- Files given in dataDownloads in the `--custom-config` are downloaded to /local_resources/ with the name you specify.
