#!/usr/bin/env python3
import pathlib

import yaml

download_template = """\
# Which model to download
modelID: {model_id}

# Where the resources should be stored:
bucketPath: default-bucket/models/{model_id}
bucketStorageHost: http://minio.minio-tenant-default.svc.cluster.local:80

# Download & Upload configuration:
downloadExcludeGlob: "original/*"  # Exclude things from the HuggingFace download with this
allowOverwrite: false

# Storage configuration:
storageClass: mlstorage
storageQuantity: "{storage_quantity}"

# HF Token:
hfTokenSecret:
  name: hf-token
  key: hf-token

kaiwo:
  enabled: true
"""
output_dir = pathlib.Path(__file__).parent.parent.parent / "download-huggingface-model-to-bucket" / "helm" / "overrides"


def get_model_info(modelfile):
    modeldata = yaml.safe_load(modelfile.read_text())
    model_id = modeldata.get("model")
    if model_id is None:
        raise ValueError(f"Model file {modelfile} does not contain a 'model' key.")
    storage_quantity = modeldata.get("downloadsReservedSize", "160Gi")
    return model_id, storage_quantity


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate model download overrides.")
    parser.add_argument("modelfile", type=pathlib.Path, help="Path to the model YAML file.")
    args = parser.parse_args()
    model_id, storage_quantity = get_model_info(args.modelfile)
    print(f"Generating download config for {model_id} with storage {storage_quantity}.")
    download_config = download_template.format(
        model_id=model_id,
        storage_quantity=storage_quantity,
    )
    output_path = output_dir / f"{model_id.replace('/', '_')}.yaml"
    print(f"Writing download config to {output_path}")
    with open(output_path, "w") as f:
        f.write(download_config)
