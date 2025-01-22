# kaiwo workload template for example materials

This is an example of a delivery container, which downloads a model and some data, and uploads them to GCP.
- `values.yaml` specifies the model and target paths
- `download_data.py` is the script which downloads and preprocesses the data, saving it into the local directory local_datasets/
- `env` file contains bucket access secret specification, and potentially huggingface hub token secret spec

Example command to submit via kaiwo
```bash
kaiwo submit \
    --imagepullsecret regcred \
    --template template/deliver-example-materials.yaml.tmpl \
    --custom-config llama-3.1-tiny-random-and-argilla-human-prompts/values.yaml \
    --path llama-3.1-tiny-random-and-argilla-human-prompts \
    --namespace silogen
```
