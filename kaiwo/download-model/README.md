# kaiwo workload template to download a model

This is an workload which downloads a model and uploads it to GCP.
- `values.yaml` specifies the model and target paths
- `env` file contains bucket access secret specification, and potentially huggingface hub token secret spec

Example command to submit via kaiwo
```bash
kaiwo submit \
    --imagepullsecret regcred \
    --template template/download-huggingface-model.yaml.tmpl \
    --custom-config workloads/llama-3.1-tiny-random/values.yaml \
    --path workloads/llama-3.1-tiny-random \
    --namespace silogen
```
