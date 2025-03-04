# Finetuning with the Silogen finetuning engine

This is a Helm Chart for finetuning Jobs based on the Silogen finetuning engine.
The chart integrates the finetuning config as part of the values.yaml input.

See the values.yaml file for the general structure (more documentation coming soon).

## Running the workload
Since the `helm install` semantics are centered around on-going installs, not jobs that run once,
it's best to just run `helm template` and pipe the result to `kubectl create`.

Example command:
```bash
helm template ./chart \
  -f ../examples/llama-31-tiny-random-deepspeed-values.yaml \
  --namepsace silogen --name-template llama-31-tiny-random-deepspeed-alpha \
  | kubectl create -f -
```
