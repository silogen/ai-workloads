# Life Science - SemlaFlow

This Helm Chart deploys the LLM Inference SemlaFlow workload.

# SemlaFlow model

Original repo is [here](https://github.com/rssrwn/semla-flow)

This project creates a novel equivariant attention-based message passing architecture, Semla, for molecular design and dynamics tasks. We train a molecular generation model, SemlaFlow, using flow matching with optimal transport to generate realistic 3D molecular structures.

## Scripts

There are 4 scripts in the original semlaflow repository:
* `preprocess` - Used for preprocessing larger datasets into the internal representation used by the model for training
* `train` - Trains a MolFlow model on preprocessed data
* `evaluate` - Evaluates a trained model and prints the results
* `predict` - Runs the sampling for a trained model and saves the generated molecules

## Instructions on choosing a GPU to attach a docker container to
 1. check with `amd-smi process` which GPU is free
 2. check with `rocm-smi` what is the node id of the free GPU (Note: node id is not the same as the device id and is displayed in the second column of the rocm-smi output)
 3. If say, the node id 2 gpu is free, the device to be added to docker run is given by `cat /sys/class/kfd/kfd/topology/nodes/2/properties | grep drm_render_minor`
 4. you can directly create a container using docker run --device=/dev/kfd --device=/dev/dri/renderD<ID output from step 3>

## Running inference interactively

Start a container with the above mentioned image on a cluster and connect.

Each script can be run as follows (where `<script>` is replaced by the script name above without `.py`): `python -m semlaflow.<script> --data_path <path/to/data> <other_args>`

Example:

```
cd semla-flow
python -m semlaflow.evaluate --data_path data/qm9/smol --ckpt_path models/300epochs.ckpt --dataset qm9
```

This workload evaluates a pretrained SemlaFlow model. By changing the script name and arguments in the values.yaml file, you can use the same container to either train a new model or run predictions with a pretrained one.

### Expected outputs

The training script trains a model and saves checkpoints in the lightning_logs folder. The evaluate script then assesses a specified model, producing results like these.


| Metric              | Result                     |
|---------------------|----------------------------|
| connected-validity  | 0.91710 ± 0.0014142        |
| energy              | 113.95503 ± 0.3756851      |
| energy-per-atom     | 2.45049 ± 0.0050934        |
| energy-validity     | 0.94130 ± 0.0024590        |
| novelty             | 0.99664 ± 0.0001804        |
| opt-energy-validity | 0.94130 ± 0.0024590        |
| opt-rmsd            | 0.86588 ± 0.0062672        |
| strain              | 73.87280 ± 0.2942116       |
| strain-per-atom     | 1.55302 ± 0.0092829        |
| uniqueness          | 0.99986 ± 0.0001997        |
| validity            | 0.94203 ± 0.0023099        |
| atom-stability      | 0.99887 ± 0.0000082        |
| molecule-stability  | 0.97513 ± 0.0005249        |


## Prerequisites

Ensure the following prerequisites are met before deploying any workloads:

1. **Helm**: Install `helm`. Refer to the [Helm documentation](https://helm.sh/) for instructions.

## Deploying the Workload

It is recommended to use `helm template` and pipe the result to `kubectl create` , rather than using `helm install`. Generally, a command looks as follows

```bash
helm template [optional-release-name] <helm-dir> -f <overrides/xyz.yaml> --set <name>=<value> | kubectl apply -f -
```

The chart provides three main ways to deploy models, detailed below.

## User Input Values

Refer to the `values.yaml` file for the user input values you can provide, along with instructions.

## Interacting with Deployed Model

### Verify Job

Check the job status:

```bash
kubectl get jobs
```
