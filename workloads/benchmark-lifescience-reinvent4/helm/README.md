# Life Science - Reinvent4

This Helm Chart deploys a workload as a Kubernetes job for REINVENT4 run mode Transfer Learning (TL)

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

### Verify Job

Check the job status:

```bash
kubectl get jobs
```

# Running Reinvent inference interactively

Connect to the pod with your favorite terminal.

The job runs the script `docker/lifescience/reinvent4/Reinvent_TLRL_clean.py` automatically. The logs can be followed by running `kubectl logs <pod_name> -f`

Alternatively, you can uncomment the bottom part of the `values.yaml` file to run `Reinvent_demo_clean.py` as well. Or you can interactively connect to the job and run either of the notebooks manually. Just make sure you don't run two scripts at the same time by accident.

```sh
# Connect to the pod
kubectl exec -it <pod_name> -- /bin/bash

python3 notebooks/<notebook_name.py>
```

Alternatively, Reinvent jobs can be run by:
```sh
reinvent -l <log_name> <config_name>
```

## Expected outputs from the demo runs

- `Reinvent_demo_clean.py`

|    Agent   |   Prior   |   Target   |     Score    |                                               SMILES                                               | SMILES_state |     QED     | QED (raw) | Stereo | Stereo (raw) | Alerts | Alerts (raw) | step |
|------------|-----------|------------|--------------|-----------------------------------------------------------------------------------------------------|--------------|-------------|-----------|--------|---------------|--------|---------------|------|
|  38.0957   |  38.0957  | -38.0276   |  0.000531    | `Cc1ccc(C2CC(=O)Nc3cccc(NC45CC6CC(CC4C6)C5)c32)cc1`                                               |      1       |  0.751039   |   0.7510  |  0.0   |      3.0      |  1.0   |      1.0      |  1   |
|  29.6628   |  29.6628  |  89.1671   |  0.928359    | `Cc1cc(-c2nnn(CC3CCCCC3)n2)nc(C(=O)NC2CCC2)`n1                                                    |      1       |  0.883472   |   0.8835  |  1.0   |      0.0      |  1.0   |      1.0      |  1   |
|  33.0351   |  33.0351  | -33.0351   |  0.000000    |` COc1ccc(CCCCON=C2CCC3CC2CN3C(=O)OCc2ccccc2)cc1`                                                  |      1       |  0.000000   |   0.0000  |  0.0   |      0.0      |  0.0   |      0.0      |  1   |
|  22.6197   |  22.6197  | -22.6197   |  0.000000    | `CCOC(=O)C1C(=O)N=C(N)NC1c1ccc2c(c1)OCO2`                                                         |      1       |  0.000000   |   0.0000  |  0.0   |      0.0      |  0.0   |      0.0      |  1   |
|  28.3920   |  28.3920  | -28.3144   |  0.000606    | `CC(NC(=O)c1nccs1)c1ccc2c(c1)COC2`                                                                  |      1       |  0.935827   |   0.9358  |  0.0   |      1.0      |  1.0   |      1.0      |  1   |

Total number of SMILES generated: 30000\
Total number of invalid SMILES: 283\
Total number of batch duplicate SMILES: 8\
Total number of duplicate SMILES: 1317

- `Reinvent_TLRL_clean.py`

(This is the number of produced "good binders" defined by `QED < 0.8` and `ChemProp (raw) < -25.0` before and after removing duplicates.)

4\
4
