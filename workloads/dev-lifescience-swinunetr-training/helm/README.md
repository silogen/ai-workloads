# Life Science - SwinUNETR Training

This Helm Chart to train the MONAI [SwinUNETR](https://arxiv.org/abs/2201.01266) model on the [nsclc-radiomics](https://www.cancerimagingarchive.net/collection/nsclc-radiomics/) dataset.

The [nsclc-radiomics](https://www.cancerimagingarchive.net/collection/nsclc-radiomics/) is loaded using the MONAI [TciaDataset](https://github.com/Project-MONAI/MONAI/blob/b58e883c887e0f99d382807550654c44d94f47bd/monai/apps/datasets.py#L404).

# SwinUNETR model

SwinUNETR is a deep learning architecture designed for medical image segmentation, particularly in 3D volumetric data
such as CT or MRI scans with the aim to detect tumors in the images. It combines the strengths of two powerful
models: 1. Swin Transformer - a hierarchical vision transformer that captures long-range dependencies and contextual
information efficiently and 2. UNETR (UNet with Transformers) - a transformer-based encoder-decoder architecture
tailored for medical image segmentation.

## Prerequisites

Ensure the following prerequisites are met before deploying any workloads:

1. **Helm**: Install `helm`. Refer to the [Helm documentation](https://helm.sh/) for instructions.

## Deploying the Workload

It is recommended to use `helm template` and pipe the result to `kubectl create`, rather than using `helm install`. Generally, a command looks as follows

```bash
helm template [your-release-name] ./helm | kubectl apply -f -
```

The chart provides three main ways to deploy models, detailed below.

## User Input Values

Refer to the `values.yaml` file for the user input values you can provide, along with instructions.

## Interacting with Deployed Model

### Verify Deployment

Check the your deployed job status:

```bash
kubectl get jobs
```
