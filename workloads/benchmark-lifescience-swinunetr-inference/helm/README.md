# Life Science - SwinUNETR Inference

This Helm Chart deploys a [SwinUNETR](https://arxiv.org/abs/2201.01266) model as an Inference Service, for multiorgan segmentation of 3D CT scans.

## SwinUNETR model

[SwinUNETR](https://arxiv.org/abs/2201.01266) is a deep learning architecture designed for medical image segmentation, particularly in 3D volumetric data
such as CT or MRI scans with the aim to detect tumors in the images. It combines the strengths of two powerful
models: 1. Swin Transformer - a hierarchical vision transformer that captures long-range dependencies and contextual
information efficiently and 2. UNETR (UNet with Transformers) - a transformer-based encoder-decoder architecture
tailored for medical image segmentation.

Model weights are loaded from [HuggingFace](https://huggingface.co/darragh/swinunetr-btcv-base).

Check out the [demo_inference_service.ipynb](../examples/demo_inference_service.ipynb) example to see how it works.

## Data

The training data are 3D CT scans from the [BTCV challenge dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217752). The target segmentation includes 13 abdominal organs:

1. Spleen
2. Right Kidney
3. Left Kideny
4. Gallbladder
5. Esophagus
6. Liver
7. Stomach
8. Aorta
9. IVC
10. Portal and Splenic Veins
11. Pancreas
12. Right adrenal gland
13. Left adrenal gland


## Prerequisites

Ensure the following prerequisites are met before deploying any workloads:

1. **Helm**: Install `helm`. Refer to the [Helm documentation](https://helm.sh/) for instructions.

## Deploying the Workload

It is recommended to use `helm template` and pipe the result to `kubectl create` , rather than using `helm install`. Generally, a command looks as follows

```bash
helm template [your-release-name] ./helm | kubectl apply -f -
```

The chart provides three main ways to deploy models, detailed below.

## User Input Values

Refer to the `values.yaml` file for the user input values you can provide, along with instructions.

## Interacting with Deployed Model

### Verify Deployment

Check the deployment status:

```bash
kubectl get deployment
```

### Send prediction request

Follow the [demo_inference_service.ipynb](../examples/demo_inference_service.ipynb) notebook to see how to use the inference service.
