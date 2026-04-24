# LeRobot Imitation Learning Fine-tuning

This Helm Chart deploys [Kubernetes](https://kubernetes.io/docs/home/) fine-tuning job for robot learning using [LeRobot](https://github.com/huggingface/lerobot), a state-of-the-art library for real-world robotics in PyTorch on AMD GPU and [ROCm](https://rocm.docs.amd.com/en/latest/). LeRobot provides efficient implementations of imitation learning and reinforcement learning algorithms specifically designed for robotics applications, while AMD GPU and [ROCm](https://rocm.docs.amd.com/en/latest/) provide efficient way of running finetuning jobs.

## Overview

LeRobot enables end-to-end training of policies that map visual observations and robot states to robot actions using imitation learning. There are multiple policies available for finetuning through LeRobot library, e.g., [ACT](https://arxiv.org/abs/2304.13705), [Pi0.5](https://github.com/Physical-Intelligence/openpi), [SmolVLA](https://huggingface.co/papers/2506.01844), [Gr00t](https://github.com/NVIDIA/Isaac-GR00T) etc. You can explore full list in the [LeRobot github](https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies). Each algorithm has different hyperparameters and architectural choices. Refer to the [LeRobot documentation](https://huggingface.co/docs/lerobot/index) for algorithm-specific details.

For more information, see:
- [LeRobot Documentation](https://huggingface.co/docs/lerobot/index)
- [Imitation Learning Tutorial](https://huggingface.co/docs/lerobot/il_robots)
- [LeRobot GitHub Repository](https://github.com/huggingface/lerobot)

## Typical Workflow

The complete robot learning workflow consists of four stages:

1. **Robot assembly and calibration**: Imitation learning typically involves assembling a pair of leader and follower arms and calibrating them so that the movements of the leader arm can be accurately followed by the follower arm. See e.g. instructions for assembling and calibrating [SO101 arm set](https://huggingface.co/docs/lerobot/en/so101)

2. **Data Collection**: Record robot demonstrations using teleoperation or other methods. This step is done outside of this workload using LeRobot's data collection tools.

3. **Policy Training** (this workload): Fine-tune a policy model on the collected demonstration data using chosen imitation learning algorithm from the [available set](https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies).

4. **Policy Deployment**: Deploy the trained policy for inference on a real robot or in simulation. This step is done outside of this workload using LeRobot's evaluation and deployment tools, see e.g. [ACT tutorial](https://huggingface.co/docs/lerobot/en/act)

This Helm chart handles the finetuning stage, taking pre-collected demonstration datasets and producing trained policy checkpoints.

## Configuration

The training job is configured through the `values.yaml` file. Refer to the `values.yaml` file for the complete list of configuration options.

Some of the important points are related to setting up Weight&Biases (for tracking the progress) and Huggingface tokens (for uploading finetuned checkpoints and potentially downloading pretrained models from gated repos). To set them properly one has to give working values to the `.envVars.WANDB_API_KEY` and `envVars.HF_TOKEN`. By default `WANDB_API_KEY` env variable is set from the `wandb-token` secret with `wandb-token` key and `HF_TOKEN` is is set from the `hf-token` secret with `hf-token` key:

```
envVars:
  WANDB_API_KEY:
    name: wandb-token
    key: wandb-token
  HF_TOKEN:
    name: hf-token
    key: hf-token
```

## Dataset Specification

This workload supports datasets from HuggingFace Hub. Datasets should be in LeRobot's expected format with episodes containing observations (images, states) and actions.

## Running the Workload

The simplest way to run the training job is to use `helm template` and pipe the result to `kubectl create`.

Example command:

```bash
helm template workloads/robotics-finetune-lerobot/helm \
  --values workloads/robotics-finetune-lerobot/helm/overrides/pi05-kettle.yaml \
  --name-template pi05-kettle \
  --set hfFinetunedModelId=<HF model id to upload final checkpoint> \
  | kubectl create -f -
```

### Monitoring Training Progress

To view training logs:

```bash
kubectl logs -f job/lerobot-finetune-job-pi05-kettle
```

Training checkpoints are automatically uploaded to HuggingFace Hub in the end of the job to a specified repo id in `hfFinetunedModelId`. Make sure you specify HF token that has write access.

## Cleanup

After training is complete, delete the Kubernetes resources:

```bash
helm template workloads/robotics-finetune-lerobot/helm \
  --values workloads/robotics-finetune-lerobot/helm/overrides/pi05-kettle.yaml \
  --name-template pi05-kettle \
  | kubectl delete -f -
```
