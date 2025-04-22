# Axolotl ROCm image

[Axolotl](https://github.com/axolotl-ai-cloud/axolotl) is an LLM continued pretraining and finetuning framework. This
Dockerfile builds an image that can run axolotl on the ROCm platform.

## Building

Currently, the build requires you to have access to the image
`ghcr.io/silogen/rocm6.2-vllm0.6.3-flash-attn2.6.3-wheels:static`, which is private.
Once you have access, to build the image, simply run
```bash
docker build -f axolotl-rocm.Dockerfile -t ghcr.io/silogen/rocm-silogen-axolotl-worker:YOUR_VERSION .
```

## Missing functionality

Currently, certain dependencies are not installed due to incompatibility issues:
- bitsandbytes, which is built and installed, but currently has some unsolved issues. It is installed so that axolotl
  launch works, but avoid functionality that actually uses bitsandbytes.
- mamba-ssm, which should be possible to run on ROCm with a patch
- nvidia-ml-py, as Nvidia management functionality is not used on ROCm
