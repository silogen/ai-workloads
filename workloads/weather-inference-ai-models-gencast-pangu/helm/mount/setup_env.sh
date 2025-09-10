#!/bin/bash
set -e

mkdir /workspace/models
cd /workspace/models

# Install graphcast from DeepMind
pip install git+https://github.com/deepmind/graphcast.git

# ECMWF AI model framework
git clone https://github.com/ecmwf-lab/ai-models.git
cp /workload/mount/rocm.patch /workspace/models/ai-models/
cd /workspace/models/ai-models && git apply rocm.patch
pip install /workspace/models/ai-models

# GenCast model
cd /workspace/models/
git clone https://github.com/ecmwf-lab/ai-models-gencast.git
pip install /workspace/models/ai-models-gencast

# Override downgraded haiku
pip install dm-haiku==0.0.13

# PanguWeather model
git clone https://github.com/ecmwf-lab/ai-models-panguweather.git
pip install /workspace/models/ai-models-panguweather
pip uninstall -y onnxruntime-gpu
pip3 install onnxruntime-rocm -f https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3.4/

# Tensorboard and profiling tools
pip install tensorflow==2.18.0 tensorboard-plugin-profile==2.18.0 importlib_resources etils
pip install boto3 botocore
pip install --upgrade ml_dtypes
