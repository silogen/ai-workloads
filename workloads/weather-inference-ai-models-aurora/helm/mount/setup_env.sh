#!/bin/bash
set -e

mkdir -p /workspace/models
cd /workspace/models

# ECMWF AI model framework
git clone https://github.com/ecmwf-lab/ai-models.git
pip install /workspace/models/ai-models

# Aurora model
git clone https://github.com/ecmwf-lab/ai-models-aurora.git
pip install /workspace/models/ai-models-aurora


# Install dependencies for minio uploader
pip install boto3 botocore
pip install matplotlib==3.10.3
pip install --upgrade multiurl==0.3.1
