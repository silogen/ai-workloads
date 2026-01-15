#!/bin/bash
set -e

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PIP_NO_CACHE_DIR=off
export PIP_DISABLE_PIP_VERSION_CHECK=on
export CONDA_ENV_PATH=/opt/conda/envs/py_3.10
export PATH="$CONDA_ENV_PATH/bin:$PATH"

TS_COMMIT_ID=62c4d6a1fdc1d071dbcf758ebd756029af20bd5e
DIFFUSERS_COMMIT=20e4b6a628c7e433f5805de49afc28f991c185c0

# Copy files from readonly ConfigMap to writable location
mkdir -p /workspace/wan21-torchserve
cp /workload/mount/entrypoint.sh /workspace/
find /workload/mount -maxdepth 1 -xtype f ! -name 'entrypoint.sh' -exec cp {} /workspace/wan21-torchserve/ \;

cd /workspace

# Install Java and zip (needed by TorchServe)
apt-get update -qq && \
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openjdk-17-jdk zip curl

# Download MinIO client binary to custom directory (no root access needed)
echo 'Installing MinIO client'
curl -s https://dl.min.io/client/mc/release/linux-amd64/mc \
    --create-dirs \
    -o /minio-binaries/mc
chmod +x /minio-binaries/mc
export PATH="${PATH}:/minio-binaries/"

# Configure MinIO client
mc alias set minio-host $BUCKET_STORAGE_HOST $BUCKET_ACCESS_KEY $BUCKET_SECRET_KEY

python -m pip freeze
# Install Python requirements
python -m pip install -r ./wan21-torchserve/requirements-torchserve.txt
python -m pip install git+https://github.com/huggingface/diffusers@$DIFFUSERS_COMMIT
python -m pip install "numpy==1.26.4"

# first check if serve directory exists (from previous crash/fail)
if [ -d "serve" ]; then
    echo "Directory 'serve' already exists. Removing it..."
    rm -rf serve
fi

# Clone TorchServe
git clone https://github.com/pytorch/serve.git
cd serve
git checkout $TS_COMMIT_ID
# Rewrite/create empty requirements file for rocm62
: > requirements/torch_rocm62.txt
python -m pip install torch-model-archiver
python ./ts_scripts/install_dependencies.py --rocm=rocm62
python ./ts_scripts/install_from_src.py
# SetUp for torchserve:
cd /workspace/wan21-torchserve

echo "Running model setup script..."
bash model_setup.sh
echo "Model archive created successfully."

echo "Starting TorchServe..."
if torchserve --start \
  --ncs \
  --model-store model_store \
  --models wan21.mar \
  --disable-token-auth \
  --ts-config config.properties; then

  echo "TorchServe started."
  touch /workload/healthy
else
  echo "TorchServe failed to start." >&2
  exit 1
fi

# Keep alive
while true; do sleep 10; done
