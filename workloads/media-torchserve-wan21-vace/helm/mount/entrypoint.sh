#!/bin/bash
set -e

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PIP_NO_CACHE_DIR=off
export PIP_DISABLE_PIP_VERSION_CHECK=on
export CONDA_ENV_PATH=/opt/conda/envs/py_3.10
export PATH="$CONDA_ENV_PATH/bin:$PATH"

# Configuration
TS_REPO_URL="https://github.com/pytorch/serve.git"
TS_REPO_DIR="/workspace/serve"
VACE_REPO_URL="https://github.com/ali-vilab/VACE.git"
VACE_REPO_DIR="/workspace/VACE"
MODEL_STORE="/workspace/model-store"
TEMP="/workspace/tmp"
TS_CONFIG="/workload/mount/config.properties"

# Copy files from readonly ConfigMap to writable location
mkdir -p /workspace/vace-torchserve
cp /workload/mount/entrypoint.sh /workspace/
find /workload/mount -maxdepth 1 -xtype f ! -name 'entrypoint.sh' -exec cp {} /workspace/vace-torchserve/ \;

cd /workspace

# SYSTEM DEPENDENCIES INSTALLATION
apt-get update -qq && \
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    openjdk-17-jdk \
    zip \
    curl \
    git-lfs

pip install "huggingface_hub[cli]"

# Download MinIO client binary to custom directory (no root access needed)
echo 'Installing MinIO client'
curl -s https://dl.min.io/client/mc/release/linux-amd64/mc \
    --create-dirs \
    -o /minio-binaries/mc
chmod +x /minio-binaries/mc
export PATH="${PATH}:/minio-binaries/"

# Configure MinIO client
mc alias set minio-host $BUCKET_STORAGE_HOST $BUCKET_ACCESS_KEY $BUCKET_SECRET_KEY

# TORCHSERVE REPOSITORY SETUP
echo "[entrypoint] Cloning TorchServe repo into $TS_REPO_DIR"
git clone "$TS_REPO_URL" "$TS_REPO_DIR"
cd "$TS_REPO_DIR"
: > requirements/torch_rocm62.txt

# VACE REPOSITORY SETUP AND DEPENDENCIES
echo "[entrypoint] Cloning VACE repo into $VACE_REPO_DIR"
git clone "$VACE_REPO_URL" "$VACE_REPO_DIR"
echo "[entrypoint] Installing VACE requirements"
cd "$VACE_REPO_DIR"
pip install -r requirements.txt
pip install wan@git+https://github.com/Wan-Video/Wan2.1
pip install ltx-video@git+https://github.com/Lightricks/LTX-Video@ltx-video-0.9.1 sentencepiece --no-deps

echo "[entrypoint] Downloading vace model and annotators for preprocessing"
pip install -r requirements/annotator.txt
git lfs install
git clone https://huggingface.co/ali-vilab/VACE-Annotators "$VACE_REPO_DIR"/models/VACE-Annotators
echo "[entrypoint] VACE ANNOTATORS CLONED"

# TORCHSERVE INSTALLATION AND SETUP
cd "$TS_REPO_DIR"
echo "[entrypoint] Installing TorchServe and dependencies"
python -m pip install torch-model-archiver
python -m pip install pygit2
python ts_scripts/install_dependencies.py --rocm=rocm62
python ts_scripts/install_from_src.py

# Update PYTHONPATH to include VACE repo
export PYTHONPATH="${VACE_REPO_DIR}/vace:${PYTHONPATH}"
export VACE_REPO_DIR="${VACE_REPO_DIR}"

# MODEL DOWNLOAD FROM STORAGE
echo "[entrypoint] Selected model: ${MODEL_NAME}"
echo "[entrypoint] Attempting to download ${MODEL_NAME} model from MinIO..."

MODEL_S3_NAME="Wan-AI/${MODEL_NAME}-diffusers"

if mc cp minio-host/$BUCKET_PATH/$MODEL_S3_NAME/packaged/model.zip \
    ./model.zip; then
    echo "[entrypoint] SUCCESS: Downloaded model.zip for ${MODEL_NAME} successfully"
    echo "[entrypoint] File size: $(du -h ./model.zip | cut -f1)"
else
    echo "[entrypoint] ERROR: Failed to download model.zip for ${MODEL_NAME} from storage"
    exit 1
fi

# MODEL ARCHIVING AND TORCHSERVE STARTUP
echo "[entrypoint] Archiving model with custom handler"
mkdir -p "$MODEL_STORE"
if torch-model-archiver \
  --model-name "$MODEL_NAME" \
  --version 1.0 \
  --export-path "$MODEL_STORE" \
  --handler /workspace/vace-torchserve/vace_handler.py \
  --extra-files model.zip \
  --force; then
    echo "[entrypoint] SUCCESS: Model archived successfully"
else
    echo "[entrypoint] ERROR: torch-model-archiver failed"
    exit 1
fi

# Start TorchServe
echo "[entrypoint] Starting TorchServe..."
mkdir -p "$TEMP"
if torchserve --start \
  --ncs \
  --model-store "$MODEL_STORE" \
  --models "${MODEL_NAME}.mar" \
  --disable-token-auth \
  --ts-config "$TS_CONFIG"; then

  echo "TorchServe started."
  touch /workload/healthy
else
  echo "TorchServe failed to start." >&2
  exit 1
fi

# Keep alive
while true; do sleep 10; done
