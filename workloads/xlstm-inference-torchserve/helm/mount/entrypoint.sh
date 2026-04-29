MOD_MODEL_NAME="${MODEL_NAME//\//_}"

DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y -qq \
    openjdk-17-jdk \
    zip \
    curl

echo "[entrypoint] Using TS_REPO_URL=${TS_REPO_URL}"
echo "[entrypoint] Cloning TorchServe repo into $TS_REPO_DIR"
git clone "$TS_REPO_URL" "$TS_REPO_DIR"

echo "[entrypoint] Removing content from requirements/torch_rocm62.txt"
cd "$TS_REPO_DIR"
: > requirements/torch_rocm62.txt

echo "[entrypoint] Installing xlstm specific requirements"
python -m pip install xlstm mlstm_kernels
python -m pip uninstall -y transformers
python -m pip install --upgrade --force-reinstall --no-deps 'transformers @ git+https://github.com/NX-AI/transformers.git@integrate_xlstm'
python -m pip install pygit2==1.17.0
python -m pip install 'accelerate>=0.26.0'
python3 -m pip install --upgrade --force-reinstall "tokenizers<0.21"

echo "[entrypoint] Installing TorchServe and dependencies"
echo "[entrypoint] Installing torch-model-archiver"
python3 -m pip install --no-cache-dir torch-model-archiver

echo "[entrypoint] Installing ROCm dependencies"
python3 ts_scripts/install_dependencies.py --rocm=rocm62

echo "[entrypoint] Building TorchServe from source"
python3 ts_scripts/install_from_src.py

# MINIO CLIENT INSTALLATION AND CONFIGURATION
# Download MinIO client binary to custom directory (no root access needed)
echo '[entrypoint] Installing MinIO client'
curl -s https://dl.min.io/client/mc/release/linux-amd64/mc \
    --create-dirs \
    -o /minio-binaries/mc
chmod +x /minio-binaries/mc
export PATH="${PATH}:/minio-binaries/"

# Configure MinIO client
mc alias set minio-host $BUCKET_STORAGE_HOST $BUCKET_ACCESS_KEY $BUCKET_SECRET_KEY

# MODEL DOWNLOAD FROM STORAGE
# Download model.zip from storage based on MODEL_NAME
echo "[entrypoint] Selected model: ${MODEL_NAME}"
echo "[entrypoint] Attempting to download ${MODEL_NAME} model from MinIO..."

if mc cp minio-host/$BUCKET_PATH/$MODEL_NAME/packaged/model.zip \
    ./model.zip; then
    echo "[entrypoint] SUCCESS: Downloaded model.zip for ${MODEL_NAME} successfully"
    echo "[entrypoint] File size: $(du -h ./model.zip | cut -f1)"
    echo "[entrypoint] File location: $(pwd)/model.zip"
else
    echo "[entrypoint] ERROR: Failed to download model.zip for ${MODEL_NAME} from storage"
    exit 1
fi


# MODEL ARCHIVING AND TORCHSERVE STARTUP
# Package handler into .mar
echo "[entrypoint] Archiving model with custom handler"
mkdir -p "$MODEL_STORE"
if torch-model-archiver \
  --model-name "$MOD_MODEL_NAME" \
  --version 1.0 \
  --export-path "$MODEL_STORE" \
  --handler /mount/xlstm_handler.py \
  --extra-files model.zip \
  --force; then
    echo "[entrypoint] SUCCESS: Model archived successfully"
    echo "[entrypoint] DEBUG: Archive contents: $(ls -la "$MODEL_STORE")"
else
    echo "[entrypoint] ERROR: torch-model-archiver failed"
    exit 1
fi

mkdir -p "$TEMP"

# Start TorchServe
echo "[entrypoint] Starting TorchServe"
if torchserve \
  --start \
  --ncs \
  --model-store "$MODEL_STORE" \
  --ts-config "$TS_CONFIG" \
  --models "${MOD_MODEL_NAME}.mar" \
  --disable-token-auth; then
    echo "[entrypoint] TorchServe started."
else
    echo "[entrypoint] TorchServe failed to start."
    exit 1
fi

while true; do sleep 10; done;
