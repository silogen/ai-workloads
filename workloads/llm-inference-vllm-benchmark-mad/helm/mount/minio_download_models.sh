# model defined in ENV
if ! mc ls minio-host/${BUCKET_MODEL_PATH}/${MAD_MODEL}/ &>/dev/null; then
  echo "ERROR: Path 'minio-host/${BUCKET_MODEL_PATH}/${MAD_MODEL}/' not found. Check model path, bucket name, and permissions."
  echo "WARNING: Skipping this model and continuing with other models."
else
  mc cp --recursive minio-host/${BUCKET_MODEL_PATH}/${MAD_MODEL}/ $WORKPATH/models/${MAD_MODEL}/ || \
    echo "ERROR: Failed to download model from 'minio-host/${BUCKET_MODEL_PATH}/${MAD_MODEL}/'."
fi

# models defined in scenarios
awk -F, '{print $1}' $WORKPATH/mount/scenarios*.csv | sort | uniq | while read MODEL_ID; do
    if ! mc ls minio-host/${BUCKET_MODEL_PATH}/${MODEL_ID}/ &>/dev/null; then
      echo "ERROR: Path 'minio-host/${BUCKET_MODEL_PATH}/${MODEL_ID}/' not found. Check model path, bucket name, and permissions."
      echo "WARNING: Skipping this model and continuing with other models."
    else
      mc cp --recursive minio-host/${BUCKET_MODEL_PATH}/${MODEL_ID}/ $WORKPATH/models/${MODEL_ID}/ || \
        echo "ERROR: Failed to download model from 'minio-host/${BUCKET_MODEL_PATH}/${MODEL_ID}/'."
    fi
done
