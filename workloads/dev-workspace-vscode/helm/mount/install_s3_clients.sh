#!/bin/bash

echo '============================================'
echo 'Installing S3 Storage Clients (AWS CLI, MinIO, rclone)'
echo '============================================'

# Function to install AWS CLI and boto3
install_aws_cli() {
    echo '--------------------------------------------'
    echo 'Installing AWS CLI and boto3'
    echo '--------------------------------------------'

    # Download AWS cli binary
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" --create-dirs -o "/workload/bin/awscliv2.zip"
    unzip "/workload/bin/awscliv2.zip" -d /workload/bin/
    /workload/bin/aws/install --bin-dir /usr/local/bin --install-dir /usr/local/aws-cli

    # Configure AWS CLI with MinIO S3 settings
    mkdir -p ~/.aws
    cat > ~/.aws/config <<EOF
[profile default]
aws_access_key_id = ${BUCKET_STORAGE_ACCESS_KEY}
aws_secret_access_key = ${BUCKET_STORAGE_SECRET_KEY}
region = us-east-1
services = minio-s3

[services minio-s3]
s3 =
  endpoint_url = ${BUCKET_STORAGE_HOST}
EOF

    echo 'AWS CLI installed and configured!'
    pip install boto3
    echo 'boto3 installed!'
}

# Function to install MinIO Client
install_minio_client() {
    echo '--------------------------------------------'
    echo 'Installing MinIO Client'
    echo '--------------------------------------------'

    # Download MinIO client binary
    curl -s https://dl.min.io/client/mc/release/linux-amd64/mc \
        --create-dirs \
        -o /usr/local/bin/mc

    # Make the binary executable
    chmod +x /usr/local/bin/mc

    # Configure MinIO alias
    mc alias set minio "${BUCKET_STORAGE_HOST}" "${BUCKET_STORAGE_ACCESS_KEY}" "${BUCKET_STORAGE_SECRET_KEY}"

    echo 'MinIO Client installed and configured!'
}

# Function to install rclone
install_rclone() {
    echo '--------------------------------------------'
    echo 'Installing rclone'
    echo '--------------------------------------------'

    # Install rclone via apt
    apt-get update && apt-get install rclone -y

    # Configure rclone with MinIO S3 settings
    mkdir -p ~/.config/rclone
    cat > ~/.config/rclone/rclone.conf <<EOF
[minio]
type = s3
provider = Minio
env_auth = false
access_key_id = ${BUCKET_STORAGE_ACCESS_KEY}
secret_access_key = ${BUCKET_STORAGE_SECRET_KEY}
region = us-east-1
endpoint = ${BUCKET_STORAGE_HOST}
location_constraint =
server_side_encryption =
EOF

    echo 'rclone installed and configured!'
}

# Install all clients
install_aws_cli
install_minio_client
install_rclone

echo '============================================'
echo 'All S3 Storage Clients Installed Successfully!'
echo '============================================'

# Display comprehensive usage instructions
cat << 'EOF'
--------------------------------------------
USAGE INSTRUCTIONS:
--------------------------------------------

MinIO Client (mc) Commands:
    - List configured aliases:    mc alias list
    - List available buckets:     mc ls minio
    - Copy files from bucket:     mc cp minio/<bucket>/prefix <destination>
    - Upload files to bucket:     mc cp <source> minio/<bucket>/prefix
    - Remove files from bucket:   mc rm minio/<bucket>/prefix
    - Create a new bucket:        mc mb minio/<bucket>
    - Get help:                   mc --help

rclone Commands:
    - List configured remotes:    rclone listremotes
    - List available buckets:     rclone lsd minio:
    - List files in bucket:       rclone ls minio:<bucket>
    - Copy files from bucket:     rclone copy minio:<bucket>/prefix <destination>
    - Upload files to bucket:     rclone copy <source> minio:<bucket>/prefix
    - Remove files from bucket:   rclone delete minio:<bucket>/prefix
    - Create a new bucket:        rclone mkdir minio:<bucket>
    - Get help:                   rclone help

AWS CLI Commands:
    - List configured profiles:   aws configure list-profiles
    - List available buckets:     aws s3 ls
    - Copy files from bucket:     aws s3 cp s3://<bucket>/prefix <destination>
    - Upload files to bucket:     aws s3 cp <source> s3://<bucket>/prefix
    - Remove files from bucket:   aws s3 rm s3://<bucket>/prefix
    - Create a new bucket:        aws s3 mb s3://<bucket>
    - Get help:                   aws s3 help

boto3 Python Examples:
    import boto3
    s3 = boto3.Session(profile_name="default").client('s3')  # Create S3 client with MinIO endpoint
    buckets = s3.list_buckets()  # List buckets
    objects = s3.list_objects_v2(Bucket='default-bucket')  # List objects in bucket named "default-bucket"

--------------------------------------------
EOF
