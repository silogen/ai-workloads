"""
MinIO Uploader

Uploads a local directory to a MinIO bucket, creating it if needed.

Usage:
    python minio_uploader.py <directory_path> <bucket_name> [prefix]
"""

import os
import sys

import boto3


def upload_directory_to_minio(directory, bucket_name, prefix=""):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("MINIO_SECRET_KEY"),
        endpoint_url=os.getenv("MINIO_ENDPOINT"),
    )

    try:
        s3.create_bucket(Bucket=bucket_name)
        print(f"Created bucket: {bucket_name}")
    except s3.exceptions.BucketAlreadyOwnedByYou:
        pass
    except s3.exceptions.BucketAlreadyExists:
        pass

    for root, _, files in os.walk(directory):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, directory)
            s3_key = os.path.join(prefix, relative_path) if prefix else relative_path
            try:
                s3.upload_file(local_path, bucket_name, s3_key)
                print(f"Uploaded: {local_path} -> {s3_key}")
            except Exception as e:
                print(f"Failed to upload {local_path}: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python minio_uploader.py <directory_path> <bucket_name> [prefix]")
        sys.exit(1)
    directory = sys.argv[1]
    bucket_name = sys.argv[2]
    prefix = sys.argv[3] if len(sys.argv) > 3 else ""
    upload_directory_to_minio(directory, bucket_name, prefix)
