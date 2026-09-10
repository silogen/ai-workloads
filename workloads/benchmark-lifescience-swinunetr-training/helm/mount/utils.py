# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by Reference Models team (AMD) on 2025:
# - Added MinIO utility functions: download_dataset_from_minio and upload_directory_to_minio
# - Added class Tee to log standard output and error to file
# - Added class SafeLoadImaged to fix the error when ussing num_workers > 1 in the TciaDataset

import os
import subprocess
from pathlib import Path
from typing import Union

import boto3
import botocore
import numpy as np
import scipy.ndimage as ndimage
import torch
from monai.transforms import LoadImaged


class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

    def close(self):
        for f in self.files:
            # Avoid closing the original sys.stdout/stderr if they were passed
            if hasattr(f, "fileno") and f.fileno() > 2:  # Heuristic: 0=stdin, 1=stdout, 2=stderr
                try:
                    f.close()
                except Exception:
                    pass


class SafeLoadImaged:
    """Simple wrapper for the LoadImaged that doesn't do anything else.
    It somehow fixes the error when ussing num_workers > 1 in the TciaDataset. No idea why, it might be changing
     how the transforms are constructed and pickled, altering the process/thread state.
    """

    def __init__(self, **kwargs):
        self.loader = LoadImaged(**kwargs)

    def __call__(self, data):
        try:
            return self.loader(data)
        except Exception as e:
            print(f"Failed to load {data}: {e}")
            raise


def resample_3d(img, target_size):
    imx, imy, imz = img.shape
    tx, ty, tz = target_size
    zoom_ratio = (float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
    img_resampled = ndimage.zoom(img, zoom_ratio, order=0, prefilter=False)
    return img_resampled


def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def distributed_all_gather(
    tensor_list, valid_batch_size=None, out_numpy=False, world_size=None, no_barrier=False, is_valid=None
):
    if world_size is None:
        world_size = torch.distributed.get_world_size()
    if valid_batch_size is not None:
        valid_batch_size = min(valid_batch_size, world_size)
    elif is_valid is not None:
        is_valid = torch.tensor(bool(is_valid), dtype=torch.bool, device=tensor_list[0].device)
    if not no_barrier:
        torch.distributed.barrier()
    tensor_list_out = []
    with torch.no_grad():
        if is_valid is not None:
            is_valid_list = [torch.zeros_like(is_valid) for _ in range(world_size)]
            torch.distributed.all_gather(is_valid_list, is_valid)
            is_valid = [x.item() for x in is_valid_list]
        for tensor in tensor_list:
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(gather_list, tensor)
            if valid_batch_size is not None:
                gather_list = gather_list[:valid_batch_size]
            elif is_valid is not None:
                gather_list = [g for g, v in zip(gather_list, is_valid_list) if v]
            if out_numpy:
                gather_list = [t.cpu().numpy() for t in gather_list]
            tensor_list_out.append(gather_list)
    return tensor_list_out


def upload_directory_to_minio(bucket_name: str, local_directory: Union[str, Path], destination_directory: str = ""):
    """
    Uploads a whole directory recursively to a MinIO bucket, preserving the
    top-level folder and all of its sub-folder structure.
    """
    print(f"Uploading directory {local_directory} to {bucket_name}/{destination_directory}...")

    endpoint_host = os.environ["BUCKET_STORAGE_HOST"]
    if not endpoint_host.startswith(("http://", "https://")):
        endpoint_url = f"https://{endpoint_host}"
    else:
        endpoint_url = endpoint_host

    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.environ["BUCKET_STORAGE_ACCESS_KEY"],
        aws_secret_access_key=os.environ["BUCKET_STORAGE_SECRET_KEY"],
        endpoint_url=endpoint_url,
        verify="/etc/ssl/certs/tls-ca-bundle.pem",
    )

    local_directory = Path(local_directory)
    if not local_directory.is_dir():
        print(f"Error: {local_directory} is not a directory.")
        return

    for local_path in local_directory.rglob("*"):

        if local_path.is_file():
            # Get the path of the file relative to the starting directory to preserve
            # the sub-folder structure.
            relative_path = local_path.relative_to(local_directory)
            source_dir_name = local_directory.name

            # Construct the final object key
            if destination_directory:
                object_key = str(Path(destination_directory) / source_dir_name / relative_path)
            else:
                object_key = str(Path(source_dir_name) / relative_path)

            try:
                s3.upload_file(str(local_path), bucket_name, object_key)
                print(f"File {local_path} uploaded successfully to {bucket_name}/{object_key}.")
            except botocore.exceptions.ClientError as e:
                print(f"Error uploading file {local_path}: {e}")


def download_dataset_from_minio(bucket: str, bucket_directory: str, data_root_dir: str):
    print(f"Downloading dataset from {bucket}/{bucket_directory} to {data_root_dir}")

    # Ensure local directory exists
    os.makedirs(data_root_dir, exist_ok=True)

    # Use existing minio-host alias
    source_path = f"minio-host/{bucket}/{bucket_directory}"

    try:
        result = subprocess.run(
            ["mc", "cp", "--recursive", source_path, f"{data_root_dir}/NSCLC-Radiomics"], check=True, text=True
        )

        print("✓ Download completed successfully!")

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Download failed: {e}")
