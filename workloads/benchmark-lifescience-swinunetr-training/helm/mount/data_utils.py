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
# - Adaptation and transforms for training on the NSCLCL-Radiomics dataset
# - Dataloader optimizations

import logging
import math
from typing import List, Union

import numpy as np
import torch
from monai import data, transforms
from monai.apps import TciaDataset
from monai.apps.tcia import TCIA_LABEL_DICT
from utils import SafeLoadImaged

if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO)

IMAGE_DATA = "image"
LABEL_DATA = "seg"
SPACING_MODES = ("bilinear", "nearest")
ORIENTATION = "RAS"
COLLECTION = "NSCLC-Radiomics"
SEG_TYPE = "SEG"


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def print_tensor_shape(tensor, key_name="tensor"):
    """Simple function to print the shape and dtype of a tensor."""
    if isinstance(tensor, torch.Tensor):
        print(
            f"--- Debug Shape ({key_name}): {tensor.shape}, min/max val: {tensor.min()}/{tensor.max()} - dtype: {tensor.dtype} ---"
        )
    else:
        print(f"--- Debug Type ({key_name}): {type(tensor)} ---")  # If not a tensor
    return tensor


def select_channels(tensor: Union[torch.Tensor, np.ndarray], channel_indices: List[int]):
    """Simple function to select specific channel from the segmentation mask"""
    if not isinstance(tensor, (torch.Tensor, np.ndarray)):
        raise TypeError(f"Input 'tensor' must be a torch.Tensor or np.ndarray, got {type(tensor)}")
    return tensor[..., channel_indices]


def get_transforms(name: str, args):
    labels = args.dataset_labels.split(",")
    if len(labels) != args.out_channels:
        raise ValueError(
            f"Number of labels {len(labels)} doesn't match the number of out_channels {args.out_channels}: {labels}"
        )

    channel_indices = [TCIA_LABEL_DICT[COLLECTION][label] for label in labels]
    print(f"Predicting labels: {labels} - Channel: {channel_indices} (available labels: {TCIA_LABEL_DICT[COLLECTION]}")

    if name == "training":
        train_transform = transforms.Compose(
            [
                SafeLoadImaged(
                    reader="PydicomReader",
                    keys=[IMAGE_DATA, LABEL_DATA],
                    label_dict=TCIA_LABEL_DICT[COLLECTION],
                    image_only=False,
                    fname_regex=r".*\.dcm$",
                ),
                transforms.Lambdad(keys=LABEL_DATA, func=lambda x: select_channels(x, channel_indices=channel_indices)),
                transforms.EnsureChannelFirstd(keys=[IMAGE_DATA], channel_dim="no_channel"),
                transforms.EnsureChannelFirstd(keys=[LABEL_DATA], channel_dim=3),
                transforms.Orientationd(keys=[IMAGE_DATA, LABEL_DATA], axcodes=ORIENTATION),
                transforms.Spacingd(
                    keys=[IMAGE_DATA, LABEL_DATA], pixdim=(args.space_x, args.space_y, args.space_z), mode=SPACING_MODES
                ),
                transforms.ScaleIntensityRanged(
                    keys=[IMAGE_DATA], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
                ),
                transforms.CropForegroundd(keys=[IMAGE_DATA, LABEL_DATA], source_key=IMAGE_DATA, allow_smaller=False),
                transforms.SpatialPadd(
                    keys=[IMAGE_DATA, LABEL_DATA], spatial_size=(args.roi_x, args.roi_y, args.roi_z), method="symmetric"
                ),
                transforms.RandCropByPosNegLabeld(
                    keys=[IMAGE_DATA, LABEL_DATA],
                    label_key=LABEL_DATA,
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key=IMAGE_DATA,
                    image_threshold=0,
                ),
                transforms.RandFlipd(keys=[IMAGE_DATA, LABEL_DATA], prob=args.RandFlipd_prob, spatial_axis=0),
                transforms.RandFlipd(keys=[IMAGE_DATA, LABEL_DATA], prob=args.RandFlipd_prob, spatial_axis=1),
                transforms.RandFlipd(keys=[IMAGE_DATA, LABEL_DATA], prob=args.RandFlipd_prob, spatial_axis=2),
                transforms.RandRotate90d(keys=[IMAGE_DATA, LABEL_DATA], prob=args.RandRotate90d_prob, max_k=3),
                transforms.RandScaleIntensityd(keys=IMAGE_DATA, factors=0.1, prob=args.RandScaleIntensityd_prob),
                transforms.RandShiftIntensityd(keys=IMAGE_DATA, offsets=0.1, prob=args.RandShiftIntensityd_prob),
                transforms.ToTensord(keys=[IMAGE_DATA, LABEL_DATA]),
            ]
        )
        return train_transform

    elif name == "validation":
        val_transform = transforms.Compose(
            [
                SafeLoadImaged(
                    reader="PydicomReader",
                    keys=[IMAGE_DATA, LABEL_DATA],
                    label_dict=TCIA_LABEL_DICT[COLLECTION],
                    image_only=False,
                    fname_regex=r".*\.dcm$",
                ),
                transforms.Lambdad(keys=LABEL_DATA, func=lambda x: select_channels(x, channel_indices=channel_indices)),
                transforms.EnsureChannelFirstd(keys=[IMAGE_DATA], channel_dim="no_channel"),
                transforms.EnsureChannelFirstd(keys=[LABEL_DATA], channel_dim=3),
                transforms.Orientationd(keys=[IMAGE_DATA, LABEL_DATA], axcodes=ORIENTATION),
                transforms.Spacingd(
                    keys=[IMAGE_DATA, LABEL_DATA], pixdim=(args.space_x, args.space_y, args.space_z), mode=SPACING_MODES
                ),
                transforms.ScaleIntensityRanged(
                    keys=[IMAGE_DATA], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
                ),
                transforms.CropForegroundd(keys=[IMAGE_DATA, LABEL_DATA], source_key=IMAGE_DATA, allow_smaller=False),
                transforms.SpatialPadd(
                    keys=[IMAGE_DATA, LABEL_DATA], spatial_size=(args.roi_x, args.roi_y, args.roi_z), method="symmetric"
                ),
                transforms.ToTensord(keys=[IMAGE_DATA, LABEL_DATA]),
            ]
        )
        return val_transform

    elif name == "test":
        test_transform = transforms.Compose(
            [
                SafeLoadImaged(
                    reader="PydicomReader",
                    keys=[IMAGE_DATA, LABEL_DATA],
                    label_dict=TCIA_LABEL_DICT[COLLECTION],
                    image_only=False,
                    fname_regex=r".*\.dcm$",
                ),
                transforms.Lambdad(keys=LABEL_DATA, func=lambda x: select_channels(x, channel_indices=channel_indices)),
                transforms.EnsureChannelFirstd(keys=[IMAGE_DATA], channel_dim="no_channel"),
                transforms.EnsureChannelFirstd(keys=[LABEL_DATA], channel_dim=3),
                transforms.Orientationd(keys=[IMAGE_DATA, LABEL_DATA], axcodes=ORIENTATION),
                transforms.Spacingd(
                    keys=[IMAGE_DATA, LABEL_DATA],
                    pixdim=(args.space_x, args.space_y, args.space_z),
                    mode=SPACING_MODES[1],
                ),
                transforms.ScaleIntensityRanged(
                    keys=[IMAGE_DATA], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
                ),
                transforms.ToTensord(keys=[IMAGE_DATA, LABEL_DATA]),
            ]
        )
        return test_transform

    elif name == "inference":
        inference_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=[IMAGE_DATA]),
                transforms.EnsureChannelFirstd(keys=[IMAGE_DATA], channel_dim="no_channel"),
                transforms.Orientationd(keys=[IMAGE_DATA], axcodes=ORIENTATION),
                transforms.Spacingd(
                    keys=[IMAGE_DATA], pixdim=(args.space_x, args.space_y, args.space_z), mode=SPACING_MODES[0]
                ),
                transforms.ScaleIntensityRanged(
                    keys=[IMAGE_DATA], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
                ),
            ]
        )
        return inference_transform

    else:
        raise ValueError(f"Unknown transform with name {name}")


def get_loader(args):
    if args.test_mode:
        test_transform = get_transforms(name="test", args=args)

        test_ds = TciaDataset(
            root_dir=args.data_root_dir,
            collection=COLLECTION,
            section="validation",
            download=args.download_data,
            seg_type=SEG_TYPE,
            progress=True,
            cache_rate=0.0,
            transform=test_transform,
            runtime_cache=False,
        )
        test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=test_sampler,
            pin_memory=True,
            prefetch_factor=2,
        )
        loader = test_loader
    else:

        train_transform = get_transforms(name="training", args=args)
        train_ds = TciaDataset(
            root_dir=args.data_root_dir,
            collection=COLLECTION,
            section="training",
            download=args.download_data,
            seg_type=SEG_TYPE,
            progress=True,
            cache_num=24,
            cache_rate=1.0,
            val_frac=0.2,
            num_workers=args.workers,
            transform=train_transform,
            runtime_cache=False,
        )
        train_sampler = Sampler(train_ds) if args.distributed else None
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )

        val_transform = get_transforms(name="validation", args=args)
        val_ds = TciaDataset(
            root_dir=args.data_root_dir,
            collection=COLLECTION,
            section="validation",
            download=args.download_data,
            seg_type=SEG_TYPE,
            progress=True,
            cache_rate=0.0,
            val_frac=0.2,
            num_workers=args.workers,
            transform=val_transform,
            runtime_cache=False,
        )
        val_sampler = Sampler(val_ds, shuffle=True) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds,
            batch_size=1,
            shuffle=True,
            num_workers=args.workers,
            sampler=val_sampler,
            pin_memory=True,
            prefetch_factor=2,
        )
        loader = [train_loader, val_loader]

    return loader


def get_post_transforms_inverter(forward_transforms_obj):
    """Transform to revert all transforms previously applied."""
    return transforms.Invertd(
        keys=IMAGE_DATA,
        transform=forward_transforms_obj,
        orig_keys=IMAGE_DATA,
        orig_meta_keys=f"{IMAGE_DATA}_meta_dict",
        nearest_interp=True,
        to_tensor=True,
        device="cuda",
    )
