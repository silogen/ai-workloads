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
# - Adaptations for training on the NSCLCL-Radiomics dataset
# - Parameter changes
# - Logging of standard output and error to file

import argparse
import json
import os
import sys
import time
from datetime import datetime as dt
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
from data_utils import get_loader
from lr_scheduler import LinearWarmupCosineAnnealingLR
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import Activations, AsDiscrete, Compose, EnsureType
from monai.utils.enums import MetricReduction
from trainer import run_training
from utils import Tee, download_dataset_from_minio, upload_directory_to_minio

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument(
    "--logdir",
    default="test",
    type=str,
    help=(
        "Directory to save tensorboard logs and other artifacts."
        "The files will be located under `./runs/<LOG-DIR>/<RUN-ID>/`, where <RUN-ID> is assigned at runtime based on the current time."
    ),
)
parser.add_argument(
    "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--data_root_dir", default="/workload_outputs/data", type=str, help="dataset root directory")
parser.add_argument("--download_data", action="store_true", help="Download the dataset")
parser.add_argument(
    "--dataset_labels",
    default="GTV-1",
    type=str,
    help="One or more comma-separated labels (segmentation mask) to predict from the dataset",
)
parser.add_argument(
    "--pretrained_model_name",
    default="swin_unetr.epoch.b4_5000ep_f48_lr2e-4_pretrained.pt",
    type=str,
    help="pretrained model name",
)
parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")
parser.add_argument("--max_epochs", default=5000, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--n_crops", default=1, type=int, help="number of crops")
parser.add_argument("--n_crops_val", default=4, type=int, help="number of crops validation")
parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--val_every", default=100, type=int, help="validation frequency")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument(
    "--local-rank", "--local_rank", "--rank", default=0, type=int, help="node rank for distributed training"
)
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization name")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=1, type=int, help="number of output channels")
parser.add_argument("--a_min", default=-1000.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=1000.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.0, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.0, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=3.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--use_ssl_pretrained", action="store_true", help="use self-supervised pretrained weights")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--loss_lambda_dice", default=1.5, type=float, help="lambda_dice value for the DiceCELoss")
parser.add_argument("--loss_lambda_ce", default=1, type=float, help="lambda_ce value for the DiceCELoss")
parser.add_argument("--loss_ce_class_weights", default=5, type=float, help="weight value for the DiceCELoss")
parser.add_argument("--squared_dice", action="store_true", help="use squared Dice")
parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
parser.add_argument("--save_results", action="store_true", help="save examples of validation results")
parser.add_argument("--profile", action="store_true", help="enable the torch profiler")
parser.add_argument(
    "--profiler_schedule",
    default="1,1,3,1",
    type=str,
    help="Comma separated values for the profiler wait, warmup, active, repeat",
)
parser.add_argument(
    "--dataset_bucket_name", required=True, type=str, help="Name of the storage bucket to download the dataset"
)
parser.add_argument(
    "--dataset_bucket_directory",
    required=True,
    type=str,
    help="Directory in the storage bucket to download the dataset",
)

parser.add_argument(
    "--storage_bucket_name", required=True, type=str, help="Name of the storage bucket to upload the training outputs"
)
parser.add_argument(
    "--storage_destination_directory",
    required=True,
    type=str,
    help="Destination directory in the storage bucket to upload the training outputs",
)


def main():
    args = parser.parse_args()
    args.amp = not args.noamp
    args.rank = args.local_rank
    args.logdir = os.path.join("/workload_outputs", args.logdir, dt.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(args.logdir, exist_ok=False)
    print(f"Log dir: {args.logdir}")

    # record input arguments
    with open(os.path.join(args.logdir, "args.json"), "wt") as fo:
        json.dump(vars(args), fo, indent=2)

    data_root_dir = args.data_root_dir
    Path(data_root_dir).mkdir(exist_ok=True, parents=True)
    print(f"Using data root dir: {data_root_dir} - Download: {args.download_data}")

    # Download dataset
    download_dataset_from_minio(args.dataset_bucket_name, args.dataset_bucket_directory, data_root_dir)

    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)


def main_worker(gpu, args):
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    tee_stdout = None
    log_file_handle = None

    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )

    if args.distributed:
        log_file_path = os.path.join(args.logdir, f"output_rank_{args.rank}.log")
    else:
        # Single process, single log file.
        log_file_path = os.path.join(args.logdir, "output.log")

    try:
        log_file_handle = open(log_file_path, "a", encoding="utf-8")  # 'a' for append
        tee_stdout = Tee(log_file_handle, original_stdout)
        tee_stderr = Tee(log_file_handle, original_stderr)  # Log stderr to the same file
        sys.stdout = tee_stdout
        sys.stderr = tee_stderr
        print(f"Worker {args.rank} (GPU {gpu}) logging to: {log_file_path}")

        torch.cuda.set_device(args.gpu)
        torch.backends.cudnn.benchmark = True
        args.test_mode = False
        loader = get_loader(args)
        print(args.rank, " gpu", args.gpu)
        if args.rank == 0:
            print("Batch size is:", args.batch_size, "epochs", args.max_epochs)
        inf_size = [args.roi_x, args.roi_y, args.roi_z]

        pretrained_dir = args.pretrained_dir

        model = SwinUNETR(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=args.dropout_path_rate,
            use_checkpoint=args.use_checkpoint,
        )

        if args.resume_ckpt:
            model_dict = torch.load(os.path.join(pretrained_dir, args.pretrained_model_name))["state_dict"]
            model.load_state_dict(model_dict)
            print("Use pretrained weights")

        if args.use_ssl_pretrained:
            try:
                model_dict = torch.load("pretrained_models/model_swinvit.pt")
                state_dict = model_dict["state_dict"]
                # fix potential differences in state dict keys from pre-training to
                # fine-tuning
                if "module." in list(state_dict.keys())[0]:
                    print("Tag 'module.' found in state dict - fixing!")
                    for key in list(state_dict.keys()):
                        state_dict[key.replace("module.", "")] = state_dict.pop(key)
                if "swin_vit" in list(state_dict.keys())[0]:
                    print("Tag 'swin_vit' found in state dict - fixing!")
                    for key in list(state_dict.keys()):
                        state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)
                # We now load model weights, setting param `strict` to False, i.e.:
                # this load the encoder weights (Swin-ViT, SSL pre-trained), but leaves
                # the decoder weights untouched (CNN UNet decoder).
                model.load_state_dict(state_dict, strict=False)
                print("Using pretrained self-supervised Swin UNETR backbone weights !")
            except ValueError:
                raise ValueError("Self-supervised pre-trained weights not available for" + str(args.model_name))

        if args.squared_dice:
            dice_loss = DiceCELoss(
                to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr
            )
        else:
            lambda_dice_weight = args.loss_lambda_dice
            lambda_ce_weight = args.loss_lambda_ce

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if args.out_channels > 1:
                ce_class_weights = torch.ones(args.out_channels, dtype=torch.float32, device=device)
                ce_class_weights[1:] = args.loss_ce_class_weights

                dice_loss = DiceCELoss(
                    sigmoid=True, lambda_dice=lambda_dice_weight, lambda_ce=lambda_ce_weight, weight=ce_class_weights
                )
            else:
                ce_class_weights = torch.tensor(args.loss_ce_class_weights, dtype=torch.float32, device=device)

                dice_loss = DiceCELoss(
                    softmax=True, lambda_dice=lambda_dice_weight, lambda_ce=lambda_ce_weight, weight=ce_class_weights
                )

        post_label = EnsureType()
        post_pred = Compose(
            [Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
        )  # TODO: confirm which post_label and post_pred needs to be used

        dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True)

        model_inferer = partial(
            sliding_window_inference,
            roi_size=inf_size,
            sw_batch_size=args.sw_batch_size,
            predictor=model,
            overlap=args.infer_overlap,
        )

        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total parameters count", pytorch_total_params)

        best_acc = 0
        start_epoch = 0

        if args.checkpoint is not None:
            checkpoint = torch.load(args.checkpoint, map_location="cpu")
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in checkpoint["state_dict"].items():
                new_state_dict[k.replace("backbone.", "")] = v
            model.load_state_dict(new_state_dict, strict=False)
            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"]
            if "best_acc" in checkpoint:
                best_acc = checkpoint["best_acc"]
            print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))

        model.cuda(args.gpu)

        if args.distributed:
            torch.cuda.set_device(args.gpu)
            if args.norm_name == "batch":
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)
        if args.optim_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
        elif args.optim_name == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
        elif args.optim_name == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=args.optim_lr,
                momentum=args.momentum,
                nesterov=True,
                weight_decay=args.reg_weight,
            )
        else:
            raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

        if args.lrschedule == "warmup_cosine":
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
            )
        elif args.lrschedule == "cosine_anneal":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
            if args.checkpoint is not None:
                scheduler.step(epoch=start_epoch)
        else:
            scheduler = None

        accuracy = run_training(
            model=model,
            train_loader=loader[0],
            val_loader=loader[1],
            optimizer=optimizer,
            loss_func=dice_loss,
            acc_func=dice_acc,
            args=args,
            model_inferer=model_inferer,
            scheduler=scheduler,
            start_epoch=start_epoch,
            post_label=post_label,
            post_pred=post_pred,
        )

        upload_directory_to_minio(
            bucket_name=args.storage_bucket_name,
            local_directory=args.logdir,
            destination_directory=args.storage_destination_directory,
        )

        return accuracy

    except Exception as e:
        print(f"!!!!!!!!!! ERROR IN WORKER RANK {args.rank} (GPU {gpu}) !!!!!!!!!!")
        print(e)
        import traceback

        print(traceback.format_exc())
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if tee_stdout:
            tee_stdout.close()
        elif log_file_handle:
            log_file_handle.close()


if __name__ == "__main__":
    main()
