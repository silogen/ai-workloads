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
# - Profiling support

import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.parallel
import torch.profiler
import torch.utils.data.distributed
from data_utils import IMAGE_DATA, LABEL_DATA
from monai.data import decollate_batch
from torch.cuda.amp import GradScaler, autocast
from torch.profiler import record_function
from torch.utils.tensorboard import SummaryWriter
from utils import AverageMeter, distributed_all_gather


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    loader_iter = iter(loader)
    idx = 0

    with record_function("__sec:loader_first"):
        batch_data = next(loader_iter, None)

    while batch_data:
        with record_function("__sec:step"):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data[IMAGE_DATA], batch_data[LABEL_DATA]
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            for param in model.parameters():
                param.grad = None

            with autocast(enabled=args.amp, dtype=torch.bfloat16):
                with record_function("__sec:forward"):
                    model_forward = model(data)
                with record_function("__sec:loss"):
                    loss = loss_func(model_forward, target)

            with record_function("__sec:backward"):
                if args.amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            if args.distributed:
                with record_function("__sec:loss_gather"):
                    loss_list = distributed_all_gather(
                        [loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length
                    )
                    run_loss.update(
                        np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0),
                        n=args.batch_size * args.world_size,
                    )
            else:
                run_loss.update(loss.item(), n=args.batch_size)

            if args.rank == 0:
                print(
                    "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    "loss: {:.4f}".format(run_loss.avg),
                    "time {:.4f}s".format(time.time() - start_time),
                )
            start_time = time.time()

            idx += 1
            with record_function("__sec:loader_next"):
                batch_data = next(loader_iter, None)

    for param in model.parameters():
        param.grad = None

    return run_loss.avg


def val_epoch(model, loader, epoch, acc_func, args, model_inferer=None, post_label=None, post_pred=None, writer=None):
    model.eval()
    run_acc = AverageMeter()
    start_time = time.time()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data[IMAGE_DATA], batch_data[LABEL_DATA]
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)

            if idx == 0 and args.rank == 0 and writer is not None:
                try:
                    save_dir = args.logdir
                    os.makedirs(save_dir, exist_ok=True)

                    data_cpu = data.detach().cpu()
                    target_cpu = target.detach().cpu()
                    logits_cpu = logits.detach().cpu()
                    print(f"Validation input shape: {data_cpu.shape}")

                    if data_cpu.ndim != 5:
                        print(
                            f"Warning: Expected 5D data (B, C, H, W, D), but got {data_cpu.ndim}D. Skipping image logging."
                        )
                    else:
                        if args.out_channels != 1:
                            print(
                                "Warning: logging of images to tensorboard is only implemented for args.out_channels=1. Only the first channel will be logged."
                            )

                        if args.save_results:
                            # --- Numpy volumne saving ---
                            # 1. Image Volume (Channel 0)
                            image_volume = data_cpu[0, 0].numpy()  # Shape [H, W, D]
                            img_filename = os.path.join(save_dir, f"epoch_{epoch}_image.npy")
                            np.save(img_filename, image_volume)

                            # 2. Label Volume (Channel 0)
                            label_volume = target_cpu[0, 0].numpy()
                            lbl_filename = os.path.join(save_dir, f"epoch_{epoch}_label.npy")
                            np.save(lbl_filename, label_volume)

                            # 3. Prediction Volume
                            # Apply sigmoid and threshold to the whole volume's logits
                            logits_sample = logits_cpu[0, 0]
                            probs_volume = torch.sigmoid(logits_sample)
                            pred_volume = (probs_volume > 0.5).float().numpy()
                            pred_filename = os.path.join(save_dir, f"epoch_{epoch}_prediction.npy")
                            np.save(pred_filename, pred_volume)

                            print(f"Saved validation sample volumes for epoch {epoch} to {save_dir}")

                        # --- Tensorboard saving ---

                        # Get middle slice index
                        slice_idx = data_cpu.shape[-1] // 2

                        print(f"Log Check - Data shape: {data_cpu.shape}")
                        print(f"Log Check - Target shape: {target_cpu.shape}")
                        print(f"Log Check - Logits shape: {logits_cpu.shape}")
                        print(f"Log Check - Logits min/max: {logits_cpu.min()} / {logits_cpu.max()}")
                        # Check if guidance channel exists
                        if data_cpu.shape[1] > 1:
                            print(f"Log Check - Guidance min/max: {data_cpu[0, 1].min()} / {data_cpu[0, 1].max()}")

                        img_slice = data_cpu[0, 0, :, :, slice_idx].float().unsqueeze(0)
                        label_slice = target_cpu[0, 0, :, :, slice_idx].float().unsqueeze(0)

                        # 1. Select the single logit channel for the slice
                        logit_slice = logits_cpu[0, 0, :, :, slice_idx].float()
                        # 2. Apply Sigmoid to get probabilities
                        prob_slice = torch.sigmoid(logit_slice)
                        # 3. Threshold probabilities to get binary prediction
                        pred_slice = (prob_slice > 0.5).float().unsqueeze(0)

                        # Normalize Image/Guidance for Display
                        img_min = img_slice.min()
                        img_max = img_slice.max()
                        img_slice = (img_slice - img_min) / (img_max - img_min + 1e-6)

                        # Convert to Tensor (if MetaTensor)
                        img_slice_tensor = img_slice.as_tensor()
                        label_slice_tensor = label_slice.as_tensor()
                        pred_slice_tensor = pred_slice

                        # Add images to TensorBoard
                        writer.add_image("Validation/Input_Slice", img_slice_tensor, epoch, dataformats="CHW")
                        writer.add_image("Validation/Target_Slice", label_slice_tensor, epoch, dataformats="CHW")
                        writer.add_image("Validation/Prediction_Slice", pred_slice_tensor, epoch, dataformats="CHW")
                        print(
                            f"Logged validation images for epoch {epoch}, batch {idx}, slice {slice_idx} to {save_dir}"
                        )

                except Exception as e:
                    import traceback

                    print(f"Warning: Could not log validation images for epoch {epoch}. Error: {e}")
                    print(traceback.format_exc())

            if not logits.is_cuda:
                target = target.cpu()

            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc, not_nans = acc_func.aggregate()
            acc = acc.cuda(args.rank)

            if args.distributed:
                acc_list, not_nans_list = distributed_all_gather(
                    [acc, not_nans], out_numpy=True, is_valid=idx < loader.sampler.valid_length
                )
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(al, n=nl)

            else:
                run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            if args.rank == 0:
                avg_acc = np.mean(run_acc.avg)
                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    "acc",
                    avg_acc,
                    "time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()
    return run_acc.avg


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def start_profiler(args) -> torch.profiler.profile | None:

    profiler_log_dir = os.path.join(args.logdir, "profiler")
    Path(profiler_log_dir).mkdir(parents=True, exist_ok=True)

    wait, warmup, active, repeat = (int(i) for i in args.profiler_schedule.split(","))

    # tb_trace_handler = torch.profiler.tensorboard_trace_handler(
    #     profiler_log_dir, use_gzip=True
    # )

    def on_trace_ready(prof: torch.profiler.profile):

        # for simplicity we'll disable memory recording after the first trace is ready
        # which should capture the majority of use cases.
        # note that everything up to this point will be capture (up to `max_entries` events),
        # not just the profiler's active steps.
        memory_path = os.path.join(profiler_log_dir, "memory.pickle")
        if not os.path.exists(memory_path):
            try:
                print("Exporting memory timeline")
                torch.cuda.memory._dump_snapshot()
            except Exception as e:
                print(f"Exporting memory failed with error: {e}")
            finally:
                torch.cuda.memory._record_memory_history(enabled=False)

        print("Exporting chrome trace")
        prof.export_chrome_trace(os.path.join(profiler_log_dir, f"{prof.step_num}-chrome_trace.json.gz"))

        # print("Exporting tensorboard trace")
        # tb_trace_handler(prof)

        print("Saving trace summary")
        with open(os.path.join(profiler_log_dir, f"{prof.step_num}-trace_summary.txt"), "wt") as fo:
            fo.write(str(prof.key_averages().table(sort_by="cuda_time_total", row_limit=1000)))

    wait, warmup, active, repeat = (int(i) for i in args.profiler_schedule.split(","))
    print(
        f"INFO: PyTorch Profiler enabled. Waiting {wait}, warming up {warmup}, recording {active} epochs and repeat {repeat}."
    )

    prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
        record_shapes=True,
        profile_memory=False,
        with_stack=True,
        on_trace_ready=on_trace_ready,
    )
    prof.start()

    torch.cuda.memory._record_memory_history(max_entries=100000)

    return prof


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_label=None,
    post_pred=None,
):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0

    prof = None
    if args.rank == 0 and args.profile:
        prof = start_profiler(args)

    if args.rank == 0:
        torch.cuda.reset_peak_memory_stats(device=args.gpu)

    start_time_global = time.time()

    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
        )
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
            print("Time since start: {:.2f}s".format(time.time() - start_time_global))

        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_avg_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
                writer=writer,
            )

            val_avg_acc = np.mean(val_avg_acc)

            if args.rank == 0:
                print(
                    "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                    "acc",
                    val_avg_acc,
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                if writer is not None:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)
                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                        )
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))

        if scheduler is not None:
            scheduler.step()

        if args.rank == 0:
            current_device = args.gpu
            max_mem_allocated_bytes = torch.cuda.max_memory_allocated(device=current_device)
            max_mem_allocated_gb = max_mem_allocated_bytes / (1024 * 1024 * 1024)
            max_mem_reserved_bytes = torch.cuda.max_memory_reserved(device=current_device)
            max_mem_reserved_gb = max_mem_reserved_bytes / (1024 * 1024 * 1024)

            print(
                f"Epoch {epoch} GPU Peak Memory - "
                f"Allocated: {max_mem_allocated_gb:.2f} GB, "
                f"Reserved: {max_mem_reserved_gb:.2f} GB"
            )
            torch.cuda.reset_peak_memory_stats(device=current_device)

        if prof:
            prof.step()

    if prof:
        # NOTE: torch.profiler.profile is meant to be used as a context manager.
        #       Calling `.__exit__` here instead of just `.stop` to perform some additional
        #       cleanup needed, but should consider refactoring to avoid unexpected behaviour.
        prof.__exit__(None, None, None)

    print("Training Finished !, Best Accuracy: ", val_acc_max)
    return val_acc_max
