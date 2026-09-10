import argparse
import itertools
import logging
import os
import time
from typing import Callable

import numpy as np
import open_clip
import torch
from accelerate import Accelerator
from clipora.config import TrainConfig, parse_yaml_to_config, save_config_to_yaml
from clipora.data import get_dataloader
from clipora.lora.inject import inject_linear_attention
from clipora.scheduler.cosine import cosine_lr
from peft import LoraConfig, PeftModel, get_peft_model
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def compute_clip_loss(model, X, Y):
    loss = open_clip.ClipLoss()
    image_features, text_features, logit_scale = model(X, Y)
    total_loss = loss(image_features, text_features, logit_scale)
    return total_loss


@torch.no_grad()
def evaluate(model, dataloader, config):
    out = {}
    model.eval()
    losses = torch.zeros(config.eval_steps)
    for k in range(config.eval_steps):
        X, Y = next(iter(dataloader))
        loss = compute_clip_loss(model, X, Y)
        losses[k] = loss.item()
    out["eval_loss"] = losses.mean()
    model.train()
    return out


def init_model(config: TrainConfig, lora_adapter_path=None, full_weights_path=None):
    # lora_adapter_path = checkpoint path.
    # full_weights_path = if given, dont download pretrained weights, instead load weights from this
    pretrained = None if full_weights_path is not None else config.pretrained
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name=config.model_name,
        pretrained=pretrained,
    )
    model_config = open_clip.get_model_config(config.model_name)
    if config.lora_text:
        model = inject_linear_attention(
            model=model,
            encoders={"transformer"},
            embed_dim=model_config["embed_dim"],
            num_heads=model_config["text_cfg"]["heads"],
        )
    if config.lora_vision:
        model = inject_linear_attention(
            model=model,
            encoders={"visual.transformer"},
            embed_dim=model_config["vision_cfg"]["width"],
            num_heads=config.vision_heads,
        )

    if full_weights_path:
        model.load_state_dict(torch.load(full_weights_path))

    # If not None, load existing loras from here
    if lora_adapter_path:
        model = PeftModel.from_pretrained(model, lora_adapter_path)

    else:
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["qkv", "proj"],
        )
        model = get_peft_model(model, lora_config)

    if config.compile:
        model.compile()
    return model, preprocess_train


def main(config: TrainConfig, job_callback: Callable | None = None):
    """Main training loop.

    Args:
        config (TrainConfig): clipora training config
        job_callback (Callable | None, optional): A callback function to update job status
        each iteration. Used by API. Defaults to None.
    """
    logging.basicConfig(level=logging.INFO)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="wandb" if config.wandb else None,
    )

    if accelerator.is_main_process:
        accelerator.print()
        if config.output_dir is not None:
            accelerator.print(f"Output directory: {config.output_dir}")
            os.makedirs(config.output_dir, exist_ok=True)

        if config.wandb:
            accelerator.init_trackers(
                project_name=config.wandb_project if config.wandb_project else None,
            )

    if config.seed is not None:
        seed = config.seed
        accelerator.print(f"Using seed {seed}")
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    model, preprocess_train = init_model(config)

    train_dataloader = get_dataloader(config, preprocess_train, "train")
    eval_dataloader = get_dataloader(config, preprocess_train, "val")
    assert len(train_dataloader), "No data found, please check your data location."

    if config.gradient_checkpointing:
        model.set_grad_checkpointing(True)

    if config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    if isinstance(config.learning_rate, str):
        config.learning_rate = float(config.learning_rate)

    params_to_optimize = [
        {
            "params": itertools.chain(model.parameters()),
            "lr": config.learning_rate,
        },
    ]

    optimizer = optimizer_class(
        params_to_optimize,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
    )

    # create scheduler if train
    total_steps = train_dataloader.num_batches * config.epochs
    # if args.warmup is float, it is a percentage of total_steps
    if isinstance(config.warmup, float):
        assert 0 <= config.warmup <= 1, "Warmup must be between 0 and 1 if not a fixed number of steps."
        config.warmup = int(config.warmup * total_steps)

    scheduler = cosine_lr(optimizer, config.learning_rate, config.warmup, total_steps)

    model, optimizer, scheduler, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader, eval_dataloader
    )

    print("***** Running training *****")
    print(f"  Using device: {accelerator.device}")
    print(f"  Num Iters = {len(train_dataloader)}")
    print(f"  Num Epochs = {config.epochs}")
    print(f"  Instantaneous batch size per device = {config.batch_size}")
    print(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(config.epochs * len(train_dataloader)),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(config.epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            if accelerator.is_local_main_process:
                if global_step % config.eval_interval == 0:
                    if accelerator.is_local_main_process:
                        eval_loss = evaluate(model, eval_dataloader, config)
                        accelerator.log(eval_loss, step=global_step)
                        progress_bar.write(f"Step: {global_step}, Eval loss: {eval_loss['eval_loss']}")
                        if eval_loss["eval_loss"] < best_val_loss:
                            best_val_loss = eval_loss["eval_loss"]
                            checkpoint_name = f"checkpoint_{global_step}"
                            save_path = os.path.join(config.output_dir, checkpoint_name)
                            model.save_pretrained(save_path)
                            if job_callback:
                                job_callback(best_finetuned_model_path=checkpoint_name)
                            # save the clipora config we used for training for later use and bookkeeping
                            save_config_to_yaml(config, os.path.join(save_path, "train_config.yaml"))

            X, Y = batch
            loss = compute_clip_loss(model, X, Y)
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                params_to_clip = model.parameters()
                accelerator.clip_grad_norm_(params_to_clip, 1.0)  # args.max_grad_norm)
            optimizer.step()
            scheduler(global_step)
            progress_bar.update(1)
            if job_callback:
                percent = int((epoch * len(train_dataloader) + step) / (config.epochs * len(train_dataloader)) * 100)
                job_callback(status="training", detail=f"Training at {percent}%")
            global_step += 1

            logs = {
                "loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "step": global_step,
                "epoch": epoch,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

    accelerator.wait_for_everyone()

    if accelerator.is_local_main_process:
        save_path = os.path.join(config.output_dir, "final")
        model.save_pretrained(save_path)
        save_config_to_yaml(config, os.path.join(save_path, "train_config.yaml"))

    accelerator.print("\n\nTraining completed.\n\n")
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="The path to the yaml file containing the training configuration.",
    )
    print(f"Starting clipora training with config: {parser.parse_args().config}")
    config = parse_yaml_to_config(parser.parse_args().config)
    main(config)
