# Script should: merge models? do inference?

# Will get the base model name from train config and merge it with trained
# lora adapter weights.

import argparse
import os
import re
import shutil

import open_clip
import torch
import visualize_results
from clipora.config import TrainConfig, parse_yaml_to_config, save_config_to_yaml
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from safetensors.torch import load_file
from train import get_dataloader, init_model
from train import main as train_main

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        X, Y = X.to(device), Y.to(device)
        loss = compute_clip_loss(model, X, Y)
        losses[k] = loss.item()
    out["eval_loss"] = losses.mean()
    model.train()
    return out


def get_newest_checkpoint(output_dir):
    # Get the checkpoint path with the highest iteration
    if not os.path.exists(output_dir):
        return None

    files = os.listdir(output_dir)

    # Filter for checkpoint files and extract their numbers
    checkpoint_pattern = re.compile(r"^checkpoint_(\d+)$")
    checkpoints = []

    for file in files:
        match = checkpoint_pattern.match(file)
        if match:
            checkpoint_num = int(match.group(1))
            checkpoints.append((checkpoint_num, file))

    if not checkpoints:
        return None

    latest_checkpoint = max(checkpoints, key=lambda x: x[0])

    return os.path.join(output_dir, latest_checkpoint[1])


def save_full_model_weights(model, lora_adapter_path, config, output_dir):
    # It's not really needed to save the full weights as LORAs purpose is to be able to
    # share them more easily?
    output_path = os.path.join(lora_adapter_path, "merged_model_weights.pt")
    print(f"Saving merged model weights to {output_path}")
    merged_model = model.merge_and_unload()
    torch.save(merged_model.state_dict(), output_path)


def run_inference_comparison(lora_model, preprocess, config):
    print("Running inference comparison...")

    # Load the original CLIP model (no LORA)
    original_model, _, _ = open_clip.create_model_and_transforms(
        model_name=config.model_name,
        pretrained=config.pretrained,
    )

    original_model = original_model.to(device)
    lora_model = lora_model.to(device)
    eval_dataloader = get_dataloader(config, preprocess, "val")
    original_eval_loss = evaluate(original_model, eval_dataloader, config)
    print("Original eval loss:")
    print(original_eval_loss)
    lora_eval_loss = evaluate(lora_model, eval_dataloader, config)
    print("Lora eval loss:")
    print(lora_eval_loss)
    print("Visualizing results...")
    visualize_results.main(original_model, lora_model, preprocess, config)


def run_single_inference(job_dict, image_path, classes: list[str]):
    # run inference on a single image using the LORA model
    # gets the finetuned model that was saved in the training job job_id
    if not job_dict:
        print("No job info provided")
        return None, None

    TRAIN_JOB_OUTPUT_DIR = os.getenv("TRAIN_JOB_OUTPUT_DIR", "/tmp/trained_models/")
    lora_adapter_path = job_dict["best_finetuned_model_path"]
    # If relative path, assume it's relative to TRAIN_JOB_OUTPUT_DIR
    if not os.path.isabs(lora_adapter_path):
        lora_adapter_path = os.path.join(TRAIN_JOB_OUTPUT_DIR, job_dict["id"], lora_adapter_path)
    config_path = os.path.join(lora_adapter_path, "train_config.yaml")
    config = parse_yaml_to_config(config_path)
    lora_model, preprocess = init_model(config, lora_adapter_path=lora_adapter_path)
    lora_model.to(device)
    processed_image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    text_tokens = open_clip.tokenize(classes).to(device)

    with torch.no_grad(), torch.autocast("cuda"):
        image_features = lora_model.encode_image(processed_image)
        text_features = lora_model.encode_text(text_tokens)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()

    print(text_probs)
    return text_probs.tolist(), classes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference comparison between original CLIP model and LORA fine-tuned model.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default=None,
        help="The path to the yaml file containing the training configuration. "
        "If not provided, will try to load train_config.yaml from lora_adapter_path",
    )

    parser.add_argument(
        "--save_full_model_weights",
        action="store_true",
        help="Save pytorch merged model weights. Often not needed if you use pretrained CLIP and trained LORAs",
    )

    parser.add_argument(
        "--lora_adapter_path",
        type=str,
        required=False,
        help="The path to the LoRA adapter weights, e.g. checkpoint. If not provided, will use newest checkpoint",
        default=None,
    )

    args = parser.parse_args()
    # TODO: exception handling
    if args.lora_adapter_path is None and args.config is None:
        raise Exception("Both lora_adapter_path and config cannot be None")

    # If None, assume the train config that was used to train the model was saved in the checkpoint folder
    lora_adapter_path = args.lora_adapter_path
    if lora_adapter_path is None:
        print(f"Loading config from: {args.config}")
        config = parse_yaml_to_config(args.config)
        # By default, load latest checkpoint
        lora_adapter_path = (
            args.lora_adapter_path if args.lora_adapter_path else get_newest_checkpoint(config.output_dir)
        )

    if args.config is None:
        print("Trying to load train config from lora adapter/checkpoint path")
        config_path = os.path.join(lora_adapter_path, "train_config.yaml")
        config = parse_yaml_to_config(config_path)

    print(f"Config output dir: {config.output_dir}, lora adapter path: {lora_adapter_path}")
    lora_model, preprocess = init_model(config, lora_adapter_path=lora_adapter_path)
    if args.save_full_model_weights:
        save_full_model_weights(lora_model, lora_adapter_path, config, config.output_dir)
    run_inference_comparison(lora_model, preprocess, config)
