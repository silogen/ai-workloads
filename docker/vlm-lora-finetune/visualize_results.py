import argparse
import os
import random

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import open_clip  # or open_clip
import pandas as pd
import torch
from clipora.config import parse_yaml_to_config
from matplotlib.colors import Normalize
from PIL import Image
from train import init_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualize_results(image_path, texts, before, after, output_dir):
    """Saves an image with a table showing text probabilities before and after LORA fine-tuning.
    texts, before and after have to be the same length
    Args:
        image_path (str): The path to the input image.
        texts (list): The list of text prompts.
        before (list): The probabilities before fine-tuning.
        after (list): The probabilities after fine-tuning.
        output_dir (str): The directory where the output image will be saved.
    """
    img = Image.open(image_path)
    norm = Normalize(vmin=0, vmax=1)

    # Find max indices
    max_before_idx = np.argmax(before)
    max_after_idx = np.argmax(after)

    # Calculate dynamic column widths based on text length
    max_text_length = max(len(text) for text in texts)

    # Adjust column widths dynamically
    if max_text_length > 50:  # Long text
        col_widths = [0.6, 0.2, 0.2]  # Give more space to text
    elif max_text_length > 30:  # Medium text
        col_widths = [0.5, 0.25, 0.25]
    else:  # Short text
        col_widths = [0.4, 0.3, 0.3]  # Original proportions

    # Create figure with dynamic width
    fig_width = max(12, max_text_length * 0.15)  # Scale figure width with text length
    fig, ax = plt.subplots(figsize=(fig_width, max(6, len(texts) * 0.5 + 2)))
    ax.axis("off")

    # Image panel - adjust based on figure width
    img_width = min(0.25, 3.0 / fig_width)  # Cap image width but scale with figure
    img_ax = fig.add_axes([0.05, 0.1, img_width, 0.8])
    img_ax.imshow(img)
    img_ax.axis("off")

    # Table panel - use remaining space
    table_start = 0.05 + img_width + 0.05
    table_width = 0.9 - table_start
    table_ax = fig.add_axes([table_start, 0.1, table_width, 0.8])
    table_ax.axis("off")

    # Calculate positions
    x_positions = np.cumsum([0] + col_widths[:-1])

    # Table headers
    headers = ["Text", "Before", "After"]
    for i, header in enumerate(headers):
        table_ax.text(
            x_positions[i], 1, header, ha="left", va="bottom", fontsize=12, weight="bold", transform=table_ax.transAxes
        )

    # Calculate row height based on number of items
    row_height = min(0.08, 0.7 / len(texts))  # Dynamic row height

    # Draw cells
    for i, (text, b, a) in enumerate(zip(texts, before, after)):
        y = 0.9 - i * row_height

        # Handle long text with wrapping
        if len(text) > 40:
            # Split long text into multiple lines
            words = text.split()
            lines = []
            current_line = ""
            max_chars_per_line = int(40 * col_widths[0] / 0.4)  # Scale with column width

            for word in words:
                if len(current_line + " " + word) <= max_chars_per_line:
                    current_line += " " + word if current_line else word
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)

            # Display multiline text
            for j, line in enumerate(lines):
                table_ax.text(
                    x_positions[0], y - j * 0.02, line, ha="left", va="center", fontsize=9, transform=table_ax.transAxes
                )
        else:
            # Display single line text
            table_ax.text(x_positions[0], y, text, ha="left", va="center", fontsize=10, transform=table_ax.transAxes)

        # Draw probability boxes
        for j, val in enumerate([b, a], start=1):
            color = plt.cm.Blues(norm(val)) if j == 1 else plt.cm.Reds(norm(val))
            x = x_positions[j]
            width = col_widths[j]

            # Highlight max values with red border
            edge_color = "red" if (j == 1 and i == max_before_idx) or (j == 2 and i == max_after_idx) else "black"

            rect = patches.Rectangle(
                (x, y - row_height / 2 + 0.01),
                width,
                row_height - 0.02,
                transform=table_ax.transAxes,
                color=color,
                ec=edge_color,
                lw=1.5,
            )
            table_ax.add_patch(rect)

            table_ax.text(
                x + width / 2,
                y,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if val > 0.5 else "black",  # Better contrast
                weight="bold",
                transform=table_ax.transAxes,
            )

    plt.savefig(os.path.join(output_dir, "output_image.png"), bbox_inches="tight", dpi=150)
    plt.close()


def main(original_model, lora_model, preprocess, config, csv_path=None):
    """Display probabilities before and after LORA fine-tuning for 10 random texts
    from the evaluation dataset.
    """
    # === Load CSV and Select Data ===
    csv_path = csv_path or config.eval_dataset
    df = pd.read_csv(csv_path)
    # Get 10 random texts from csv as classes
    classes = df[config.text_col].drop_duplicates().sample(10).tolist()
    # Get an image which has the first text as the correct one
    correct_text = classes[0]
    img_path = df[df[config.text_col] == correct_text][config.image_col].iloc[0]
    # === Preprocess Inputs ===
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    text_tokens = open_clip.tokenize(classes).to(device)

    with torch.no_grad():
        img_feat_before = original_model.encode_image(image)
        txt_feat_before = original_model.encode_text(text_tokens)
        probs_before = (img_feat_before @ txt_feat_before.T).softmax(dim=-1).squeeze().cpu().numpy()

        img_feat_after = lora_model.encode_image(image)
        txt_feat_after = lora_model.encode_text(text_tokens)
        probs_after = (img_feat_after @ txt_feat_after.T).softmax(dim=-1).squeeze().cpu().numpy()

    print("probs before:")
    print(probs_before)
    print("probs after:")
    print(probs_after)

    visualize_results(img_path, classes, probs_before, probs_after, config.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="The path to the yaml file containing the training configuration.",
    )

    parser.add_argument(
        "--lora_adapter_path",
        type=str,
        help="The path to the LoRA adapter weights, e.g. checkpoint folder.",
    )

    args = parser.parse_args()
    lora_adapter_path = args.lora_adapter_path
    config = parse_yaml_to_config(args.config)
    lora_model, preprocess = init_model(config, lora_adapter_path=lora_adapter_path)
    # Load the original CLIP model (no LORA)
    original_model, _, _ = open_clip.create_model_and_transforms(
        model_name=config.model_name,
        pretrained=config.pretrained,
    )

    original_model = original_model.to(device)
    lora_model = lora_model.to(device)
    main(original_model, lora_model, preprocess, config)
