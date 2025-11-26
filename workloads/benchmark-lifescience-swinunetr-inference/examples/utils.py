import logging
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import requests
from monai.transforms import AsDiscreted, Compose, EnsureChannelFirstd, LoadImaged, Spacingd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)


def send_prediction_request(
    image_path: str, output_path: str = "prediction_output.nii.gz", server_url: str = "http://localhost:8000/predict/"
):
    """
    Sends an image to the prediction server and saves the result.

    Args:
        image_path (str): Path to the input image file (.nii, .nii.gz, or .npy).
        output_path (str): Path to save the received prediction file.
        server_url (str): URL of the inference service.
    """
    if not os.path.exists(image_path):
        LOGGER.error(f"Error: Input image file not found at {image_path}")
        return

    LOGGER.info(f"Attempting to send {image_path} to {server_url}")
    try:
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f)}
            response = requests.post(server_url, files=files, timeout=120)  # 120-second timeout

        LOGGER.info(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            with open(output_path, "wb") as out_f:
                out_f.write(response.content)
            LOGGER.info(f"Success! Prediction saved to {output_path}")
        else:
            LOGGER.error("Error: Request failed.")
            try:
                error_detail = response.json()
                LOGGER.error(f"Server Error Detail: {error_detail}")
            except requests.exceptions.JSONDecodeError:
                LOGGER.error(f"Server Error (non-JSON): {response.text}")

    except requests.exceptions.ConnectionError:
        LOGGER.error(f"Error: Could not connect to the server at {server_url}. Is it running?")
    except requests.exceptions.Timeout:
        LOGGER.error("Error: The request timed out.")
    except Exception as e:
        LOGGER.error(f"An unexpected error occurred: {e}")


def load_and_transform_input(keys: List[str], data_dict: dict):
    image_transforms = Compose(
        [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys="input", channel_dim="no_channel"),
            Spacingd(keys=keys, pixdim=(1.5, 1.5, 2), mode=("bilinear",)),
        ]
    )
    return image_transforms(data_dict)


def load_and_transform(keys: List[str], data_dict: dict):
    """
    Loads and transforms data, converting the label to one-hot format.
    """
    num_classes = 14

    image_transforms = Compose(
        [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=["input", "label"], channel_dim="no_channel"),
            AsDiscreted(keys="label", to_onehot=num_classes),
            Spacingd(keys=keys, pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest", "nearest")),
        ]
    )
    return image_transforms(data_dict)


def plot_results(
    input_path_str: str,
    label_path_str: str,
    pred_path_str: str,
    channel_idx: int,
    num_slices_to_plot: int = 4,
):
    """
    Plots slices of the input, prediction, and optionally label NIfTI images.

    Args:
        input_path_str (str): Path to the original input NIfTI file.
        label_path_str (str): Path to the label NIfTI file.
        pred_path_str (str): Path to the prediction NIfTI file.
        channel_idx (int): Channel/Label to extract from the multichannel label file.
        num_slices_to_plot (int): Number of slices to display.
    """
    try:
        data_dict = {"input": input_path_str, "pred": pred_path_str, "label": label_path_str}
        keys = ["input", "label", "pred"]

        processed_dict = load_and_transform(keys=keys, data_dict=data_dict)

        input_data = processed_dict["input"].squeeze()
        pred_data = processed_dict["pred"][channel_idx, ...].squeeze()
        label_data = processed_dict["label"][channel_idx, ...].squeeze()

        num_cols = 3
        depth = input_data.shape[2]

        slice_indices = np.linspace(depth // 4, 3 * depth // 4, num_slices_to_plot, dtype=int)
        slice_indices = np.clip(np.unique(slice_indices), 0, depth - 1)

        fig, axes = plt.subplots(len(slice_indices), num_cols, figsize=(num_cols * 3, len(slice_indices) * 3))
        if len(slice_indices) == 1:
            axes = np.array([axes]).reshape(1, -1)

        for i, slice_idx in enumerate(slice_indices):
            axes[i, 0].imshow(np.rot90(input_data[:, :, slice_idx]), cmap="gray")
            axes[i, 0].set_title(f"Input - Slice {slice_idx}")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(np.rot90(label_data[:, :, slice_idx]), cmap="viridis")
            axes[i, 1].set_title(f"Label - Slice {slice_idx}")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(np.rot90(pred_data[:, :, slice_idx]), cmap="viridis")
            axes[i, 2].set_title(f"Prediction - Slice {slice_idx}")
            axes[i, 2].axis("off")

        fig.suptitle(f"Input vs. Prediction (vs. Label) - {os.path.basename(input_path_str)}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        plt.close(fig)
    except Exception as e:
        LOGGER.error(f"An error occurred during plotting: {e}", exc_info=True)


def plot_results_overlap(
    input_path_str: str,
    label_path_str: str,
    pred_path_str: str,
    channel_idx: int,
    slice_to_plot: int,
):
    """
    Plots a specific slice showing overlaps of input, prediction, and target label.

    This function generates a 1x3 plot for a single specified slice:
    1. Input slice with the target mask superimposed.
    2. Input slice with the prediction mask superimposed.
    3. Input slice with both target and prediction masks superimposed.

    Args:
        input_path_str (str): Path to the original input NIfTI file.
        label_path_str (str): Path to the label NIfTI file.
        pred_path_str (str): Path to the prediction NIfTI file.
        channel_idx (int): Labels to extract from the multichannel label file.
        slice_to_plot (int): The specific slice index to visualize.
    """
    try:
        data_dict = {"input": input_path_str, "pred": pred_path_str, "label": label_path_str}
        keys = ["input", "label", "pred"]

        processed_dict = load_and_transform(keys=keys, data_dict=data_dict)

        input_data = processed_dict["input"].squeeze()
        pred_data = processed_dict["pred"][channel_idx, ...].squeeze()
        label_data = processed_dict["label"][channel_idx, ...].squeeze()

        # Validate slice index
        if not (0 <= slice_to_plot < input_data.shape[2]):
            raise ValueError(f"Slice index {slice_to_plot} is out of bounds for depth {input_data.shape[2]}.")

        input_slice = np.rot90(input_data[:, :, slice_to_plot])
        pred_slice = np.rot90(pred_data[:, :, slice_to_plot])
        label_slice = np.rot90(label_data[:, :, slice_to_plot])

        # Use a masked array to only show the "on" pixels of the masks
        pred_mask = np.ma.masked_where(pred_slice == 0, pred_slice)
        label_mask = np.ma.masked_where(label_slice == 0, label_slice)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # Plot 1: Input with Target
        axes[0].imshow(input_slice, cmap="gray")
        axes[0].imshow(label_mask, cmap="autumn", alpha=0.5)  # autumn is yellow-red
        axes[0].set_title(f"Input + Target (Slice {slice_to_plot})")
        axes[0].axis("off")

        # Plot 2: Input with Prediction
        axes[1].imshow(input_slice, cmap="gray")
        axes[1].imshow(pred_mask, cmap="cool", alpha=0.5)  # cool is cyan-magenta
        axes[1].set_title(f"Input + Prediction (Slice {slice_to_plot})")
        axes[1].axis("off")

        # Plot 3: Input with Target and Prediction
        axes[2].imshow(input_slice, cmap="gray")
        axes[2].imshow(label_mask, cmap="autumn", alpha=0.7)
        axes[2].imshow(pred_mask, cmap="cool", alpha=0.4)
        axes[2].set_title("Input + Target (Red) + Pred (Blue)")
        axes[2].axis("off")

        # --- 4. Finalize and Show ---
        fig.suptitle(f"Overlap Visualization - {os.path.basename(input_path_str)}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        plt.close(fig)
    except Exception as e:
        LOGGER.error(f"An error occurred during overlap plotting: {e}", exc_info=True)


def plot_input_scan(input_path: str, num_slices_to_plot: int = 4):
    """
    Plots several axial slices of a single NIfTI input scan.

    Args:
        input_path (str): Path to the original input NIfTI file.
        num_slices_to_plot (int): Number of slices to display.
    """
    try:
        # 1. Load only the input data
        data_dict = {"input": input_path}
        processed_dict = load_and_transform_input(keys=["input"], data_dict=data_dict)
        input_data = processed_dict["input"].squeeze()
        LOGGER.info(f"Input data shape: {input_data.shape}")
        depth = input_data.shape[2]

        # 2. Select representative slices to plot
        slice_indices = np.linspace(depth // 4, 3 * depth // 4, num_slices_to_plot, dtype=int)
        slice_indices = np.clip(np.unique(slice_indices), 0, depth - 1)

        if len(slice_indices) == 0:
            LOGGER.error("Could not determine valid slice indices to plot.")
            return

        # 3. Create the plot grid
        num_cols = int(np.ceil(np.sqrt(len(slice_indices))))
        num_rows = int(np.ceil(len(slice_indices) / num_cols))

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
        # Flatten the axes array to make it easy to iterate over
        axes = axes.flatten()
        for i, slice_idx in enumerate(slice_indices):
            ax = axes[i]
            ax.imshow(np.rot90(input_data[:, :, slice_idx]), cmap="gray")
            ax.set_title(f"Slice {slice_idx}")
            ax.axis("off")

        # Turn off any unused subplots in the grid
        for i in range(len(slice_indices), len(axes)):
            axes[i].axis("off")

        fig.suptitle(f"Input Scan: {os.path.basename(input_path)}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
    except Exception as e:
        LOGGER.error(f"An error occurred during plotting: {e}", exc_info=True)
