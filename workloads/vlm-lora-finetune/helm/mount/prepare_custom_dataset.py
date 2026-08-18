"""
Dataset preparation script for Bridge robotics dataset. NOTE: this script was created
as a one-off script for parsing part of bridge dataset (TFRecord) to openCLIP format
(a CSV with image paths and associated text descriptions and images). See helm README.md for more details.

This script downloads and processes the Bridge robotics dataset from Hugging Face,
converting TFRecord format data into a custom format suitable for vision-language
model fine-tuning. The script:

1. Downloads TFRecord files and metadata from a Hugging Face repository
2. Parses episode data including images, language instructions, actions, and states
3. Extracts and saves images as PNG files organized by episodes
4. Creates CSV files mapping image paths to language instructions
5. Splits the data into training and evaluation sets. These are saved in the folder of
'train_dataset' variable in the config yaml file.

The resulting dataset format is compatible with openCLIP and similar vision-language
model training pipelines.

Example usage:
    python prepare_custom_dataset.py --repo_id dusty-nv/bridge_orig_ep100 --config /path/to/clipora_config.yml
"""

import argparse
import csv
import json
import os
import random

import numpy as np
import tensorflow as tf
import yaml
from huggingface_hub import hf_hub_download
from PIL import Image


def get_tfrecord_dataset(repo_id="dusty-nv/bridge_orig_ep100"):
    """
    Download TFRecord dataset files from a Hugging Face repository.

    Downloads the Bridge dataset TFRecord files and associated metadata files
    from the specified Hugging Face repository.

    Args:
        repo_id (str): The Hugging Face repository ID (default: "dusty-nv/bridge_orig_ep100").

    Returns:
        tf.data.TFRecordDataset: The loaded TFRecord dataset.
        dict: Features metadata loaded from features.json.
    """
    # TODO: how to make it so we don't need to hardcode filenames here
    files = [
        "1.0.0/bridge_orig_ep100-train.tfrecord-00000-of-00002",
        "1.0.0/bridge_orig_ep100-train.tfrecord-00001-of-00002",
        "1.0.0/dataset_info.json",
        "1.0.0/features.json",
    ]

    tfrecord_files = []

    features_json_path = None
    for fname in files:
        path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=fname)
        # collect tfrecords for later use
        if "tfrecord" in fname:
            tfrecord_files.append(path)
        elif "features.json" in fname:
            features_json_path = path

        print(path)

    raw_dataset = tf.data.TFRecordDataset(tfrecord_files)
    # features.json contains tfrecord schema
    with open(features_json_path, "r") as f:
        features_data = json.load(f)

    return raw_dataset, features_data


def build_feature_description_from_json(features_data):
    """
    Build TensorFlow feature description from features.json schema.

    Parses the features.json schema to create TensorFlow feature descriptions
    that can be used with tf.io.parse_single_example. Handles nested features,
    variable-length sequences, and different data types.

    Args:
        features_data (dict): Features schema loaded from features.json.

    Returns:
        dict: Feature description mapping for tf.io.parse_single_example.
        dict: Reshape information for tensors that need reshaping.
        set: Fields that should be converted from int64 to bool.
    """
    # Map dtype strings to TensorFlow dtypes
    # Note: VarLenFeature doesn't support bool, so we use int64 and convert later
    dtype_mapping = {
        "float32": tf.float32,
        "int64": tf.int64,
        "bool": tf.int64,  # Parse as int64, convert to bool later
        "string": tf.string,
        "uint8": tf.uint8,
    }

    # Keep track of fields that should be converted to bool after parsing
    bool_fields = set()

    feature_description = {}
    reshape_info = {}

    def process_features(features_dict, prefix=""):
        """
        Recursively process features dictionary from features.json schema.

        Traverses the nested feature structure and creates appropriate TensorFlow
        feature descriptions for parsing TFRecord data.

        Args:
            features_dict (dict): Features dictionary from schema.
            prefix (str): Current key prefix for nested features.
        """
        for key, feature_spec in features_dict["features"].items():
            full_key = f"{prefix}/{key}" if prefix else key

            if feature_spec["pythonClassName"] == "tensorflow_datasets.core.features.text_feature.Text":
                if prefix and "steps" in prefix:  # Variable length sequence
                    feature_description[full_key] = tf.io.VarLenFeature(tf.string)
                else:  # Fixed length
                    feature_description[full_key] = tf.io.FixedLenFeature([], tf.string)

            elif feature_spec["pythonClassName"] == "tensorflow_datasets.core.features.scalar.Scalar":
                dtype_str = feature_spec["tensor"]["dtype"]
                tf_dtype = dtype_mapping.get(dtype_str, tf.string)

                # Track boolean fields for later conversion
                if dtype_str == "bool":
                    bool_fields.add(full_key)

                if prefix and "steps" in prefix:  # Variable length sequence
                    feature_description[full_key] = tf.io.VarLenFeature(tf_dtype)
                else:  # Fixed length
                    feature_description[full_key] = tf.io.FixedLenFeature([], tf_dtype)

            elif feature_spec["pythonClassName"] == "tensorflow_datasets.core.features.tensor_feature.Tensor":
                tensor_info = feature_spec["tensor"]
                dtype_str = tensor_info["dtype"]
                tf_dtype = dtype_mapping.get(dtype_str, tf.string)

                # Extract shape information
                if "shape" in tensor_info and "dimensions" in tensor_info["shape"]:
                    shape = [int(dim) for dim in tensor_info["shape"]["dimensions"]]
                    if prefix and "steps" in prefix:  # Variable length sequence of tensors
                        feature_description[full_key] = tf.io.VarLenFeature(tf_dtype)
                        reshape_info[full_key] = shape  # Store original shape for reshaping
                    else:  # Fixed length tensor
                        feature_description[full_key] = tf.io.FixedLenFeature(shape, tf_dtype)
                else:
                    # Scalar tensor
                    if prefix and "steps" in prefix:
                        feature_description[full_key] = tf.io.VarLenFeature(tf_dtype)
                    else:
                        feature_description[full_key] = tf.io.FixedLenFeature([], tf_dtype)

            elif feature_spec["pythonClassName"] == "tensorflow_datasets.core.features.image_feature.Image":
                # Images are stored as encoded strings (PNG/JPEG)
                if prefix and "steps" in prefix:
                    feature_description[full_key] = tf.io.VarLenFeature(tf.string)
                else:
                    feature_description[full_key] = tf.io.FixedLenFeature([], tf.string)

            elif feature_spec["pythonClassName"] == "tensorflow_datasets.core.features.features_dict.FeaturesDict":
                # Recursively process nested features
                process_features(feature_spec["featuresDict"], full_key)

            elif feature_spec["pythonClassName"] == "tensorflow_datasets.core.features.dataset_feature.Dataset":
                # Process sequence features with "steps" prefix to indicate variable length
                sequence_feature = feature_spec["sequence"]["feature"]
                if "featuresDict" in sequence_feature:
                    process_features(sequence_feature["featuresDict"], full_key)

    # Process the root features
    process_features(features_data["featuresDict"])

    return feature_description, reshape_info, bool_fields


def parse_bridge_episode(example_proto, features_data):
    """
    Parse a Bridge dataset episode from TFRecord format.

    Parses a single episode from the Bridge dataset using the features schema,
    converts sparse tensors to dense, handles data type conversions, and
    performs necessary reshaping for sequence data.

    Args:
        example_proto: Raw TFRecord example protocol buffer.
        features_data (dict): Features schema from features.json.

    Returns:
        dict: Parsed episode data with properly shaped tensors.
    """

    feature_description, reshape_info, bool_fields = build_feature_description_from_json(features_data)

    parsed = tf.io.parse_single_example(example_proto, feature_description)

    # Convert sparse tensors to dense
    for key in parsed:
        if isinstance(parsed[key], tf.SparseTensor):
            parsed[key] = tf.sparse.to_dense(parsed[key])

    # Convert int64 fields back to bool where appropriate
    for key in bool_fields:
        if key in parsed:
            parsed[key] = tf.cast(parsed[key], tf.bool)

    # For debugging: let's check if the data is already properly shaped
    # and avoid reshaping if it's not needed
    if reshape_info:
        step_keys = [k for k in parsed.keys() if k.startswith("steps/")]
        if step_keys:
            reference_key = step_keys[0]
            num_steps = tf.shape(parsed[reference_key])[0]

            for key, original_shape in reshape_info.items():
                if key in parsed:
                    current_tensor = parsed[key]
                    current_shape = tf.shape(current_tensor)

                    # Check if already has the right shape (might be [num_steps, feature_size] already)
                    if len(original_shape) == 1:  # 1D feature like [7]
                        expected_shape = [num_steps, original_shape[0]]

                        # If current shape matches expected, no reshaping needed
                        # If it's flattened [num_steps * feature_size], reshape it
                        if tf.rank(current_tensor) == 1:
                            # It's flattened, try to reshape
                            total_elements = tf.size(current_tensor)
                            feature_size = original_shape[0]

                            # Only reshape if the math works out
                            if tf.math.equal(total_elements % feature_size, 0):
                                inferred_num_steps = total_elements // feature_size
                                parsed[key] = tf.reshape(current_tensor, [inferred_num_steps, feature_size])

    return parsed


def save_episode_images(parsed_example, episode_num=1, base_output_dir="episodes"):
    """
    Save episode images to disk and prepare CSV data.

    Extracts images and language instructions from a parsed episode,
    saves images as PNG files in an episode-specific directory,
    and returns CSV data mapping image paths to instructions.

    Args:
        parsed_example (dict): Parsed episode data from parse_bridge_episode.
        episode_num (int): Episode number for directory naming (default: 1).
        base_output_dir (str): Base directory for saving episodes (default: "episodes").

    Returns:
        dict: Episode metadata including:
            - instructions: List of language instructions
            - num_steps: Number of steps in the episode
            - actions: Action data as numpy array
            - states: State data as numpy array
            - episode_dir: Path to episode directory
            - csv_data: List of [image_path, instruction] pairs for CSV
    """

    # Create episode directory
    episode_str = f"episode_{episode_num:04d}"
    episode_dir = os.path.join(base_output_dir, episode_str)
    os.makedirs(episode_dir, exist_ok=True)

    # Get language instruction (should be the same for all steps in episode)
    language_instructions = parsed_example["steps/language_instruction"]
    if len(language_instructions) > 0:
        instructions = [inst.numpy().decode("utf-8") for inst in language_instructions]

    # Get images and already properly shaped action/state tensors
    images = parsed_example["steps/observation/image"]
    actions = parsed_example["steps/action"]
    states = parsed_example["steps/observation/state"]

    num_steps = len(images)

    # Prepare CSV data
    csv_data = []

    # Save each image and record in CSV
    for step_idx in range(num_steps):
        # Decode image
        image = tf.io.decode_png(images[step_idx], channels=3)
        instruction = instructions[step_idx]
        # Save image as PNG
        image_filename = f"step_{step_idx:04d}.png"
        image_path = os.path.abspath(os.path.join(episode_dir, image_filename))

        # Convert to PIL Image and save
        pil_image = Image.fromarray(image.numpy())
        pil_image.save(image_path)

        # Add to CSV data
        csv_data.append([image_path, instruction])

    print(f"Saved {num_steps} images to {episode_dir}")

    return {
        "instructions": instructions,
        "num_steps": num_steps,
        "actions": actions.numpy(),  # Already correctly shaped
        "states": states.numpy(),  # Already correctly shaped
        "episode_dir": episode_dir,
        "csv_data": csv_data,
    }


# Create a simple test to see what we're getting
def debug_first_episode(raw_dataset, features_data):
    """
    Debug function to inspect raw TFRecord data structure.

    Parses and displays information about the first episode in the dataset
    to help understand data shapes and structure before processing.

    Args:
        raw_dataset (tf.data.TFRecordDataset): Raw TFRecord dataset.
        features_data (dict): Features schema from features.json.
    """
    feature_description, reshape_info, bool_fields = build_feature_description_from_json(features_data)

    # Parse one example without reshaping
    for raw_example in raw_dataset.take(1):
        parsed = tf.io.parse_single_example(raw_example, feature_description)

        # Convert sparse tensors to dense
        for key in parsed:
            if isinstance(parsed[key], tf.SparseTensor):
                parsed[key] = tf.sparse.to_dense(parsed[key])

        print("Raw parsed shapes and info:")
        for key, tensor in parsed.items():
            print(f"  {key}: shape={tensor.shape}, size={tf.size(tensor).numpy()}")

        print("\nReshape info from features.json:")
        for key, shape in reshape_info.items():
            print(f"  {key}: expected shape per step = {shape}")

        break


def split_csv_data(input_csv_path, train_csv_path, eval_csv_path, split_percent=80):
    """
    Split a CSV file into training and evaluation sets.

    Reads a CSV file, randomly shuffles the data, and splits it into
    training and evaluation sets based on the specified percentage.

    Args:
        input_csv_path (str): Path to the input CSV file to split.
        train_csv_path (str): Path where the training CSV will be saved.
        eval_csv_path (str): Path where the evaluation CSV will be saved.
        split_percent (int): Percentage of data for training (default: 80).

    Raises:
        ValueError: If split_percent is not between 1 and 99 (exclusive).
    """
    if not (0 < split_percent < 100):
        raise ValueError("split_percent must be between 1 and 99 (exclusive).")

    all_rows = []
    with open(input_csv_path, "r", newline="", encoding="utf-8") as infile:
        reader = csv.reader(infile)
        header = next(reader)  # Read the header row
        for row in reader:
            all_rows.append(row)

    random.shuffle(all_rows)  # Shuffle the rows to ensure random distribution

    num_rows = len(all_rows)
    num_train_rows = int(num_rows * (split_percent / 100))

    train_data = all_rows[:num_train_rows]
    eval_data = all_rows[num_train_rows:]

    # Write to train.csv
    with open(train_csv_path, "w", newline="", encoding="utf-8") as train_file:
        writer = csv.writer(train_file, quoting=csv.QUOTE_ALL)
        writer.writerow(header)  # Write the header
        writer.writerows(train_data)

    # Write to eval.csv
    with open(eval_csv_path, "w", newline="", encoding="utf-8") as eval_file:
        writer = csv.writer(eval_file, quoting=csv.QUOTE_ALL)
        writer.writerow(header)  # Write the header
        writer.writerows(eval_data)

    print("Data split complete:")
    print(f"  Total rows: {num_rows}")
    print(f"  Training rows ({split_percent}%): {len(train_data)}")
    print(f"  Evaluation rows ({100 - split_percent}%): {len(eval_data)}")
    print(f"  Train CSV saved to: {os.path.abspath(train_csv_path)}")
    print(f"  Eval CSV saved to: {os.path.abspath(eval_csv_path)}")


def main(repo_id, train_csv_path, eval_csv_path, split_percent=80):
    """
    Main function to process the dataset from a Hugging Face repository.

    Downloads and processes a Bridge dataset, extracts images and instructions,
    saves them to a directory structure, and creates train/eval CSV files.

    Args:
        repo_id (str): The Hugging Face repository ID for the dataset.
        train_csv_path (str): Path where the training CSV file will be saved.
        eval_csv_path (str): Path where the evaluation CSV file will be saved.
        split_percent (int): Percentage of data to use for training (default: 80).
    """
    # Get the dataset from the specified repository
    raw_dataset, features_data = get_tfrecord_dataset(repo_id=repo_id)
    debug_first_episode(raw_dataset, features_data)

    parsed_dataset = raw_dataset.map(lambda x: parse_bridge_episode(x, features_data))
    base_output_dir = os.path.dirname(train_csv_path)
    csv_path = os.path.join(base_output_dir, "images_and_instructions.csv")
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            # Write the header row
            writer.writerow(["image_path", "language_instruction"])

            for i, episode in enumerate(parsed_dataset):
                episode_data = save_episode_images(episode, episode_num=i + 1, base_output_dir=base_output_dir)
                for csv_row in episode_data["csv_data"]:
                    writer.writerow(csv_row)

        print(f"Successfully generated {csv_path}")
        # Split the generated CSV into training and validation sets
        if 0 < split_percent < 100:
            print(f"Splitting CSV data into train and eval sets with split percent: {split_percent}")
            split_csv_data(csv_path, train_csv_path, eval_csv_path, split_percent=split_percent)
        else:
            print("Skipping CSV split as split_percent is not between 1 and 99.")
    except IOError as e:
        print(f"Error writing to CSV file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Process bridge dataset episodes from a Hugging Face repository.")

    # Add the argument for the repository ID
    parser.add_argument(
        "--repo_id",
        type=str,
        default="dusty-nv/bridge_orig_ep100",
        help="The Hugging Face repository ID to pull the dataset from.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="/configs/bridge_train_config.yml",
        help="The path to the yaml file containing the training configuration.",
    )

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    # Use config to get the output folder of data from train_dataset variable
    # this way only need to change train_dataset in config to change data folder
    base_output_dir = os.path.dirname(config_dict["train_dataset"])
    print("Saving custom dataset to: ", base_output_dir)
    os.makedirs(base_output_dir, exist_ok=True)
    if len(os.listdir(base_output_dir)) > 60:
        print(f"Output directory {base_output_dir} already contains >60 files, skipping dataset preparation.")
    else:
        main(args.repo_id, config_dict["train_dataset"], config_dict["eval_dataset"])
