import json
import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd


def format_time(seconds) -> str:
    """
    Formats a time value in seconds to a string in the format 'XmYY.YYs'.
    Returns 'N/A' if input is 'N/A' or NaN.

    Args:
        seconds: Time in seconds, or 'N/A', or None.

    Returns:
        Formatted time string.
    """
    if seconds == "N/A" or pd.isna(seconds):
        return "N/A"
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes}m{remaining_seconds:.2f}s"


def round_values(value: Any) -> Any:
    """
    Rounds numeric values to 2 decimal places, leaves others unchanged.

    Args:
        value: Value to round.

    Returns:
        Rounded value or original value.
    """
    if isinstance(value, (int, float)) and not pd.isna(value):
        return round(value, 2)
    return value


def format_num_gs(value: Any) -> Any:
    """
    Formats a number as millions with 'M' suffix.

    Args:
        value: Numeric value.

    Returns:
        Formatted string or original value.
    """
    if isinstance(value, (int, float)):
        return f"{value / 1e6:.2f}M"
    return value


def average_train_jsons(files: List[str]) -> Tuple[Union[float, str], Union[float, str]]:
    """
    Computes the average memory and elapsed time from a list of JSON files.

    Args:
        files: List of file paths to JSON files.

    Returns:
        Tuple of (average memory, average elapsed time), or 'N/A' if not available.
    """
    mems, times = [], []
    for f in files:
        with open(f, "r") as tf:
            d = json.load(tf)
            if "mem" in d and d["mem"] != "N/A":
                mems.append(d["mem"])
            if "ellipse_time" in d and d["ellipse_time"] != "N/A":
                times.append(d["ellipse_time"])
    mem_avg = np.mean(mems) if mems else "N/A"
    time_avg = np.mean(times) if times else "N/A"
    return mem_avg, time_avg


def create_scene_metrics_table(base_dir: str, mode: str = "single") -> pd.DataFrame:
    """
    Creates a table of scene metrics (PSNR, SSIM, LPIPS, memory, time) for each scene/configuration.

    Args:
        base_dir: Path to stats_basic[_suffix] directory.
        mode: 'single', 'multi', or 'multi8'.

    Returns:
        DataFrame with metrics for each scene/configuration.
    """
    data: Dict[str, List[Any]] = {
        "Scene": [],
        "Configuration": [],
        "PSNR": [],
        "SSIM": [],
        "LPIPS": [],
        "Train Mem (Gb)": [],
        "Train Time": [],
    }

    if mode == "single":
        configs = [("gsplat-7k(1GPU)", "6999"), ("gsplat-30k(1GPU)", "29999")]
        for scene in os.listdir(base_dir):
            scene_path = os.path.join(base_dir, scene)
            stats_path = os.path.join(scene_path, "stats")
            if os.path.isdir(stats_path):
                for config, step in configs:
                    train_file = os.path.join(stats_path, f"train_step{step}_rank0.json")
                    val_file = os.path.join(stats_path, f"val_step{step}.json")
                    if os.path.exists(train_file) and os.path.exists(val_file):
                        with open(train_file, "r") as tf:
                            train_data = json.load(tf)
                        with open(val_file, "r") as vf:
                            val_data = json.load(vf)
                        data["Scene"].append(scene)
                        data["Configuration"].append(config)
                        data["PSNR"].append(val_data.get("psnr", "N/A"))
                        data["SSIM"].append(val_data.get("ssim", "N/A"))
                        data["LPIPS"].append(val_data.get("lpips", "N/A"))
                        data["Train Mem (Gb)"].append(train_data.get("mem", "N/A"))
                        data["Train Time"].append(train_data.get("ellipse_time", "N/A"))
    elif mode == "multi":
        config = "gsplat-30k(4GPU)"
        step = "7499"
        for scene in os.listdir(base_dir):
            scene_path = os.path.join(base_dir, scene)
            stats_path = os.path.join(scene_path, "stats")
            if os.path.isdir(stats_path):
                train_files = [os.path.join(stats_path, f"train_step{step}_rank{r}.json") for r in range(4)]
                val_file = os.path.join(stats_path, f"val_step{step}.json")
                if all(os.path.exists(f) for f in train_files) and os.path.exists(val_file):
                    mem_avg, time_avg = average_train_jsons(train_files)
                    with open(val_file, "r") as vf:
                        val_data = json.load(vf)
                    data["Scene"].append(scene)
                    data["Configuration"].append(config)
                    data["PSNR"].append(val_data.get("psnr", "N/A"))
                    data["SSIM"].append(val_data.get("ssim", "N/A"))
                    data["LPIPS"].append(val_data.get("lpips", "N/A"))
                    data["Train Mem (Gb)"].append(mem_avg)
                    data["Train Time"].append(time_avg)
    elif mode == "multi8":
        config = "gsplat-30k(8GPU)"
        step = "3749"
        for scene in os.listdir(base_dir):
            scene_path = os.path.join(base_dir, scene)
            stats_path = os.path.join(scene_path, "stats")
            if os.path.isdir(stats_path):
                train_files = [os.path.join(stats_path, f"train_step{step}_rank{r}.json") for r in range(8)]
                val_file = os.path.join(stats_path, f"val_step{step}.json")
                if all(os.path.exists(f) for f in train_files) and os.path.exists(val_file):
                    mem_avg, time_avg = average_train_jsons(train_files)
                    with open(val_file, "r") as vf:
                        val_data = json.load(vf)
                    data["Scene"].append(scene)
                    data["Configuration"].append(config)
                    data["PSNR"].append(val_data.get("psnr", "N/A"))
                    data["SSIM"].append(val_data.get("ssim", "N/A"))
                    data["LPIPS"].append(val_data.get("lpips", "N/A"))
                    data["Train Mem (Gb)"].append(mem_avg)
                    data["Train Time"].append(time_avg)
    else:
        raise ValueError("mode must be 'single', 'multi', or 'multi8'")

    df = pd.DataFrame(data)

    # Calculate averages and stds for each configuration
    for config in df["Configuration"].unique():
        config_data = df[df["Configuration"] == config]
        avg = config_data.mean(numeric_only=True)
        std = config_data.std(numeric_only=True)
        # Compute mean and std for Train Time before formatting
        train_time_vals = pd.to_numeric(config_data["Train Time"], errors="coerce")
        train_time_avg = train_time_vals.mean()
        train_time_std = train_time_vals.std()
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "Scene": ["Average", "Std"],
                        "Configuration": [config, config],
                        "PSNR": [avg.get("PSNR", "N/A"), std.get("PSNR", "N/A")],
                        "SSIM": [avg.get("SSIM", "N/A"), std.get("SSIM", "N/A")],
                        "LPIPS": [avg.get("LPIPS", "N/A"), std.get("LPIPS", "N/A")],
                        "Train Mem (Gb)": [avg.get("Train Mem (Gb)", "N/A"), std.get("Train Mem (Gb)", "N/A")],
                        "Train Time": [train_time_avg, train_time_std],
                    }
                ),
            ],
            ignore_index=True,
        )

    # Ensure Train Time column is object dtype to allow string assignment
    df["Train Time"] = df["Train Time"].astype(object)

    # Format and round
    for i, row in df.iterrows():
        if row["Scene"] not in ["Average", "Std"]:
            df.at[i, "Train Time"] = format_time(row["Train Time"])
    df["PSNR"] = df["PSNR"].apply(round_values)
    df["SSIM"] = df["SSIM"].apply(round_values)
    df["LPIPS"] = df["LPIPS"].apply(round_values)
    df["Train Mem (Gb)"] = df["Train Mem (Gb)"].apply(round_values)
    for i, row in df.iterrows():
        if row["Scene"] in ["Average", "Std"]:
            df.at[i, "Train Time"] = format_time(row["Train Time"])

    return df


def create_num_gs_table(base_dir: str, mode: str = "single") -> pd.DataFrame:
    """
    Creates a table of the number of Gaussians for each scene/configuration.

    Args:
        base_dir: Path to stats_basic[_suffix] directory.
        mode: 'single', 'multi', or 'multi8'.

    Returns:
        DataFrame with number of Gaussians for each scene/configuration.
    """
    if mode == "single":
        num_gs_data: Dict[str, List[Any]] = {"Configuration": ["gsplat-7k(1GPU)", "gsplat-30k(1GPU)"]}
        for scene in os.listdir(base_dir):
            scene_path = os.path.join(base_dir, scene)
            stats_path = os.path.join(scene_path, "stats")
            if os.path.isdir(stats_path):
                num_gs_7k: Any = "N/A"
                num_gs_30k: Any = "N/A"
                val_7k_file = os.path.join(stats_path, "val_step6999.json")
                val_30k_file = os.path.join(stats_path, "val_step29999.json")
                if os.path.exists(val_7k_file):
                    with open(val_7k_file, "r") as vf:
                        val_data = json.load(vf)
                        num_gs_7k = val_data.get("num_GS", "N/A")
                if os.path.exists(val_30k_file):
                    with open(val_30k_file, "r") as vf:
                        val_data = json.load(vf)
                        num_gs_30k = val_data.get("num_GS", "N/A")
                num_gs_data[scene] = [num_gs_7k, num_gs_30k]
    elif mode == "multi":
        num_gs_data = {"Configuration": ["gsplat-30k(4GPU)"]}
        for scene in os.listdir(base_dir):
            scene_path = os.path.join(base_dir, scene)
            stats_path = os.path.join(scene_path, "stats")
            if os.path.isdir(stats_path):
                num_gs: Any = "N/A"
                val_file = os.path.join(stats_path, "val_step7499.json")
                if os.path.exists(val_file):
                    with open(val_file, "r") as vf:
                        val_data = json.load(vf)
                        num_gs = val_data.get("num_GS", "N/A")
                num_gs_data[scene] = [num_gs]
    elif mode == "multi8":
        num_gs_data = {"Configuration": ["gsplat-30k(8GPU)"]}
        for scene in os.listdir(base_dir):
            scene_path = os.path.join(base_dir, scene)
            stats_path = os.path.join(scene_path, "stats")
            if os.path.isdir(stats_path):
                num_gs = "N/A"
                val_file = os.path.join(stats_path, "val_step3749.json")
                if os.path.exists(val_file):
                    with open(val_file, "r") as vf:
                        val_data = json.load(vf)
                        num_gs = val_data.get("num_GS", "N/A")
                num_gs_data[scene] = [num_gs]
    else:
        raise ValueError("mode must be 'single', 'multi', or 'multi8'")

    for scene in num_gs_data.keys():
        if scene != "Configuration":
            num_gs_data[scene] = [format_num_gs(v) for v in num_gs_data[scene]]

    return pd.DataFrame(num_gs_data)


def save_table(df: pd.DataFrame, filename_prefix: str) -> None:
    """
    Saves a DataFrame to CSV and Markdown files.

    Args:
        df: DataFrame to save.
        filename_prefix: Prefix for output files.
    """
    df.to_csv(f"{filename_prefix}.csv", index=False)
    with open(f"{filename_prefix}.md", "w") as md_file:
        md_file.write("| " + " | ".join(df.columns) + " |\n")
        md_file.write("|" + "|".join(["-------"] * len(df.columns)) + "|\n")
        for _, row in df.iterrows():
            md_file.write("| " + " | ".join(map(str, row)) + " |\n")


if __name__ == "__main__":
    # Example usage:
    # For single GPU:
    single_base_dir = "./results/benchmark"
    df_single = create_scene_metrics_table(single_base_dir, mode="single")
    save_table(df_single, "scene_metrics_table")
    num_gs_df_single = create_num_gs_table(single_base_dir, mode="single")
    save_table(num_gs_df_single, "num_gs_table")
