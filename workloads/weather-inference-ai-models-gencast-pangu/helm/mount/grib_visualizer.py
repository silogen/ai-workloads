"""
GRIB File Visualizer

This script loads a GRIB file and creates animated GIFs
for each variable over time, optionally across pressure levels.

Usage:
    python grib_visualizer.py --input <path_to_grib_file> [--output <output_dir>]
"""

import os
import sys
from typing import List, Optional, Tuple

import cfgrib
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def open_grib_datasets(path: str) -> List[xr.Dataset]:
    return cfgrib.open_datasets(path, decode_timedelta=xr.coding.times.CFTimedeltaCoder)


def describe_datasets(datasets: List[xr.Dataset], path: str) -> List[dict]:
    descriptions = []
    for ds_idx, ds in enumerate(datasets):
        model_name = "gencast" if "gencast" in path else ("panguweather" if "pangu" in path else "other")
        descriptions.append(
            {
                "model_name": model_name,
                "dataset_idx": ds_idx,
                "dataset": ds,
                "has_levels": "isobaricInhPa" in ds,
                "has_time": "time" in ds.dims,
                "has_step": "step" in ds.dims,
            }
        )
    return descriptions


def select_data(
    data: xr.DataArray, has_step: bool, has_time: bool, max_steps: int, level: Optional[int] = None
) -> xr.DataArray:
    if has_step and data.sizes.get("step", 0) > max_steps:
        data = data.isel(step=range(max_steps))
    if has_time and data.sizes.get("time", 0) > max_steps:
        data = data.isel(time=range(max_steps))
    if level is not None and "isobaricInhPa" in data.coords:
        data = data.sel(isobaricInhPa=level)
    return data.sortby("latitude")


def compute_scale(
    data: xr.DataArray, center: Optional[float] = None, robust: bool = False
) -> Tuple[matplotlib.colors.Normalize, str]:
    vmin = np.nanpercentile(data, 2 if robust else 0)
    vmax = np.nanpercentile(data, 98 if robust else 100)
    if center is not None:
        diff = max(vmax - center, center - vmin)
        vmin = center - diff
        vmax = center + diff
    return matplotlib.colors.Normalize(vmin, vmax), ("RdBu_r" if center is not None else "viridis")


def save_variable_gif(
    data: xr.DataArray,
    time_dim: str,
    cmap: str,
    norm: matplotlib.colors.Normalize,
    var: str,
    frames: int,
    out_path: str,
    nanoseconds_per_step: Optional[int],
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data.isel({time_dim: 0}, missing_dims="ignore"), cmap=cmap, norm=norm, origin="lower")
    ax.set_title(f"{var} ({time_dim})")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.4, pad=0.01)

    def update(frame):
        im.set_data(data.isel({time_dim: frame}, missing_dims="ignore"))
        ax.set_title(f"{var} {time_dim}={frame}")

    def step_to_fps(nanoseconds: int) -> float:
        hours = nanoseconds / (60 * 60 * 1e9)
        return 2 * 24 / hours if hours > 0 else 1.0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ani = animation.FuncAnimation(fig, update, frames=frames)
    fps = step_to_fps(nanoseconds_per_step) if nanoseconds_per_step else 5
    ani.save(out_path, writer="pillow", fps=fps)
    plt.close(fig)


def process_dataset(
    ds: xr.Dataset,
    dataset_idx: int,
    output_dir: str,
    model_name: str,
    has_levels: bool,
    has_time: bool,
    has_step: bool,
    max_steps: int,
) -> None:
    step_to_nanoseconds = int(ds.get("step")[1]) if has_step and "step" in ds else None
    model_output_dir = os.path.join(output_dir, model_name)

    for var in ds.data_vars:
        variable_data = ds[var]
        levels = (
            [int(level) for level in variable_data.coords["isobaricInhPa"].values]
            if has_levels and "isobaricInhPa" in variable_data.coords
            else [None]
        )

        print(f"""Processing: {var} on {'surface_level' if levels == [None] else 'pressure levels'}""")

        for level in levels:
            data = select_data(variable_data, has_step, has_time, max_steps, level)
            if has_time and has_step:
                data = data.isel(time=0)

            time_dim = "step" if has_step else ("time" if has_time else None)
            if not time_dim:
                print(f"""Skipping {var}, no time dimension. Dataset #{dataset_idx} likely static map.""")
                continue

            frames = data.sizes[time_dim]
            norm, cmap = compute_scale(data)
            prefix = f"level_{level}" if level is not None else ""
            gif_name = f"{var}.gif"
            out_path = os.path.join(model_output_dir, prefix, gif_name)
            save_variable_gif(data, time_dim, cmap, norm, var, frames, out_path, step_to_nanoseconds)


def main():
    if "--input" not in sys.argv:
        print("Usage: python grib_visualizer.py --input <path_to_grib_file> [--output <output_dir>]")
        sys.exit(1)

    input_file = sys.argv[sys.argv.index("--input") + 1]

    if "--output" in sys.argv:
        output_dir = sys.argv[sys.argv.index("--output") + 1]
    else:
        output_dir = "outputs/"

    input_file = sys.argv[sys.argv.index("--input") + 1]
    datasets = open_grib_datasets(input_file)
    descriptions = describe_datasets(datasets, input_file)

    for desc in descriptions:
        print("Processing dataset #", desc["dataset_idx"])
        process_dataset(
            ds=desc["dataset"],
            dataset_idx=desc["dataset_idx"],
            output_dir=output_dir,
            model_name=desc["model_name"],
            has_levels=desc["has_levels"],
            has_time=desc["has_time"],
            has_step=desc["has_step"],
            max_steps=desc["dataset"].sizes.get("step", 40),
        )


if __name__ == "__main__":
    main()
