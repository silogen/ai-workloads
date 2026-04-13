# Weather Model Inference and Visualization

Files in this directory are mounted to the workload at `/workload/mount`.

**Note:** Subdirectories and binary files are not supported.


##
- run model inference for weather forecasts using the `ai-models` plugin
- use simple tools to  visualize forecasts of models from ECMWF `ai-models`.
- analyze performance using some simple tools

## Quick Start
- Edit the `date` parameter in the `run_model.sh` file to set the date for the date for input data.
- The lead times are calculated from the input date and time

To set a different lead time, or model, use the commands below:

```bash
# Run with default settings
./run_model.sh

# Run with custom settings
bash run_model.sh --model_name=aurora --lead_time=24 --visualize=true
```

## Configuration

The script accepts these environment variables:

| Variable   | Default  | Description                                |
|------------|----------|--------------------------------------------|
| MODEL_NAME | aurora   | AI model to use (aurora, aurora-0.1-finetuned, aurora-2.5-finetuned, aurora-2.5-pretrained)             |
| LEAD_TIME  | 24       | Forecast lead time in hours                |
| VISUALIZE  | true     | Generate visualization animations          |

## Output

Forecast files are saved to `predictions/` directory.

Visualizations are generated as animated GIFs in:
- `outputs/{model_name}/{variable}.gif` for surface variables
- `outputs/{model_name}/level_{pressure}/{variable}.gif` for pressure level variables

## Scripts

- `run_model.sh` - Main script that downloads models. data and runs inference
- `grib_visualizer.py` - Creates visualizations (GIFs) from GRIB files

## Note
Currently tested only aurora.
