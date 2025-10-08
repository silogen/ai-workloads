# Running Reinvent inference interactively

Connect to the pod with your favorite terminal.

This repo provides an altered version of these notebooks to be runnable from the terminal with the subscript `_clean`. These can simply be run by:

```sh
python3 notebooks/<notebook_name.py>
```
Alternatively, Reinvent jobs can be run by:
```sh
reinvent -l <log_name> <config_name>
```

## Running inference job automatically (non-interactive)

In order to run Reinvent jobs automatically using the above image do the following:
- Set up config and output directory:

Put your config files as well any other files needed such as datasets or priors in `CONFIG_PATH`. In `OUTPUT_PATH`, the job will write output logs.
```sh
export CONFIG_PATH=<local_path_to_configs>
export OUTPUT_PATH=<local_path_to_output>
```

Then, the following command will run the job:

```sh
docker run --rm -v $CONFIG_PATH:/data -v $OUTPUT_PATH:/output  --device=/dev/kfd --device=/dev/dri/renderD<RENDER_ID> rocm-reinvent /data/<config_file>.toml /output/<output_log_name>
```
where the last two arguments provide paths to the config file to run as well where to save outputs.
