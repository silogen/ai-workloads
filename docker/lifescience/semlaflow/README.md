## Running inference job automatically (non-interactive)

In order to run SemlaFlow jobs automatically using the above image do the following:
- Set up config and output directory:

Put your config files as well any other files needed such as datasets or priors in `CONFIG_PATH`. In `OUTPUT_PATH`, the job will write output logs.

```sh
export DATA_PATH=<local_path_to_semlaflow_data_directory>
export OUTPUT_PATH=<local_path_to_save_output>
```

Then, the following command will run the job:

```sh
docker run -it --shm-size=256g --device=/dev/kfd --device=/dev/dri/renderD<RENDER_ID> --network host --ipc host --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $DATA_PATH:/data -v $OUTPUT_PATH:/output rocm-semlaflow <script> <output-file> --data_path /data/<path-to-dataset> <other_args>
```
