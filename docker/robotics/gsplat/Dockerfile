FROM rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0

WORKDIR /code

# Install gsplat from the AMD-hosted PyPI repository
RUN pip install gsplat --index-url=https://pypi.amd.com/simple


# Get gsplat code, so we have the example scripts available, including the training one
RUN git clone --no-checkout https://github.com/rocm/gsplat.git
RUN cd /code/gsplat && git sparse-checkout init --cone && git sparse-checkout add examples && git checkout main

# Install requirements for the gsplat training script
RUN pip install -r /code/gsplat/examples/requirements.txt

COPY create_scene_metrics_tables.py /code/create_scene_metrics_tables.py
# entrypoint.sh is a wrapper script to download the dataset. run the single-GPU benchmark, and reformat the output metrics
COPY entrypoint.sh /code/entrypoint.sh

# RUN chmod +x /code/entrypoint.sh
# ENTRYPOINT ["/code/entrypoint.sh"]

# Default command to run if no arguments are provided to the container
CMD ["/bin/bash"]
