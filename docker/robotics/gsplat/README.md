# gsplat by Silogen using ROCm

https://github.com/rocm/gsplat

An open-source AMD-GPU-optimzed implementation of 3D gaussian splatting.

# Prerequisities
- Access to a server with AMD GPU available
- Docker installed

## Benchmark ##
1. Create the container and connect to it.
    ```sh
    docker build . -t gsplat:latest
    docker run --shm-size=128GB --device /dev/kfd --device /dev/dri -d --name gsplat gsplat:latest
    docker exec -it gsplat /bin/bash
    ```
2. Benchmark gsplat by running the following commands:
    ```sh
    # Downloading the dataset, single-GPU gsplat benchmarking, and converting the output to the markdown tables
    chmod +x /code/entrypoint.sh
    bash /code/entrypoint.sh
    ```
3. Delete the container
    ```sh
    docker stop gsplat
    docker image rm gsplat:latest
    ```
