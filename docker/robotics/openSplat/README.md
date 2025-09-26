# OpenSplat by Silogen using ROCm

https://github.com/pierotofy/OpenSplat

A free and open source implementation of 3D gaussian splatting written in C++, focused on being portable, lean and fast.

OpenSplat takes camera poses + sparse points in COLMAP, OpenSfM, ODM, OpenMVG or nerfstudio project format and computes a scene file (.ply or .splat) that can be later imported for viewing, editing and rendering in other software.

# Prerequisities
- Access to a server with AMD GPU available
- Docker installed

## Benchmark ##
1. Create the container and connect to it.
    ```sh
    docker build . -t opensplat:latest
    docker run --device /dev/kfd --device /dev/dri -d --name opensplat opensplat:latest
    docker exec -it opensplat /bin/bash
    ```
2. Benchmark OpenSplat by running the following commands:
    ```sh
    # Disconnect from the container above and go back to your local terminal
    docker cp benchmark.sh opensplat:/var/lib/jenkins/benchmark.sh
    docker exec -it opensplat /bin/bash
    chmod +x benchmark.sh

    # This script will eventually fail at some point, and thats intended. The point is to see how far it makes it before it fails.
    bash benchmark.sh &> opensplat.log
    # The following will reformat the benchmarking logs into two output tables (output_table1 and output_table2) similar to the tables in the section **Benchmarking Results**.
    python reformat_log.py
    ```
    **Note on benchmarking:** We cannot give the number of Gaussians as input to the OpenSplat (the number of Gaussians come from the sparse points in the input data, e.g., keypoints in COLMAP). We just can change splitting and densifying thresholds. But, there is a [simple_trainer script](https://github.com/pierotofy/OpenSplat/blob/main/simple_trainer.cpp) in the OpenSplat repository that we can use for testing with different image widths, heights, and different numbers of Gaussians. We need to see when the command results in Memory overflow on MI300 and compare it with H100.

    **Note:** simple_trainer script in OpenSplat is similar to [image_fitting script](https://github.com/nerfstudio-project/gsplat/blob/main/examples/image_fitting.py) in gsplat. [Here](https://docs.gsplat.studio/main/examples/image.html) is image_fitting documentation.
3. Delete the container
    ```sh
    docker stop opensplat
    docker image rm opensplat:latest
    ```
