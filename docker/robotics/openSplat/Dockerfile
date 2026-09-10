# NOTE: AN AMD GPU IS NEEDED TO BUILD THIS DOCKERFILE. THE OPENSPLAT MAKEFILE REQUIRES IT.
FROM rocm/pytorch:rocm6.4.3_ubuntu22.04_py3.10_pytorch_release_2.6.0

RUN apt-get update && apt-get install -y git \
    && apt-get install -y libopencv-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workload
COPY . /workload/

# Build OpenSplat
RUN git clone https://github.com/pierotofy/OpenSplat
WORKDIR /workload/OpenSplat

# Latest working commit
RUN git checkout 86d2500d3f50de3fee933525a1217c06634f791f

RUN source activate py_3.10 && \
    mkdir build && \
    cd build && \
    cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DGPU_RUNTIME=HIP \
      -DHIP_ROOT_DIR=/opt/rocm \
      -DOPENSPLAT_BUILD_SIMPLE_TRAINER=ON \
      -DCMAKE_PREFIX_PATH=/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch \
      -DCMAKE_INSTALL_PREFIX=/code/install && \
    make

WORKDIR /workload

# Install gdown for downloading data
RUN pip install --upgrade gdown

RUN gdown --id 1mUUZFDo2swd6CE5vwPPkjN63Hyf4XyEv --no-check-certificate -O banana.zip
RUN gdown --id 1WWXo-GKo6d-yf-K1T1CswIdkdZrBNZ_e --no-check-certificate -O truck.zip

RUN unzip banana.zip -d /workload/
RUN unzip truck.zip -d /workload/
RUN rm banana.zip truck.zip

CMD ["/bin/bash"]
