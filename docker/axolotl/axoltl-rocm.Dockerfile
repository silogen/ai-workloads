# TODO: the Dockerfile is currently not available. Could be included here as a multi-stage build.
FROM ghcr.io/silogen/rocm6.2-vllm0.6.3-flash-attn2.6.3-wheels:static AS final_wheels

# Builder compiles binaries so they can be installed in the main image
FROM rocm/dev-ubuntu-22.04:6.2-complete AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    python3.10 \
    python3.10-dev \
    python3-pip \
    rocthrust-dev \
    hipsparse-dev \
    hipblas-dev

RUN pip install --upgrade pip \
    && pip install --upgrade setuptools

# a0a95fd is referred to as a working commit here:
# https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1519#issuecomment-2667244233
RUN git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git \
    && cd bitsandbytes/ \
    && git checkout "a0a95fd" \
    && cmake -DCOMPUTE_BACKEND="hip" -S . \
    && make \
    && pip wheel --no-deps . -w /wheels

FROM rocm/dev-ubuntu-22.04:6.2 AS final

ARG DEBIAN_FRONTEND=noninteractive

# This image is designed to be run as non-root user
ENV USER_NAME=user
ENV USER_ID=1000
ENV GROUP_NAME=silogen
ENV GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} ${GROUP_NAME} && \
    useradd -m -g ${GROUP_ID} -u ${USER_ID} ${USER_NAME} \
    && mkdir /workdir \
    && chown -hR ${USER_NAME}:${GROUP_NAME} /workdir \
    && usermod -a -G video,render ${USER_NAME}  # Need to be in video and render to access GPUs

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3.10 \
    python3.10-dev \
    python3-pip \
    rocthrust-dev \
    hipsparse-dev \
    hipblas-dev \
    wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    python -m pip install -U pip

# Install minio
RUN curl https://dl.min.io/client/mc/release/linux-amd64/mc \
    --create-dirs \
    -o /minio-binaries/mc && \
    chown -hR ${USER_NAME} /minio-binaries/ && \
    chmod +x /minio-binaries/mc

ENV PATH="${PATH}:/minio-binaries/:/root/scripts/"

# The flash attention wheel will not change frequently, so place its install early on.
COPY --from=final_wheels /wheels/*.whl /libs/
RUN pip install /libs/flash_attn*.whl --no-cache-dir --no-deps

RUN pip install /opt/rocm/share/amd_smi

# Some of these are set before they are created, but that is fine
ENV LLVM_SYMBOLIZER_PATH=/opt/rocm/llvm/bin/llvm-symbolizer
ENV PATH=$PATH:/opt/rocm/bin:/libtorch/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib:/libtorch/lib
ENV CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/libtorch/include:/libtorch/include/torch/csrc/api/include:/opt/rocm/include

# Install torch as a separate step for a couple of reasons:
#  - it creates a large layer, so it is useful to have a separate layer that gets cached
#  - with ROCm we want to manage the torch version separately, to ensure compatibility
# Also, we install Ray so that the Axolotl Ray launcher can be used.
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/rocm6.2 \
  && pip install -U "ray[data,train,tune,serve]"

# Notes on dependencies:
# - mamba-ssm is todo, it needs a ROCm patch
# - we do not install Nvidia management library with ROCm
# - newer versions of triton currently run into problems
# - specify torch version in the pip install command so that version doesn't get clobbered
# - install axolotl with --no-deps, so it doesn't clobber versions.
RUN git clone --branch "v0.8.1" https://github.com/axolotl-ai-cloud/axolotl.git && \
  sed -i \
    -e '/bitsandbytes/d' \
    -e '/mamba-ssm/d' \
    -e '/nvidia-ml-py/d' \
    -e 's/triton.*/triton==3.1.0/g' \
    ./axolotl/requirements.txt && \
  pip install \
    --no-cache \
    --extra-index-url https://download.pytorch.org/whl/rocm6.2 \
    -r ./axolotl/requirements.txt \
    "torch==2.5.1" && \
  pip install \
    --no-cache \
    --no-deps \
    ./axolotl

# The bitsandbytes installation may change relatively often so it's placed at the end of the installation
COPY --from=builder /wheels/*.whl /libs/
RUN pip install /libs/bitsandbytes*.whl --no-cache-dir

WORKDIR /workdir
