# Tested with image: rocm/vllm-dev:20250205_aiter

# References:
# https://unsloth.ai/blog/deepseekr1-dynamic
# https://medium.com/@alexhe.amd/deploy-deepseek-r1-in-one-gpu-amd-instinct-mi300x-7a9abeb85f78

export TMPDIR=/tmp
cd /workload || exit

# Update package index and install required dependencies
apt update && apt install -y curl libssl-dev libcurl4-openssl-dev

# Clone Llama.cpp repository and build it with HIP support
rm -rf llama.cpp &&
    git clone https://github.com/ggerganov/llama.cpp.git &&
    cd llama.cpp || exit &&
    # Configure HIP environment variables and build Llama.cpp
    HIPCXX="$(hipconfig -l)/clang" \
    HIP_PATH="$(hipconfig -R)" \
        cmake -S . \
        -B build \
        -DGGML_HIP=ON \
        -DAMDGPU_TARGETS=${ROCM_ARCH:-gfx942} \
        -DCMAKE_BUILD_TYPE=Release &&
    # Build Llama.cpp in Release configuration with multiple threads
    cmake --build build \
        --config Release \
        -- -j ${NCPU:-4}

# Serve the downloaded model through an OpenAI-compatible API
/workload/llama.cpp/build/bin/llama-server \
    -hf $MODEL \
    --host 0.0.0.0 \
    --port ${PORT:-8080} \
    --n-gpu-layers ${GPU_LAYERS:-100} \
    --threads ${NCPU:-4} \
    --prio 2 \
    --temp ${TEMP:-0.6} \
    --ctx-size ${CTX_SIZE:-4096}
