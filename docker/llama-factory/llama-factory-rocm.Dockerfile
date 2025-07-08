FROM rocm/pytorch-training:v25.5

# Install llama-factory and dependencies
RUN git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git && \
  pip install "./LLaMA-Factory[torch,metrics]" "ray[data,train,tune,serve]" deepspeed==0.16.5 transformers==4.49

# Install minio
RUN curl https://dl.min.io/client/mc/release/linux-amd64/mc \
    --create-dirs \
    -o /minio-binaries/mc && \
    chmod +x /minio-binaries/mc
ENV PATH="${PATH}:/minio-binaries/:/root/scripts/"
