FROM python:3.10-slim as base

WORKDIR /app

COPY docker/logistics/requirements.txt /app/

RUN apt-get update && apt-get install -y curl nano && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

# Install minio
RUN curl https://dl.min.io/client/mc/release/linux-amd64/mc \
    --create-dirs \
    -o /minio-binaries/mc && \
    chmod +x /minio-binaries/mc

ENV PATH="${PATH}:/minio-binaries/"
ENV HF_HOME="/hf_cache/"

RUN mkdir -p ${HF_HOME} local_models/ local_datasets


FROM base as non-root

# Create a non-root user
ARG USER_NAME=user
ARG USER_UID=1000
RUN groupadd --gid ${USER_UID} ${USER_NAME} && \
useradd --gid ${USER_UID} --uid ${USER_UID} -m ${USER_NAME}

RUN chown -hR ${USER_NAME} /minio-binaries/ local_models/ local_datasets/ ${HF_HOME}

USER ${USER_UID}:${USER_UID}
