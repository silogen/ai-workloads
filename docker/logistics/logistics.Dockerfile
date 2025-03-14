FROM python:3.10-slim

WORKDIR /app

COPY docker/logistics/requirements.txt /app/

RUN apt-get update && apt-get install -y curl nano && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user
ARG USER_NAME=user
ARG USER_UID=1000
RUN groupadd --gid ${USER_UID} ${USER_NAME} && \
useradd --gid ${USER_UID} --uid ${USER_UID} -m ${USER_NAME}

# Install minio
RUN curl https://dl.min.io/client/mc/release/linux-amd64/mc \
    --create-dirs \
    -o /minio-binaries/mc

RUN chown -hR ${USER_NAME} /minio-binaries/ && \
    chmod +x /minio-binaries/mc

ENV PATH="${PATH}:/minio-binaries/"

RUN mkdir local_models/ && chown ${USER_NAME} local_models/ \
    && mkdir local_datasets && chown ${USER_NAME} local_datasets/

ENV HF_HOME="/hf_cache/"
RUN mkdir ${HF_HOME} && chown ${USER_NAME} ${HF_HOME}

USER ${USER_UID}:${USER_UID}
