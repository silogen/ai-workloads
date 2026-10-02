FROM python:3.10-slim AS base

WORKDIR /app

COPY docker/logistics/requirements.txt /app/

RUN apt-get update \
    && apt-get install -y curl nano apt-transport-https ca-certificates gnupg \
    && mkdir -p -m 755 /etc/apt/keyrings \
    && curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.35/deb/Release.key | gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg \
    && chmod 644 /etc/apt/keyrings/kubernetes-apt-keyring.gpg \
    && echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.35/deb/ /' | tee /etc/apt/sources.list.d/kubernetes.list \
    && chmod 644 /etc/apt/sources.list.d/kubernetes.list \
    && apt-get update \
    && apt-get install -y kubectl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

# Install minio
RUN curl https://dl.min.io/client/mc/release/linux-amd64/mc \
    --create-dirs \
    --location \
    -o /minio-binaries/mc && \
    chmod +x /minio-binaries/mc

ENV PATH="${PATH}:/minio-binaries/"
ENV HF_HOME="/hf_cache/"

RUN mkdir -p ${HF_HOME} local_models/ local_datasets


FROM base AS non-root

# Create a non-root user
ARG USER_NAME=user
ARG USER_UID=1000
RUN groupadd --gid ${USER_UID} ${USER_NAME} && \
useradd --gid ${USER_UID} --uid ${USER_UID} -m ${USER_NAME}

RUN chown -hR ${USER_NAME} /minio-binaries/ local_models/ local_datasets/ ${HF_HOME}

USER ${USER_UID}:${USER_UID}
