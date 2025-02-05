FROM rocm/vllm-dev:20250124
# rocm/pytorch-training:v25.2   => 92.5GB
# rocm/vllm-dev:20250124        => 35.9GB

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg \
        netbase \
        wget \
        s3fs \
        tzdata; \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        sudo \
        libatomic1; \
    rm -rf /var/lib/apt/lists/*

RUN curl https://dl.min.io/client/mc/release/linux-amd64/mc \
        --create-dirs \
        -o /usr/local/bin/mc; \
    chmod +x /usr/local/bin/mc

WORKDIR /home/

ARG RELEASE_TAG=openvscode-server-v1.96.4
ARG RELEASE_ORG=gitpod-io
ARG OPENVSCODE_SERVER_ROOT=/home/.openvscode-server

RUN set -eux; \
    if [ -z "${RELEASE_TAG}" ]; then \
        echo "The RELEASE_TAG build arg must be set." >&2 && \
        exit 1; \
    fi; \
    arch=$(uname -m); \
    if [ "${arch}" = "x86_64" ]; then \
        arch="x64"; \
    elif [ "${arch}" = "aarch64" ]; then \
        arch="arm64"; \
    elif [ "${arch}" = "armv7l" ]; then \
        arch="armhf"; \
    fi; \
    wget https://github.com/${RELEASE_ORG}/openvscode-server/releases/download/${RELEASE_TAG}/${RELEASE_TAG}-linux-${arch}.tar.gz; \
    tar -xzf ${RELEASE_TAG}-linux-${arch}.tar.gz; \
    mv -f ${RELEASE_TAG}-linux-${arch} ${OPENVSCODE_SERVER_ROOT}; \
    cp ${OPENVSCODE_SERVER_ROOT}/bin/remote-cli/openvscode-server ${OPENVSCODE_SERVER_ROOT}/bin/remote-cli/code; \
    rm -f ${RELEASE_TAG}-linux-${arch}.tar.gz

ARG USERNAME=ubuntu
ARG USER_UID=1337
ARG USER_GID=1337

RUN set -eux; \
    groupadd --gid $USER_GID $USERNAME; \
    useradd --uid $USER_UID --gid $USERNAME -m -s /bin/bash $USERNAME; \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME; \
    chmod 0440 /etc/sudoers.d/$USERNAME; \
    usermod -aG video $USERNAME;

RUN set -eux; \
    chmod g+rw /home; \
    mkdir -p /home/workspace; \
    chown -R $USERNAME:$USERNAME /home/workspace; \
    chown -R $USERNAME:$USERNAME ${OPENVSCODE_SERVER_ROOT}

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    HOME=/home/workspace \
    EDITOR=code \
    VISUAL=code \
    GIT_EDITOR="code --wait" \
    OPENVSCODE_SERVER_ROOT=/home/.openvscode-server

ENV OPENVSCODE_SERVER_ROOT=/home/.openvscode-server
ENV OPENVSCODE=/home/.openvscode-server/bin/openvscode-server

# Install VSCode extensions
SHELL ["/bin/bash", "-c"]
RUN \
    urls=(\
        https://open-vsx.org/api/ms-python/python/2024.23.0-dev/file/ms-python.python-2024.23.0-dev.vsix \
        https://open-vsx.org/api/GitHub/vscode-pull-request-github/0.102.0/file/GitHub.vscode-pull-request-github-0.102.0.vsix \
        https://open-vsx.org/api/ms-kubernetes-tools/vscode-kubernetes-tools/1.3.19/file/ms-kubernetes-tools.vscode-kubernetes-tools-1.3.19.vsix \
    )\
    && tdir=/tmp/exts && mkdir -p "${tdir}" && cd "${tdir}" \
    && wget "${urls[@]}" && \
    exts=(\
        gitpod.gitpod-theme \
        "${tdir}"/* \
    )\
    && for ext in "${exts[@]}"; do ${OPENVSCODE} --install-extension "${ext}"; done

EXPOSE 3000
#USER $USERNAME

WORKDIR /home/workspace/
ENTRYPOINT ["/bin/sh", "-c", "exec ${OPENVSCODE_SERVER_ROOT}/bin/openvscode-server --host 0.0.0.0 --without-connection-token \"${@}\"", "--"]