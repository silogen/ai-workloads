FROM gitpod/openvscode-server:latest

ENV OPENVSCODE_SERVER_ROOT="/home/.openvscode-server"
ENV OPENVSCODE="${OPENVSCODE_SERVER_ROOT}/bin/openvscode-server"

SHELL ["/bin/bash", "-c"]
RUN \
    # Direct download links to external .vsix not available on https://open-vsx.org/
    # The two links here are just used as example, they are actually available on https://open-vsx.org/
    urls=(\
        https://open-vsx.org/api/ms-python/python/2024.23.0-dev/file/ms-python.python-2024.23.0-dev.vsix \
        https://open-vsx.org/api/GitHub/vscode-pull-request-github/0.102.0/file/GitHub.vscode-pull-request-github-0.102.0.vsix \
        https://open-vsx.org/api/ms-kubernetes-tools/vscode-kubernetes-tools/1.3.19/file/ms-kubernetes-tools.vscode-kubernetes-tools-1.3.19.vsix \
    )\
    # Create a tmp dir for downloading
    && tdir=/tmp/exts && mkdir -p "${tdir}" && cd "${tdir}" \
    # Download via wget from $urls array.
    && wget "${urls[@]}" && \
    # List the extensions in this array
    exts=(\
        # From https://open-vsx.org/ registry directly
        gitpod.gitpod-theme \
        # From filesystem, .vsix that we downloaded (using bash wildcard '*')
        "${tdir}"/* \
    )\
    # Install the $exts
    && for ext in "${exts[@]}"; do ${OPENVSCODE} --install-extension "${ext}"; done
