FROM rocm/pytorch:rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.8.0

RUN apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        ca-certificates \
    && apt-get autoremove -y \
    && apt-get clean

RUN pip install numpy==1.26.2 pandas==2.2.2 scipy==1.11.4 rdkit lightning torchmetrics openbabel-wheel typing_extensions wandb numba hiredis tqdm ipython certifi

RUN git clone https://github.com/rssrwn/semla-flow.git
WORKDIR /semla-flow

COPY entrypoint.sh /entrypoint.sh

CMD ["/bin/bash"]
