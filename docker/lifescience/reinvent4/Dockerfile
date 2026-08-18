FROM rocm/pytorch:rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.5.1

# Use bash to support string substitution.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Clone the Reinvet4 repository and use the stable 4.6.22 version
RUN git clone https://github.com/MolecularAI/REINVENT4
WORKDIR /REINVENT4
RUN git checkout v4.6.22
RUN wget -O priors/reinvent.prior "https://zenodo.org/records/15641297/files/reinvent.prior?download=1"

# Remove torch and torchvision from pyproject.toml
RUN sed -i.bak '/torch==/d' pyproject.toml && \
    sed -i.bak '/torchvision /d' pyproject.toml

# Now run the install script as usual
RUN python install.py cpu

COPY demo_notebooks/ notebooks/

# Download the chemprop model
RUN mkdir chemprop
RUN wget -q --show-progress -O chemprop/model.pt "https://www.dropbox.com/scl/fi/zpnqc9at5a5dnkzfdbo6g/model.pt?rlkey=g005yli9364uptd94d60jtg5c&e=1&dl=1"

# Copy entrypoint script
COPY entrypoint.sh /

CMD ["/bin/bash"]
