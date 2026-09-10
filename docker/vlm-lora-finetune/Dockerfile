FROM rocm/pytorch:rocm6.4.1_ubuntu24.04_py3.12_pytorch_release_2.7.1
COPY . /app
WORKDIR /app
RUN pip install --no-cache-dir -r amd_requirements.txt
CMD ["/bin/bash"]
