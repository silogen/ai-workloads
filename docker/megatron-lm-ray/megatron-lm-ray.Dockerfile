ARG BASE_IMAGE=rocm/megatron-lm:v25.4

FROM ${BASE_IMAGE}

RUN pip install --upgrade pip && pip install --no-cache-dir "ray[data,train,tune,serve]"

CMD ["/bin/bash"]
