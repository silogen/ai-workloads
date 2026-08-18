FROM python:3.11-slim

WORKDIR /home/evaluation
RUN chown -R 1000:1000 /home/evaluation
RUN mkdir -p /HF_HOME
RUN mkdir -p /home/evaluation/results
RUN chown -R 1000:1000 /HF_HOME
RUN chown -R 1000:1000 /home/evaluation/results
COPY requirements.txt requirements.txt
RUN apt update && apt install -y --no-install-recommends \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/rocm6.2.4 --trusted-host download.pytorch.org --default-timeout=2000
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install --no-cache-dir -e .
