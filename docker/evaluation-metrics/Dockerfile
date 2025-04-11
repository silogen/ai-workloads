FROM python:3.11-slim

WORKDIR /home/evaluation
RUN chown -R 1000:1000 /home/evaluation
RUN mkdir -p /HF_HOME
RUN mkdir -p /home/evaluation/results
RUN chown -R 1000:1000 /HF_HOME
RUN chown -R 1000:1000 /home/evaluation/results
RUN apt update
RUN apt install wget -y
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install --no-cache-dir -e .
