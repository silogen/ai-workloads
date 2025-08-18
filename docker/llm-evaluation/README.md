# LLM Evaluation

Evaluation package used to run LLM evaluations (metrics evaluation, e.g. BERT-score, BLEU-score, and LLM-as-a-Judge evaluation). To be installed in a docker image used for running evaluation workloads using Kubernetes.

We have a default image available at `ghcr.io/silogen/evaluation-workloads-metrics:v0.1`, which can be used to run the evaluation workloads included in this repository. Please refer to `workloads/llm-evaluation-judge` and `workloads/llm-evaluation-metrics` for instructions on running the workloads and examples on setting up the docker images.

## Building your own image

We provide a `Makefile` to build and push a Docker image to a user image registry. Edit the Makefile for the desired image registry URL. See a tutorial [here](https://waltercode.medium.com/building-and-pushing-images-using-docker-and-makefiles-2d520b17f97e)

The `Makefile` uses the `Dockerfile` to build a Docker image. For further info about Dockerfiles, see [here](https://docs.docker.com/get-started/docker-concepts/building-images/writing-a-dockerfile/)
