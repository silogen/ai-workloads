#!/bin/bash

# Set master address and port
MASTER_ADDR=localhost
MASTER_PORT=23731

# Command to run the text generation server
echo "Starting the text generation server with torchrun..."



torchrun \
    --nproc_per_node 1 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    /workspace/Megatron-LM/tools/run_text_generation_server.py \
    --use-checkpoint-args \
    --bf16 \
    --no-async-tensor-model-parallel-allreduce \
    --no-masked-softmax-fusion \
    --no-rope-fusion \
    --no-gradient-accumulation-fusion \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /workload/tokenizer/ \
    --load /workload/model/ \
    --micro-batch-size 1
