#!/bin/bash
# Set master address and port
MASTER_ADDR=localhost
MASTER_PORT=23731
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1

echo "Starting the text generation server with torchrun..."

torchrun \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    /workspace/Megatron-LM/tools/run_text_generation_server.py \
    --use-checkpoint-args \
    --bf16 \
    --no-async-tensor-model-parallel-allreduce \
    --no-masked-softmax-fusion \
    --no-rope-fusion \
    --no-gradient-accumulation-fusion \
    --tensor-model-parallel-size $TENSOR_MODEL_PARALLEL_SIZE \
    --pipeline-model-parallel-size $PIPELINE_MODEL_PARALLEL_SIZE \
    --context-parallel-size $CONTEXT_PARALLEL_SIZE \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /workload/tokenizer/ \
    --load /workload/model/ \
    --micro-batch-size $MICRO_BATCH_SIZE
