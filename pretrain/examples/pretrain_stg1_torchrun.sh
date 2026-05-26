#!/bin/bash

# torchrun-based training script for Stage 1 pretraining
# Converts from MPI-based distributed training to torchrun
#
# Usage:
#   1. Edit the configuration section below
#   2. Run: bash examples/pretrain_stg1_torchrun.sh

set -x

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Change to the pretrain directory (parent of examples)
cd "$SCRIPT_DIR"

echo "Working directory: $(pwd)"

# Get absolute path to this directory
PRETRAIN_DIR=$(pwd)
RECIPES_DIR="${PRETRAIN_DIR}/recipes"
DATASET_CONFIG_DIR="${PRETRAIN_DIR}/examples/dataset_config"

# ============== Configuration ==============
MODEL_DIR=${MODEL_DIR:-/home/jovyan/llm-dev-datavol-1/tangyanlin/AdOneModel/OpenOneRec/Qwen3-0.6B_itemic}
OUTPUT_DIR=${OUTPUT_DIR:-/home/jovyan/llm-dev-datavol-1/tangyanlin/AdOneModel/OpenOneRec/model_output/stg1_torchrun}
DATA_PATH=${DATA_PATH:-/home/jovyan/llm-dev-datavol-1/tangyanlin/AdOneModel/OpenOneRec/data/pretrain_item_understand.parquet}

# Number of nodes and GPUs per node
NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}

# Master address and port (for multi-node training)
# For single-node training, these are auto-configured by torchrun
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}

# Rendezvous ID (unique job ID for multi-node training)
RDZV_ID=${RDZV_ID:-"pretrain_stg1_$(date +%s)"}
# ============== End Configuration ==============

mkdir -p $OUTPUT_DIR

echo `date '+%Y-%m-%d %H:%M:%S'` >> $OUTPUT_DIR/task_info.log
echo "script: ${SCRIPT_FILE}" >> $OUTPUT_DIR/task_info.log
echo "=========================" >> $OUTPUT_DIR/task_info.log

echo "Output: $OUTPUT_DIR"
echo "NNODES: $NNODES, NPROC_PER_NODE: $NPROC_PER_NODE"
echo "MASTER_ADDR: $MASTER_ADDR, MASTER_PORT: $MASTER_PORT"

export PYTHONPATH=$PRETRAIN_DIR:$PYTHONPATH

source set_env.sh

# Disable NCCL configuration (using Gloo backend instead)
export NCCL_IB_DISABLE=1
unset NCCL_IB_GID_INDEX NCCL_IB_HCA NCCL_DEBUG NCCL_IB_QPS_PER_CONNECTION NCCL_NET_OVERHEAD NCCL_IB_TIMEOUT NCCL_SOCKET_IFNAME

# Disable proxy for internal communication
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# Training parameters
# Note: torchrun needs to be run from the pretrain directory
# torchrun executes the Python script directly, not through bash -c
# Use absolute path to the Python script
torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${NPROC_PER_NODE} \
    --rdzv_id=${RDZV_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    --max_restarts=3 \
    --monitor-interval=10 \
    ${RECIPES_DIR}/train_qwen3.py \
        --model_dir $MODEL_DIR \
        --output_dir $OUTPUT_DIR \
        --data_path $DATA_PATH \
        --freeze_llm \
        --use_tie_weights \
        --start_optimize_embedding_index 151669 \
        --model_class Qwen3ForCausalLM \
        --monitor_datasource_loss \
        --monitor_datasource_cnt \
        --max_length 8192 \
        --learning_rate 2e-4 \
        --min_lr 1e-4 \
        --weight_decay 0.1 \
        --max_grad_norm 1.0 \
        --lr_scheduler_type cosine \
        --num_warmup_steps 200 \
        --num_training_steps 2000 \
        --save_checkpoint_per_step 50 \
        --minibatch_size 16384 \
        --logging_per_step 5 \
        --use_fp32_weight \
        --seed 19260817 \
        --enable_profiler \
        --enable_gradient_checkpointing \
        --use_chunked_loss_computer \
   # > $OUTPUT_DIR/stdout.log 2>$OUTPUT_DIR/stderr.log &

echo "Training started in background. Check logs at:"
echo "  - stdout: $OUTPUT_DIR/stdout.log"
echo "  - stderr: $OUTPUT_DIR/stderr.log"
echo "  - tensorboard: tensorboard --logdir=$OUTPUT_DIR/log"
