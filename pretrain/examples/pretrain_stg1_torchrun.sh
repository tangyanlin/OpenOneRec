#!/bin/bash

# torchrun-based training script for Stage 1 pretraining
# Converts from MPI-based distributed training to torchrun
#
# Usage:
#   1. Edit the configuration section below
#   2. Run: bash examples/pretrain_stg1_torchrun.sh

set -x

# ============== Configuration ==============
MODEL_DIR=/code/hf_models/Qwen3-1.7B_itemic
OUTPUT_DIR=/code/onerec_pretrain/model_output/stg1_torchrun

# Number of nodes and GPUs per node
NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

# Master address and port (for multi-node training)
# For single-node training, these are auto-configured by torchrun
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}

# Rendezvous ID (unique job ID for multi-node training)
RDZV_ID=${RDZV_ID:-"pretrain_stg1_$(date +%s)"}
# ============== End Configuration ==============

mkdir -p $OUTPUT_DIR

SCRIPT_FILE=$(readlink -f $0)
echo `date '+%Y-%m-%d %H:%M:%S'` >> $OUTPUT_DIR/task_info.log
echo "script: ${SCRIPT_FILE}" >> $OUTPUT_DIR/task_info.log
echo "=========================" >> $OUTPUT_DIR/task_info.log

echo "Output: $OUTPUT_DIR"
echo "NNODES: $NNODES, NPROC_PER_NODE: $NPROC_PER_NODE"
echo "MASTER_ADDR: $MASTER_ADDR, MASTER_PORT: $MASTER_PORT"

export PYTHONPATH=$PWD:$PYTHONPATH

source set_env.sh

# NCCL and network configuration
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=${TCP_NIC:-$(ifconfig | grep -B1 " "$(hostname -i)" " | grep -o "^\w*")}
export NCCL_IB_HCA=mlx5
export NCCL_DEBUG=WARN
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_NET_OVERHEAD=1000
export NCCL_IB_TIMEOUT=20

# Disable proxy for internal communication
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# Training parameters
torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${NPROC_PER_NODE} \
    --rdzv_id=${RDZV_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    --max_restarts=3 \
    --monitor-interval=10 \
    bash -c "bash scripts/numa_runner.sh python3 recipes/train_qwen3.py \
        --model_dir $MODEL_DIR \
        --output_dir $OUTPUT_DIR \
        --dataset_config examples/dataset_config/pretrain.json \
        --freeze_llm \
        --use_tie_weights \
        --start_optimize_embedding_index 151669 \
        --model_class Qwen3ForCausalLM \
        --monitor_datasource_loss \
        --monitor_datasource_cnt \
        --max_length 32768 \
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
    " > $OUTPUT_DIR/stdout.log 2>$OUTPUT_DIR/stderr.log &

echo "Training started in background. Check logs at:"
echo "  - stdout: $OUTPUT_DIR/stdout.log"
echo "  - stderr: $OUTPUT_DIR/stderr.log"
echo "  - tensorboard: tensorboard --logdir=$OUTPUT_DIR/log"