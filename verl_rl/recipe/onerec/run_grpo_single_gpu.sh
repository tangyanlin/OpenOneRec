#!/bin/bash
# GRPO Training Script with Two-Stage Rollout — Single Node, Single GPU
# Adapted from run_grpo.sh for single-GPU environments without flash_attn dependency.
#
# Usage:
#   export BASE_MODEL=/path/to/your/model
#   bash run_grpo_single_gpu.sh [GPU_ID] [hostfile]
#
# Arguments:
#   GPU_ID   - GPU index to use, e.g. 0 or 1 (default: 0)
#   hostfile - path to hostfile (optional, auto-created if not provided)
#
# Optional environment variables:
#   CUDA_VISIBLE_DEVICES  - overrides GPU_ID if set (takes precedence)

set -x
HOME=$(pwd)
timestamp=$(date +"%Y-%m-%d-%H:%M:%S")
export RAY_DISABLE_DASHBOARD=1

# ===== GPU Selection =====
# Priority: CUDA_VISIBLE_DEVICES env var > GPU_ID argument > default (0)
GPU_ID="${1:-0}"
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=$GPU_ID
fi
echo "[INFO] Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# ===== Single GPU: create a local hostfile =====
HOSTFILE="${2:-}"
if [ -z "$HOSTFILE" ] || [ ! -f "$HOSTFILE" ]; then
    HOSTFILE="$HOME/tmp_hostfile_dir/hostfile_single_gpu_$timestamp"
    mkdir -p "$HOME/tmp_hostfile_dir"
    echo "127.0.0.1" > "$HOSTFILE"
fi
NODES=1

if [ ! -d "$HOME/tmp_hostfile_dir" ]; then
    mkdir -p "$HOME/tmp_hostfile_dir"
fi
if [ ! -d "$HOME/timeline_dir" ]; then
    mkdir -p "$HOME/timeline_dir"
fi
cat $HOSTFILE > "$HOME/tmp_hostfile_dir/hostfile_$timestamp"

# ===== Single GPU configuration =====
export N_NODES=1
export N_GPUS=1

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ============================================================================
# Model Configuration
# ============================================================================
# BASE_MODEL should point to a HuggingFace-format model directory containing
# config.json, tokenizer files, and model weights (model.safetensors or pytorch_model.bin).
#
# If pointing to a verl FSDP checkpoint (e.g., from verl_distillation), the script will
# automatically convert it to HuggingFace format using verl.model_merger before training.
#
# FSDP checkpoint structure: global_step_XXX/actor/  (contains model_world_size_*_rank_*.pt)
# HuggingFace structure:      global_step_XXX/actor/huggingface/  (contains config.json + tokenizer, but NO weights)
# After conversion:           global_step_XXX/actor/huggingface_merged/  (contains config.json + tokenizer + weights)
export BASE_MODEL=${BASE_MODEL:-"/home/jovyan/llm-dev-datavol-1/tangyanlin/AdOneModel/OpenOneRec/verl_distillation/outputs/ckpts/verl_on_policy_distill/verl_1.7b_distill_single_gpu_2026-06-12-15:54:47/global_step_136/actor"}
export ROLLOUT_TP_SIZE=1
# Use XFORMERS backend to avoid flash_attn dependency in vLLM rollout
export VLLM_ATTENTION_BACKEND=XFORMERS

# ============================================================================
# Training Hyperparameters
# ============================================================================
export LEARNING_RATE=${LEARNING_RATE:-2e-6}
export KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
export TEMPERATURE=${TEMPERATURE:-1}

# ============================================================================
# Batch Size Configuration (optimized for single GPU)
# ============================================================================
export USE_DYNAMIC_BSZ=True
export MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-6000}
export TRAIN_BATCH_SIZE=8

# ============================================================================
# Rollout Configuration
# ============================================================================
export ROLLOUT_N=${ROLLOUT_N:-1}
export STAGE2_BEAM_SIZE=${STAGE2_BEAM_SIZE:-8}
export RESPONSE_LENGTH=${RESPONSE_LENGTH:-512}
export STAGE1_MAX_TOKENS=${STAGE1_MAX_TOKENS:-512}
export STAGE2_NUM_TOKENS=${STAGE2_NUM_TOKENS:-3}

# Think mode configuration
export ENABLE_THINK=${ENABLE_THINK:-False}
export ENABLE_NONTHINK=${ENABLE_NONTHINK:-False}
export USE_FORCE_PREFIX=${USE_FORCE_PREFIX:-False}

# ============================================================================
# Data Configuration
# ============================================================================
export DATA_DIR=${DATA_DIR:-"$(realpath ../output/rl_data)"}
export TRAIN_FILES=${TRAIN_FILES:-"[$DATA_DIR/train.parquet]"}
export VAL_FILES=${VAL_FILES:-"[$DATA_DIR/test.parquet]"}

# ============================================================================
# Logging Configuration
# ============================================================================
export CONSOLE_LOG_INTERVAL=${CONSOLE_LOG_INTERVAL:-1000}

# ============================================================================
# Output Configuration
# ============================================================================
export PROJECT_NAME=${PROJECT_NAME:-"OneRec_RL"}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-"grpo_two_stage_single_gpu_${timestamp}"}
export OUTPUT_DIR=${OUTPUT_DIR:-"./output"}
export WANDB_MODE=${WANDB_MODE:-offline}

# ============================================================================
# Print Configuration
# ============================================================================
echo "============================================"
echo "GRPO Training with Two-Stage Rollout (Single GPU, No flash_attn)"
echo "============================================"
echo "Model: $BASE_MODEL"
echo "Cluster: $N_NODES nodes x $N_GPUS GPUs"
echo "Batch Size: $TRAIN_BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Rollout N: $ROLLOUT_N"
echo "Stage2 Beam Size: $STAGE2_BEAM_SIZE"
echo "Enable Think: $ENABLE_THINK"
echo "Enable NonThink: $ENABLE_NONTHINK"
echo "Console Log Interval: $CONSOLE_LOG_INTERVAL"
echo "============================================"

# ============================================================================
# FSDP Checkpoint → HuggingFace Format Conversion
# ============================================================================
# If BASE_MODEL points to a verl FSDP checkpoint directory (contains model_world_size_*_rank_*.pt
# but no model.safetensors/pytorch_model.bin), automatically convert it to HuggingFace format
# using verl.model_merger before training.
#
# Detection logic:
#   1. If BASE_MODEL already has model.safetensors or pytorch_model.bin → already HF format, skip
#   2. If BASE_MODEL has model_world_size_*_rank_*.pt files → FSDP format, need conversion
#   3. If BASE_MODEL/actor exists with FSDP files → checkpoint root, use BASE_MODEL/actor as FSDP dir
#   4. Otherwise → assume it's a standard HF model path (e.g., HuggingFace Hub model ID)

CONVERTED_MODEL_DIR=""

# Check if already HuggingFace format (has weight files)
if ls "$BASE_MODEL"/model.safetensors 2>/dev/null || ls "$BASE_MODEL"/pytorch_model.bin 2>/dev/null || ls "$BASE_MODEL"/model*.safetensors 2>/dev/null; then
    echo "[INFO] BASE_MODEL is already in HuggingFace format, skipping conversion."
# Check if BASE_MODEL itself is an FSDP actor directory
elif ls "$BASE_MODEL"/model_world_size_*_rank_0.pt 2>/dev/null; then
    FSDP_ACTOR_DIR="$BASE_MODEL"
    CONVERTED_MODEL_DIR="$FSDP_ACTOR_DIR/huggingface_merged"
    if [ -f "$CONVERTED_MODEL_DIR/model.safetensors" ] || [ -f "$CONVERTED_MODEL_DIR/pytorch_model.bin" ]; then
        echo "[INFO] FSDP→HF conversion already exists at $CONVERTED_MODEL_DIR, skipping."
    else
        echo "[INFO] Detected FSDP checkpoint at $FSDP_ACTOR_DIR"
        echo "[INFO] Converting FSDP checkpoint to HuggingFace format..."
        mkdir -p "$CONVERTED_MODEL_DIR"
        python3 -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "$FSDP_ACTOR_DIR" \
            --target_dir "$CONVERTED_MODEL_DIR"
        if [ $? -ne 0 ]; then
            echo "[ERROR] FSDP→HF conversion failed! Please run manually:"
            echo "  python3 -m verl.model_merger merge --backend fsdp --local_dir $FSDP_ACTOR_DIR --target_dir $CONVERTED_MODEL_DIR"
            exit 1
        fi
        echo "[INFO] FSDP→HF conversion completed: $CONVERTED_MODEL_DIR"
    fi
    export BASE_MODEL="$CONVERTED_MODEL_DIR"
# Check if BASE_MODEL is a checkpoint root directory (has actor/ subdirectory with FSDP files)
elif [ -d "$BASE_MODEL/actor" ] && ls "$BASE_MODEL/actor"/model_world_size_*_rank_0.pt 2>/dev/null; then
    FSDP_ACTOR_DIR="$BASE_MODEL/actor"
    CONVERTED_MODEL_DIR="$FSDP_ACTOR_DIR/huggingface_merged"
    if [ -f "$CONVERTED_MODEL_DIR/model.safetensors" ] || [ -f "$CONVERTED_MODEL_DIR/pytorch_model.bin" ]; then
        echo "[INFO] FSDP→HF conversion already exists at $CONVERTED_MODEL_DIR, skipping."
    else
        echo "[INFO] Detected FSDP checkpoint at $FSDP_ACTOR_DIR"
        echo "[INFO] Converting FSDP checkpoint to HuggingFace format..."
        mkdir -p "$CONVERTED_MODEL_DIR"
        python3 -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "$FSDP_ACTOR_DIR" \
            --target_dir "$CONVERTED_MODEL_DIR"
        if [ $? -ne 0 ]; then
            echo "[ERROR] FSDP→HF conversion failed! Please run manually:"
            echo "  python3 -m verl.model_merger merge --backend fsdp --local_dir $FSDP_ACTOR_DIR --target_dir $CONVERTED_MODEL_DIR"
            exit 1
        fi
        echo "[INFO] FSDP→HF conversion completed: $CONVERTED_MODEL_DIR"
    fi
    export BASE_MODEL="$CONVERTED_MODEL_DIR"
else
    echo "[INFO] BASE_MODEL does not appear to be a local FSDP checkpoint. Assuming it is a HuggingFace model ID or path."
fi

echo "[INFO] Final BASE_MODEL: $BASE_MODEL"

# ============================================================================
# Launch Training
# ============================================================================
mkdir -p logs

conda activate verl 2>/dev/null || true

PYTHONUNBUFFERED=1 python3 -u -m recipe.onerec.main_onerec_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    data.max_prompt_length=1024 \
    ++data.enable_think=$ENABLE_THINK \
    ++data.enable_nonthink=$ENABLE_NONTHINK \
    ++data.use_force_prefix=$USE_FORCE_PREFIX \
    data.prompt_key='prompt' \
    data.shuffle=True \
    data.max_response_length=$RESPONSE_LENGTH \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.custom_cls.path=$SCRIPT_DIR/onerec_recipe.py \
    data.custom_cls.name=OneRecDataset \
    data.reward_fn_key='source' \
    ++data.data_source_key='source' \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.use_remove_padding=False \
    custom_reward_function.path=$SCRIPT_DIR/onerec_recipe.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKENS_PER_GPU \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$MAX_TOKENS_PER_GPU \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAX_TOKENS_PER_GPU \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_TOKENS_PER_GPU \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.name=two_stage \
    ++actor_rollout_ref.rollout.backend=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    ++actor_rollout_ref.rollout.max_length=$RESPONSE_LENGTH \
    ++actor_rollout_ref.rollout.stage1_max_tokens=$STAGE1_MAX_TOKENS \
    ++actor_rollout_ref.rollout.stage2_num_tokens=$STAGE2_NUM_TOKENS \
    ++actor_rollout_ref.rollout.stage2_beam_size=$STAGE2_BEAM_SIZE \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.max_logprobs=320 \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=$N_NODES \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$OUTPUT_DIR/ckpt \
    trainer.total_epochs=1 \
    trainer.val_before_train=True \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    ++critic.enable=False \
    ++actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    ++actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    ++actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    "$@" 2>&1 | tee $PROJECT_NAME-$EXPERIMENT_NAME-$timestamp.log
