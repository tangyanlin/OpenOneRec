#!/bin/bash
# On-policy Distillation: Single Node, Single GPU
# Adapted from run_qwen3_distill.sh for single-GPU environments.
#
# Usage:
#   export BASE_MODEL=/path/to/student_model
#   export TEACHER_MODEL=/path/to/teacher_model
#   export DATASET_PARQUET=/path/to/train.parquet
#   bash run_qwen3_distill_single_gpu.sh [GPU_ID]
#
# Arguments:
#   GPU_ID  - GPU index to use, e.g. 0 or 1 (default: 0)
#
# Optional environment variables:
#   CUDA_VISIBLE_DEVICES  - overrides GPU_ID if set (takes precedence)
#   EXTEND_VOCAB_START_TOKEN  - extended vocab token threshold (default: 151669)
#   MASK_RESPONSE_IF_HAVE_EXTEND_TOKEN  - mask response with extended tokens (default: False)

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
N_GPUS_PER_NODE=1

project_name="verl_on_policy_distill"

experiment_name="verl_1.7b_distill_single_gpu_${timestamp}"
CKPT_HOME=${CKPT_HOME:-"$HOME/outputs"}
CKPT_DIR=${CKPT_DIR:-"${CKPT_HOME}/ckpts/${project_name}/${experiment_name}/"}

# Use HuggingFace rollout backend (more memory-efficient for single GPU)
rollout_name="vllm"
export HYDRA_FULL_ERROR=1

# ===== Open-source friendly defaults =====
export BASE_MODEL=${BASE_MODEL:-"../model_output/label_pred_seqcls/step2048/global_step2048/converted"}
export TEACHER_MODEL=${TEACHER_MODEL:-"../Qwen3-0.6B"}
export DATASET_PARQUET=${DATASET_PARQUET:-"$(realpath ../output/onpolicy_distillation.parquet)"}

# Logging: default is console only.
export WANDB_API_KEY=${WANDB_API_KEY:-""}

if [ -z "$BASE_MODEL" ] || [ -z "$TEACHER_MODEL" ] || [ -z "$DATASET_PARQUET" ]; then
  echo "[ERROR] Please set BASE_MODEL / TEACHER_MODEL / DATASET_PARQUET before running."
  echo "  BASE_MODEL=$BASE_MODEL"
  echo "  TEACHER_MODEL=$TEACHER_MODEL"
  echo "  DATASET_PARQUET=$DATASET_PARQUET"
  exit 1
fi

# ===== Memory optimization for single GPU =====
export USE_DYNAMIC_BSZ=True
export MAX_TOKENS_PER_GPU=12000  # reduced for single GPU (n*(prompt_len+response_len))

# Smaller batch size for single GPU
export TRAIN_BATCH_SIZE=64
export LEARNING_RATE=5e-6

export ROLLOUT_N=1
export TEMPERATURE=1.1
export ENABLE_THINK=True
export THINK_MODE="auto"
export MAX_RESPONSE_LEN=512  # reduced from 2048 to save memory
export MAX_PROMPT_LENGTH=512  # reduced from 4096 to save vLLM KV cache memory

export DISTILL_ADV_MAX=5.0
export DISTILL_ADV_MIN=-30.0

# ===== Extended vocabulary distillation settings =====
export EXTEND_VOCAB_START_TOKEN=${EXTEND_VOCAB_START_TOKEN:-151669}
export MASK_RESPONSE_IF_HAVE_EXTEND_TOKEN=${MASK_RESPONSE_IF_HAVE_EXTEND_TOKEN:-False}

export TRAIN_FILES=$DATASET_PARQUET
export VAL_FILES=$DATASET_PARQUET

echo "============================================"
echo "Single GPU On-Policy Distillation"
echo "============================================"
echo "Student model: $BASE_MODEL"
echo "Teacher model: $TEACHER_MODEL"
echo "Dataset:       $TRAIN_FILES"
echo "GPUs:          $N_GPUS_PER_NODE"
echo "Batch size:    $TRAIN_BATCH_SIZE"
echo "Max response:  $MAX_RESPONSE_LEN"
echo "Rollout:       $rollout_name"
echo "============================================"

PYTHONUNBUFFERED=1 python3 -m recipe.onpolicy_distill.main_onpolicy_distill --config-name='onpolicy_distill_trainer'\
    +ray_kwargs.ray_init.runtime_env.env_vars.TRACE_GPU_MEM=False \
    +ray_kwargs.ray_init.runtime_env.env_vars.WORK_DIR=$HOME \
    +ray_kwargs.ray_init.runtime_env.env_vars.WANDB_API_KEY="$WANDB_API_KEY" \
    +ray_kwargs.ray_init.runtime_env.env_vars.nosp="1" \
    +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_DEBUG="VERSION" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONWARNINGS="ignore" \
    algorithm.adv_estimator=on_policy_distill \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    ++data.enable_think=$ENABLE_THINK \
    ++data.think_mode=$THINK_MODE \
    data.prompt_key=prompt \
    data.image_key=dummy \
    data.video_key=dummy \
    ++data.data_source_key='source' \
    data.reward_fn_key='source' \
    data.max_response_length=$MAX_RESPONSE_LEN \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKENS_PER_GPU \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$MAX_TOKENS_PER_GPU \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAX_TOKENS_PER_GPU \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.actor.optim.lr=${LEARNING_RATE} \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.path=$BASE_MODEL \
    +actor_rollout_ref.ref.model.path=$TEACHER_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.ref_log_prob_replace_val=-100 \
    actor_rollout_ref.ref.ref_log_prob_replace_val=-100 \
    actor_rollout_ref.rollout.name=$rollout_name \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.extend_vocab_start_token=$EXTEND_VOCAB_START_TOKEN \
    actor_rollout_ref.rollout.mask_response_if_have_extend_token=$MASK_RESPONSE_IF_HAVE_EXTEND_TOKEN \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=200 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.max_num_seqs=4 \
    actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=1 \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    algorithm.use_kl_in_reward=False \
    ++algorithm.distill_adv_max_clip=$DISTILL_ADV_MAX \
    ++algorithm.distill_adv_min_clip=$DISTILL_ADV_MIN \
    actor_rollout_ref.actor.loss_agg_mode="token-mean" \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    trainer.logger='[console]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=$NODES \
    trainer.save_freq=50 \
    trainer.max_actor_ckpt_to_keep=100 \
    trainer.test_freq=-1 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.val_before_train=False \
    trainer.val_only=False \
    trainer.rollout_data_dir=$HOME \
    +trainer.validation_data_dir=$HOME \
    +trainer.ray_timeline_dir=$HOME/tmp_hostfile_dir \
    trainer.total_epochs=1 2>&1 | tee $project_name-$experiment_name-$timestamp.log
