#!/bin/bash

set -e

BASE_MODEL_DIR=$1
MODEL_HOME=$2
STEP=$3
CONVERT_SEQCLS=${4:-""}

CKPT_DIR=${MODEL_HOME}/step${STEP}/global_step${STEP}

OUTPUT_DIR=$CKPT_DIR/converted

EXTRA_ARGS=""
if [ "$CONVERT_SEQCLS" = "seqcls_to_causal_lm" ]; then
    EXTRA_ARGS="--convert_seqcls_to_causal_lm"
    echo "Converting Qwen3ForSequenceClassification checkpoint to Qwen3ForCausalLM format"
fi

python3 tools/model_converter/convert_checkpoint_to_hf.py --checkpoint_dir $CKPT_DIR \
    --output_dir $OUTPUT_DIR \
    --source_hf_model_path $BASE_MODEL_DIR \
    $EXTRA_ARGS
