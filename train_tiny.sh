#!/bin/bash
# Usage: MODEL=tiny_gdn2_3L TRAIN_DATA_RAW=... TOKENIZER_PATH=... OUTPUT_ROOT=... ./train_tiny.sh
# MODEL is tiny_{gdn2,iso_kdn,diag_kdn,mesa_net}_{2,3}L. Extra arguments go to pretrain.py.
set -euo pipefail

: "${WANDB_API_KEY:?set WANDB_API_KEY}"
: "${TRAIN_DATA_RAW:?set TRAIN_DATA_RAW to the Parquet directory}"
: "${TOKENIZER_PATH:?set TOKENIZER_PATH to the tokenizer directory}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT to the output directory}"

MODEL="${MODEL:-tiny_iso_kdn_3L}"
TRAIN_CONFIG="${TRAIN_CONFIG:-tsz128x4k_5B}"

cd "$(dirname "$0")"

exec "${PYTHON:-python}" pretrain.py \
  --model_name "$MODEL" \
  --train_config "$TRAIN_CONFIG" \
  --exp_name "${NAME:-${MODEL}_${TRAIN_CONFIG}}" \
  --exp_group "${GROUP:-tiny}" \
  --output_root "$OUTPUT_ROOT" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --use_stream_tok \
  --train_data_dir_raw "$TRAIN_DATA_RAW" \
  --micro_batch_size "${MICRO_BATCH_SIZE:-4}" \
  --warmup_tokens "${WARMUP_TOKENS:-100000000}" \
  --nnodes "${NODES:-1}" \
  --devices_per_node "${DEVICES_PER_NODE:-1}" \
  "$@"
