#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:?Usage: tools/dist_train.sh CONFIG GPUS [TRAIN_ARGS...]}
GPUS=${2:?Usage: tools/dist_train.sh CONFIG GPUS [TRAIN_ARGS...]}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="${SCRIPT_DIR}/..:${PYTHONPATH:-}"

exec torchrun \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --nproc_per_node="${GPUS}" \
  --master_port="${PORT}" \
  "${SCRIPT_DIR}/train.py" "${CONFIG}" --launcher pytorch "${@:3}"
