#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/predict_anomaly_snippet.py}
GPUS=${2:-1}
PORT=${PORT:-29500}
shift $(( $# > 1 ? 2 : $# ))
export PORT

exec "$(dirname "$0")/tools/dist_train.sh" "${CONFIG}" "${GPUS}" "$@"
