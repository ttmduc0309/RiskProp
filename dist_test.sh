#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:?Usage: ./dist_test.sh CONFIG CHECKPOINT [GPUS] [TEST_ARGS...]}
CHECKPOINT=${2:?Usage: ./dist_test.sh CONFIG CHECKPOINT [GPUS] [TEST_ARGS...]}
GPUS=${3:-1}
PORT=${PORT:-29501}
shift $(( $# > 2 ? 3 : $# ))
export PORT

exec "$(dirname "$0")/tools/dist_test.sh" "${CONFIG}" "${CHECKPOINT}" "${GPUS}" "$@"
