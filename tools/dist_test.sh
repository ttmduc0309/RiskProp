#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:?Usage: tools/dist_test.sh CONFIG CHECKPOINT GPUS [TEST_ARGS...]}
CHECKPOINT=${2:?Usage: tools/dist_test.sh CONFIG CHECKPOINT GPUS [TEST_ARGS...]}
GPUS=${3:?Usage: tools/dist_test.sh CONFIG CHECKPOINT GPUS [TEST_ARGS...]}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29501}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="${SCRIPT_DIR}/..:${PYTHONPATH:-}"

run_test() {
  local config_path=$1
  local port=$2
  shift 2
  torchrun \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --nproc_per_node="${GPUS}" \
    --master_port="${port}" \
    "${SCRIPT_DIR}/test.py" "${config_path}" "${CHECKPOINT}" \
    --launcher pytorch "$@"
}

# A full Nexar test evaluation has two established inference protocols:
# submission-window scores for Nexar mAP and full-timeline scores for mTTA.
# Run both from this existing entrypoint and keep their artifacts together.
if [[ "$(basename -- "${CONFIG}")" == "riskprop_nexar.py" ]]; then
  BASE_WORK_DIR="work_dirs/riskprop_nexar_full"
  TEST_ARGS=("${@:4}")
  COMMON_ARGS=()
  USER_CFG_OPTIONS=()
  FAR_MAX=0.1
  for ((index = 0; index < ${#TEST_ARGS[@]}; index++)); do
    if [[ "${TEST_ARGS[index]}" == "--work-dir" ]] && ((index + 1 < ${#TEST_ARGS[@]})); then
      BASE_WORK_DIR=${TEST_ARGS[index + 1]}
      ((index += 1))
    elif [[ "${TEST_ARGS[index]}" == "--cfg-options" ]]; then
      ((index += 1))
      while ((index < ${#TEST_ARGS[@]})) && [[ "${TEST_ARGS[index]}" != --* ]]; do
        USER_CFG_OPTIONS+=("${TEST_ARGS[index]}")
        if [[ "${TEST_ARGS[index]}" == test_evaluator.fpr_max=* ]]; then
          FAR_MAX=${TEST_ARGS[index]#test_evaluator.fpr_max=}
        fi
        ((index += 1))
      done
      index=$((index - 1))
    else
      COMMON_ARGS+=("${TEST_ARGS[index]}")
    fi
  done

  MAP_PREDICTIONS="${BASE_WORK_DIR}/test_predictions.csv"
  MTTA_WORK_DIR="${BASE_WORK_DIR}/mtta"
  TEMPORAL_PREDICTIONS="${MTTA_WORK_DIR}/test_temporal_predictions.jsonl"
  mkdir -p "${BASE_WORK_DIR}" "${MTTA_WORK_DIR}"
  MTTA_PORT=$((PORT + 1))

  run_test "${CONFIG}" "${PORT}" "${COMMON_ARGS[@]}" \
    --work-dir "${BASE_WORK_DIR}" \
    --cfg-options "${USER_CFG_OPTIONS[@]}" \
      "test_evaluator.prediction_file=${MAP_PREDICTIONS}"

  MTTA_CONFIG="${SCRIPT_DIR}/../configs/riskprop_nexar_mtta.py"
  run_test "${MTTA_CONFIG}" "${MTTA_PORT}" "${COMMON_ARGS[@]}" \
    --work-dir "${MTTA_WORK_DIR}" \
    --cfg-options \
      "${USER_CFG_OPTIONS[@]}" \
      "test_dataloader.batch_size=1" \
      "test_evaluator.prediction_file=${MTTA_WORK_DIR}/test_predictions.csv" \
      "test_evaluator.temporal_prediction_file=${TEMPORAL_PREDICTIONS}"

  if [[ "${NODE_RANK}" == "0" ]]; then
    python "${SCRIPT_DIR}/analysis_tools/evaluate_nexar_submission.py" \
      "${MAP_PREDICTIONS}" \
      --temporal-predictions "${TEMPORAL_PREDICTIONS}" \
      --far-max "${FAR_MAX}" \
      --output-dir "${BASE_WORK_DIR}"
  fi
else
  exec torchrun \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --nproc_per_node="${GPUS}" \
    --master_port="${PORT}" \
    "${SCRIPT_DIR}/test.py" "${CONFIG}" "${CHECKPOINT}" --launcher pytorch "${@:4}"
fi
