#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-/data15/data15_5/yuexin2/anaconda3/envs/patho-r1/bin/python}"
EVAL_SCRIPT="${EVAL_SCRIPT:-/data15/data15_5/yuexin2/adenoma/scripts/evaluate_mist_11class_5fold.py}"
RESULT_ROOT="${RESULT_ROOT:-/data15/zhengke_usb2/yuexin_data/result/5fold_11class}"
LOG_DIR="${RESULT_ROOT}/launcher_logs/mist_eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

jobs=(
  "2p5x_5x 0 0"
  "2p5x_5x 1 1"
  "2p5x_5x 2 2"
  "2p5x_5x 3 3"
  "2p5x_5x 4 4"
  "5x_10x 0 5"
  "5x_10x 1 6"
  "5x_10x 2 7"
  "5x_10x 3 0"
  "5x_10x 4 1"
)

pids=()
for item in "${jobs[@]}"; do
  read -r combo fold gpu <<<"${item}"
  log="${LOG_DIR}/${combo}_fold${fold}_gpu${gpu}.log"
  printf '[%s] START %s fold-%s gpu-%s log=%s\n' "$(date '+%F %T')" "${combo}" "${fold}" "${gpu}" "${log}" | tee -a "${LOG_DIR}/summary.log"
  "${PY}" "${EVAL_SCRIPT}" --combo "${combo}" --fold "${fold}" --gpu-index "${gpu}" --skip-existing >"${log}" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

printf '[%s] DONE status=%s\n' "$(date '+%F %T')" "${status}" | tee -a "${LOG_DIR}/summary.log"
exit "${status}"
