#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_ROOT="/data15/zhengke_usb2/yuexin_data/training/fold_5/DS-MIL"
FOLD="5"

mkdir -p "${OUTPUT_ROOT}"

launch_job() {
  local name="$1"
  local gpu="$2"
  local config_path="$3"
  local runner="$4"

  local result_dir="${OUTPUT_ROOT}/${name}/fold-${FOLD}"
  local log_path="${OUTPUT_ROOT}/${name}_fold-${FOLD}_gpu${gpu}.log"
  local pid_path="${OUTPUT_ROOT}/${name}_fold-${FOLD}_gpu${gpu}.pid"

  mkdir -p "${OUTPUT_ROOT}/${name}"
  nohup env CONFIG_PATH="${config_path}" "${runner}" \
    --fold "${FOLD}" \
    --gpu "${gpu}" \
    --results-dir "${result_dir}" \
    > "${log_path}" 2>&1 &
  local pid=$!
  printf '%s\n' "${pid}" > "${pid_path}"
  printf '%s\tgpu=%s\tpid=%s\tresults=%s\tlog=%s\n' "${name}" "${gpu}" "${pid}" "${result_dir}" "${log_path}"
}

launch_job \
  "ssl_vs_others_2p5x_5x_uni_dsmil" \
  "0" \
  "${PROJECT_ROOT}/config/dsmil_ssl_2p5x_5x_zhengke.env" \
  "${PROJECT_ROOT}/scripts/run_dsmil_ssl_2p5x_5x.sh"

launch_job \
  "ssl_vs_others_5x_20x_uni_dsmil" \
  "5" \
  "${PROJECT_ROOT}/config/dsmil_ssl_5x_20x_zhengke.env" \
  "${PROJECT_ROOT}/scripts/run_dsmil_ssl_5x_20x.sh"

launch_job \
  "ssl_vs_others_2p5x_20x_uni_dsmil" \
  "6" \
  "${PROJECT_ROOT}/config/dsmil_ssl_2p5x_20x_zhengke.env" \
  "${PROJECT_ROOT}/scripts/run_dsmil_ssl_2p5x_20x.sh"
