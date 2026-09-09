#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/config/adenoma_yx_preprocess.env}"

# shellcheck disable=SC1090
source "${CONFIG_PATH}"

dataset_tag="${DATASET_TAG:-adenoma_yx}"
preprocess_run_name="${PREPROCESS_RUN_NAME:-${dataset_tag}_preprocess}"

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="${LOG_DIR}/${preprocess_run_name}_${timestamp}.log"
pid_file="${LOG_DIR}/${preprocess_run_name}.pid"
runner="${PROJECT_ROOT}/scripts/run_preprocess.sh"

mkdir -p "${LOG_DIR}"

nohup "${runner}" > "${log_file}" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "${pid}" > "${pid_file}"

printf 'Started preprocessing.\n'
printf 'PID: %s\n' "${pid}"
printf 'Log: %s\n' "${log_file}"
printf 'PID file: %s\n' "${pid_file}"
