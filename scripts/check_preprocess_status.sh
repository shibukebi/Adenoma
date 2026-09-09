#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/config/adenoma_yx_preprocess.env}"

# shellcheck disable=SC1090
source "${CONFIG_PATH}"

dataset_tag="${DATASET_TAG:-adenoma_yx}"
preprocess_run_name="${PREPROCESS_RUN_NAME:-${dataset_tag}_preprocess}"

pid_file="${LOG_DIR}/${preprocess_run_name}.pid"

if [[ ! -f "${pid_file}" ]]; then
    printf 'No pid file found: %s\n' "${pid_file}"
    exit 0
fi

pid="$(cat "${pid_file}")"
printf 'PID: %s\n' "${pid}"

if ps -p "${pid}" > /dev/null 2>&1; then
    ps -p "${pid}" -o pid=,etime=,pcpu=,pmem=,cmd=
else
    printf 'Process is not running.\n'
fi

latest_log="$(ls -1t "${LOG_DIR}"/"${preprocess_run_name}"_*.log 2>/dev/null | head -n 1 || true)"
if [[ -n "${latest_log}" ]]; then
    printf '\nLatest log: %s\n' "${latest_log}"
    tail -n 20 "${latest_log}"
fi
