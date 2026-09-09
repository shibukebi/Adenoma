#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/config/adenoma_yx_preprocess.env}"

# shellcheck disable=SC1090
source "${CONFIG_PATH}"

dataset_tag="${DATASET_TAG:-adenoma_yx}"
base_run_name="${PREPROCESS_RUN_NAME:-${dataset_tag}_preprocess}"

found=0
for pid_file in "${LOG_DIR}/${base_run_name}"_shard_*.pid; do
    [[ -e "${pid_file}" ]] || continue
    found=1
    shard_name="$(basename "${pid_file}" .pid)"
    pid="$(cat "${pid_file}")"
    printf '\n[%s]\n' "${shard_name}"
    printf 'PID file: %s\n' "${pid_file}"
    printf 'PID: %s\n' "${pid}"
    if ps -p "${pid}" > /dev/null 2>&1; then
        ps -p "${pid}" -o pid=,etime=,pcpu=,pmem=,cmd=
    else
        printf 'Process is not running.\n'
    fi

    latest_log="$(ls -1t "${LOG_DIR}/${shard_name}"_*.log 2>/dev/null | head -n 1 || true)"
    if [[ -n "${latest_log}" ]]; then
        printf 'Latest log: %s\n' "${latest_log}"
        tail -n 5 "${latest_log}"
    fi
done

if [[ "${found}" == "0" ]]; then
    printf 'No shard pid files found for prefix: %s_shard_*\n' "${base_run_name}"
fi
