#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/config/adenoma_yx_preprocess.env}"

# shellcheck disable=SC1090
source "${CONFIG_PATH}"

dataset_tag="${DATASET_TAG:-adenoma_yx}"
manifest_prefix="${MANIFEST_PREFIX:-${dataset_tag}}"
base_run_name="${PREPROCESS_RUN_NAME:-${dataset_tag}_preprocess}"
shared_save_dir="${RUN_ROOT}/${base_run_name}"
shard_dir="${DATA_DIR}/${dataset_tag}_preprocess_shards"
manifest_csv="${DATA_DIR}/${manifest_prefix}_manifest.csv"
runner="${PROJECT_ROOT}/scripts/run_preprocess.sh"
builder="${PROJECT_ROOT}/scripts/build_preprocess_shards.py"
num_shards=8
timestamp="$(date +%Y%m%d_%H%M%S)"
launch_manifest="${LOG_DIR}/${base_run_name}_shards_${timestamp}.tsv"
shard_ids_string="${SHARD_IDS:-0 1 2 3 4 5 6 7}"

mkdir -p "${LOG_DIR}" "${shared_save_dir}"

CONFIG_PATH="${CONFIG_PATH}" "${PROJECT_ROOT}/scripts/build_slide_manifest.sh"

"${CLAM_PYTHON}" "${builder}" \
  --manifest-csv "${manifest_csv}" \
  --output-dir "${shard_dir}" \
  --num-shards "${num_shards}"

{
    printf 'shard_id\tgpu_id\tpid\tlog_file\tpid_file\tprocess_list\n'
    launched_count=0
    for shard_idx in ${shard_ids_string}; do
        gpu_id="${shard_idx}"
        process_list="${shard_dir}/process_list_shard_${shard_idx}.csv"
        shard_run_name="${base_run_name}_shard_${shard_idx}"
        log_file="${LOG_DIR}/${shard_run_name}_${timestamp}.log"
        pid_file="${LOG_DIR}/${shard_run_name}.pid"

        nohup env \
            CONFIG_PATH="${CONFIG_PATH}" \
            PREPROCESS_RUN_NAME="${shard_run_name}" \
            CUDA_VISIBLE_DEVICES="${gpu_id}" \
            "${runner}" \
            --save-dir "${shared_save_dir}" \
            --process-list "${process_list}" \
            > "${log_file}" 2>&1 < /dev/null &

        pid=$!
        printf '%s\n' "${pid}" > "${pid_file}"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${shard_idx}" "${gpu_id}" "${pid}" "${log_file}" "${pid_file}" "${process_list}"
        launched_count=$((launched_count + 1))
    done
} > "${launch_manifest}"

printf 'Started %s shard jobs.\n' "${launched_count}"
printf 'Shared output dir: %s\n' "${shared_save_dir}"
printf 'Launch manifest: %s\n' "${launch_manifest}"
