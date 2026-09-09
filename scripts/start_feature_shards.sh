#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/config/route_a_20x.env}"

# shellcheck disable=SC1090
source "${CONFIG_PATH}"

gpu_ids_string="${GPU_IDS:-0 5 6 7}"
read -r -a gpu_ids <<< "${gpu_ids_string}"
num_shards="${#gpu_ids[@]}"

dataset_tag="${DATASET_TAG:-adenoma_yx}"
feature_run_name="${FEATURE_RUN_NAME:-${dataset_tag}_feature_extraction_20x}"
feat_dir="${FULL_FEATURE_DIR}"
pt_dir="${feat_dir}/pt_files"
pending_csv="${PENDING_FEATURE_INPUT_CSV:-${DATA_DIR}/${dataset_tag}_feature_input_pending.csv}"
shard_dir="${FEATURE_SHARD_DIR:-${DATA_DIR}/${dataset_tag}_feature_shards}"
launch_manifest="${LOG_DIR}/${feature_run_name}_shards_$(date +%Y%m%d_%H%M%S).tsv"
runner="${PROJECT_ROOT}/scripts/run_extract_features.sh"
pending_builder="${PROJECT_ROOT}/scripts/build_pending_feature_input.py"
shard_builder="${PROJECT_ROOT}/scripts/build_feature_shards.py"

mkdir -p "${LOG_DIR}" "${feat_dir}"

"${CLAM_PYTHON}" "${pending_builder}" \
  --patch-dir "${FULL_PATCH_H5_DIR}" \
  --pt-dir "${pt_dir}" \
  --output-csv "${pending_csv}"

pending_count="$("${CLAM_PYTHON}" -c "import csv; rows=list(csv.reader(open('${pending_csv}', newline='', encoding='utf-8'))); print(max(0, len(rows)-1))")"
if [[ "${pending_count}" -eq 0 ]]; then
    printf 'No pending slides for feature extraction.\n'
    exit 0
fi

"${CLAM_PYTHON}" "${shard_builder}" \
  --input-csv "${pending_csv}" \
  --patch-dir "${FULL_PATCH_H5_DIR}" \
  --output-dir "${shard_dir}" \
  --num-shards "${num_shards}"

{
    printf 'shard_id\tgpu_id\tpid\tlog_file\tpid_file\tcsv_path\n'
    for shard_idx in $(seq 0 $((num_shards - 1))); do
        gpu_id="${gpu_ids[$shard_idx]}"
        csv_path="${shard_dir}/feature_shard_${shard_idx}.csv"
        shard_run_name="${feature_run_name}_shard_${shard_idx}"
        log_file="${LOG_DIR}/${shard_run_name}_$(date +%Y%m%d_%H%M%S).log"
        pid_file="${LOG_DIR}/${shard_run_name}.pid"

        nohup env \
            CONFIG_PATH="${CONFIG_PATH}" \
            FEATURE_RUN_NAME="${shard_run_name}" \
            CUDA_VISIBLE_DEVICES="${gpu_id}" \
            "${runner}" \
            --csv-path "${csv_path}" \
            --data-h5-dir "${FULL_PATCH_H5_DIR}" \
            --feat-dir "${feat_dir}" \
            --gpu "${gpu_id}" \
            > "${log_file}" 2>&1 < /dev/null &

        pid=$!
        printf '%s\n' "${pid}" > "${pid_file}"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${shard_idx}" "${gpu_id}" "${pid}" "${log_file}" "${pid_file}" "${csv_path}"
    done
} > "${launch_manifest}"

printf 'Started %s feature shard jobs.\n' "${num_shards}"
printf 'Feature dir: %s\n' "${feat_dir}"
printf 'Launch manifest: %s\n' "${launch_manifest}"
