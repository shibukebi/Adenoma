#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/config/route_a_20x.env}"

# shellcheck disable=SC1090
source "${CONFIG_PATH}"

interval_seconds="${INTERVAL_SECONDS:-600}"
gpu_ids="${GPU_IDS:-0 5 6 7}"
feat_dir="${FULL_FEATURE_DIR}"
patch_dir="${FULL_PATCH_H5_DIR}"
pending_builder="${PROJECT_ROOT}/scripts/build_pending_feature_input.py"
pending_csv="${PENDING_FEATURE_INPUT_CSV:-${DATA_DIR}/${DATASET_TAG:-adenoma_yx}_feature_input_pending.csv}"
start_script="${PROJECT_ROOT}/scripts/start_feature_shards.sh"

mkdir -p "${feat_dir}" "${LOG_DIR}"

while true; do
  if [[ ! -d "${patch_dir}" ]]; then
    echo "patch_dir_not_ready=${patch_dir}; sleeping ${interval_seconds}s"
    sleep "${interval_seconds}"
    continue
  fi

  active_feature_count="$(
    pgrep -af 'extract_features_route_a.py' | grep -F -- "${feat_dir}" | wc -l || true
  )"

  if [[ "${active_feature_count}" -gt 0 ]]; then
    echo "feature_extractors_active=${active_feature_count}; sleeping ${interval_seconds}s"
    sleep "${interval_seconds}"
    continue
  fi

  "${CLAM_PYTHON}" "${pending_builder}" \
    --patch-dir "${patch_dir}" \
    --pt-dir "${feat_dir}/pt_files" \
    --output-csv "${pending_csv}"

  pending_count="$("${CLAM_PYTHON}" -c "import csv; rows=list(csv.reader(open('${pending_csv}', newline='', encoding='utf-8'))); print(max(0, len(rows)-1))")"
  patch_count="$(find "${patch_dir}" -maxdepth 1 -type f -name '*.h5' | wc -l)"
  done_pt_count="$(find "${feat_dir}/pt_files" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | wc -l)"

  echo "patch_count=${patch_count} done_pt_count=${done_pt_count} pending_count=${pending_count}"

  if [[ "${pending_count}" -gt 0 ]]; then
    GPU_IDS="${gpu_ids}" CONFIG_PATH="${CONFIG_PATH}" bash "${start_script}"
  else
    echo "No pending 20x feature slides right now."
  fi

  sleep "${interval_seconds}"
done
