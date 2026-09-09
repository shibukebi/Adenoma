#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

gpu_ids="${GPU_IDS:-0 5 6 7}"
interval_seconds="${INTERVAL_SECONDS:-600}"

config_paths=(
  "/data15/data15_5/yuexin2/adenoma/config/adenoma_hp_uni_10x.env"
  "/data15/data15_5/yuexin2/adenoma/config/adenoma_hp_uni_5x.env"
  "/data15/data15_5/yuexin2/adenoma/config/adenoma_hp_uni_2p5x.env"
  "/data15/data15_5/yuexin2/adenoma/config/adenoma_hp_uni_1x.env"
)

while true; do
  active_feature_count="$(
    ps -ef | grep -F 'extract_features_route_a.py' | grep -v grep | wc -l || true
  )"
  if [[ "${active_feature_count}" -gt 0 ]]; then
    echo "global_feature_extractors_active=${active_feature_count}; sleeping ${interval_seconds}s"
    sleep "${interval_seconds}"
    continue
  fi

  launched=0
  for config_path in "${config_paths[@]}"; do
    # shellcheck disable=SC1090
    source "${config_path}"

    pending_csv="${PENDING_FEATURE_INPUT_CSV:-${DATA_DIR}/${DATASET_TAG:-adenoma_yx}_feature_input_pending.csv}"
    "${CLAM_PYTHON}" "${PROJECT_ROOT}/scripts/build_pending_feature_input.py" \
      --patch-dir "${FULL_PATCH_H5_DIR}" \
      --pt-dir "${FULL_FEATURE_DIR}/pt_files" \
      --output-csv "${pending_csv}"

    pending_count="$("${CLAM_PYTHON}" -c "import csv; rows=list(csv.reader(open('${pending_csv}', newline='', encoding='utf-8'))); print(max(0, len(rows)-1))")"
    patch_count="$(find "${FULL_PATCH_H5_DIR}" -maxdepth 1 -type f -name '*.h5' 2>/dev/null | wc -l)"
    done_pt_count="$(find "${FULL_FEATURE_DIR}/pt_files" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | wc -l)"
    echo "config=$(basename "${config_path}") patch_count=${patch_count} done_pt_count=${done_pt_count} pending_count=${pending_count}"

    if [[ "${pending_count}" -gt 0 ]]; then
      GPU_IDS="${gpu_ids}" CONFIG_PATH="${config_path}" bash "${PROJECT_ROOT}/scripts/start_feature_shards.sh"
      launched=1
      break
    fi
  done

  if [[ "${launched}" == "0" ]]; then
    echo "No pending low-mag feature jobs right now."
  fi

  sleep "${interval_seconds}"
done
