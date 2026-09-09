#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/config/route_lowmag_5x.env}"

# shellcheck disable=SC1090
source "${CONFIG_PATH}"

interval_seconds="${INTERVAL_SECONDS:-300}"
available_csv="${PROJECT_ROOT}/data/adenoma_yx_feature_input_5x_available.csv"

while true; do
  "${CLAM_PYTHON}" "${PROJECT_ROOT}/scripts/build_available_feature_input.py" \
    --patch-dir "${LOWMAG_PATCH_H5_DIR}" \
    --output-csv "${available_csv}"

  row_count="$("${CLAM_PYTHON}" -c "import csv; path='${available_csv}'; rows=list(csv.reader(open(path, newline='', encoding='utf-8'))); print(max(0, len(rows)-1))")"

  if [[ "${row_count}" -gt 0 ]]; then
    "${PROJECT_ROOT}/scripts/run_extract_features_5x.sh" --csv-path "${available_csv}"
  else
    echo "No available 5X patch h5 files yet."
  fi

  patch_count=0
  if [[ -d "${LOWMAG_PATCH_H5_DIR}" ]]; then
    patch_count="$(find "${LOWMAG_PATCH_H5_DIR}" -maxdepth 1 -type f -name '*.h5' | wc -l)"
  fi
  pt_count=0
  if [[ -d "${LOWMAG_FEATURE_DIR}/pt_files" ]]; then
    pt_count="$(find "${LOWMAG_FEATURE_DIR}/pt_files" -maxdepth 1 -type f -name '*.pt' | wc -l)"
  fi

  echo "patch_count=${patch_count} pt_count=${pt_count}"

  if [[ "${patch_count}" -gt 0 && "${pt_count}" -ge "${patch_count}" ]]; then
    echo "5X features caught up with current patches."
  fi

  sleep "${interval_seconds}"
done
