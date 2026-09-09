#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/config/route_lowmag_5x.env}"

# shellcheck disable=SC1090
source "${CONFIG_PATH}"

SEGMENTATION_REUSE_DIR="${SEGMENTATION_REUSE_DIR:-${PROJECT_ROOT}/runs/adenoma_yx_preprocess_1x/segmentations}"
PREPROCESS_PRESET="${PREPROCESS_PRESET:-bwh_biopsy.csv}"
PATCH_SAVE_DIR="${RUN_ROOT}/adenoma_yx_preprocess_5x/patches"
PREPROCESS_SAVE_DIR="${RUN_ROOT}/adenoma_yx_preprocess_5x"
PROCESS_LIST_PATH="${PREPROCESS_SAVE_DIR}/process_list_recover_from_1x_segmentation.csv"
PATCH_IDS_PATH="${PREPROCESS_SAVE_DIR}/recover_patch_ids.txt"
FEATURE_CSV_PATH="${DATA_DIR}/adenoma_yx_feature_input_5x_recovered_missing.csv"
COMPLETE_FEATURE_IDS_PATH="${WORK_DIR}/5x_complete_feature_ids.txt"
MISSING_FEATURE_IDS_PATH="${WORK_DIR}/5x_missing_feature_ids.txt"

mkdir -p "${PREPROCESS_SAVE_DIR}" "${WORK_DIR}"

find "${SEGMENTATION_REUSE_DIR}" -maxdepth 1 -type f -name '*.pkl' \
  | sed 's#.*/##' | sed 's/\.pkl$//' | sort > "${PATCH_IDS_PATH}"

comm -23 "${PATCH_IDS_PATH}" <(
  find "${PATCH_SAVE_DIR}" -maxdepth 1 -type f -name '*.h5' \
    | sed 's#.*/##' | sed 's/\.h5$//' | sort
) > "${WORK_DIR}/5x_missing_patch_ids.txt"

{
  printf 'slide_id\n'
  sed 's/$/.svs/' "${WORK_DIR}/5x_missing_patch_ids.txt"
} > "${PROCESS_LIST_PATH}"

missing_patch_count="$(wc -l < "${WORK_DIR}/5x_missing_patch_ids.txt")"
printf 'missing_5x_patch_slides: %s\n' "${missing_patch_count}"

if [[ "${missing_patch_count}" -gt 0 ]]; then
  cd "${CLAM_ROOT}"
  "${CLAM_PYTHON}" "${CLAM_ROOT}/create_patches_fp.py" \
    --source "${SOURCE_DIR}" \
    --save_dir "${PREPROCESS_SAVE_DIR}" \
    --patch_size "${PATCH_SIZE}" \
    --step_size "${STEP_SIZE}" \
    --patch_level "${PATCH_LEVEL}" \
    --preset "${PREPROCESS_PRESET}" \
    --seg \
    --patch \
    --save_segmentation_pkl \
    --segmentation_mask_dir "${SEGMENTATION_REUSE_DIR}" \
    --process_list "${PROCESS_LIST_PATH}"
fi

find "${PATCH_SAVE_DIR}" -maxdepth 1 -type f -name '*.h5' \
  | sed 's#.*/##' | sed 's/\.h5$//' | sort > "${WORK_DIR}/5x_patch_ids_after_recovery.txt"

comm -12 <(
  find "${LOWMAG_FEATURE_DIR}/pt_files" -maxdepth 1 -type f -name '*.pt' \
    | sed 's#.*/##' | sed 's/\.pt$//' | sort
) <(
  find "${LOWMAG_FEATURE_DIR}/h5_files" -maxdepth 1 -type f -name '*.h5' \
    | sed 's#.*/##' | sed 's/\.h5$//' | sort
) > "${COMPLETE_FEATURE_IDS_PATH}"

comm -23 "${WORK_DIR}/5x_patch_ids_after_recovery.txt" "${COMPLETE_FEATURE_IDS_PATH}" > "${MISSING_FEATURE_IDS_PATH}"

{
  printf 'slide_id\n'
  sed 's/$/.svs/' "${MISSING_FEATURE_IDS_PATH}"
} > "${FEATURE_CSV_PATH}"

missing_feature_count="$(wc -l < "${MISSING_FEATURE_IDS_PATH}")"
printf 'missing_5x_feature_slides: %s\n' "${missing_feature_count}"

if [[ "${missing_feature_count}" -gt 0 ]]; then
  "${PROJECT_ROOT}/scripts/run_extract_features_5x.sh" \
    --data-h5-dir "${PATCH_SAVE_DIR}" \
    --csv-path "${FEATURE_CSV_PATH}" \
    --feat-dir "${LOWMAG_FEATURE_DIR}" \
    --no-auto-skip
fi
