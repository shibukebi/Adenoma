#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/config/route_c_patho_r1.env}"

# shellcheck disable=SC1090
source "${CONFIG_PATH}"

"${CLAM_PYTHON}" "${PROJECT_ROOT}/scripts/build_route_c_feature_input.py" \
  --manifest-csv "${ROUTE_C_OUTPUT_ROOT}/manifest_route_c.csv" \
  --output-csv "${ROUTE_C_FEATURE_INPUT_CSV}"

exec "${CLAM_PYTHON}" "${PROJECT_ROOT}/scripts/extract_features_route_a.py" \
  --data_h5_dir "${ROUTE_C_COORDS_PATCH_DIR}" \
  --data_slide_dir "${SOURCE_DIR}" \
  --csv_path "${ROUTE_C_FEATURE_INPUT_CSV}" \
  --feat_dir "${ROUTE_C_FEATURE_DIR}" \
  --batch_size 64 \
  --slide_ext .svs \
  --target_patch_size 224 \
  --encoder-init pretrained \
  --weights-path /data15/data15_5/yuexin2/adenoma/models/resnet50_tv_in1k_model.safetensors
