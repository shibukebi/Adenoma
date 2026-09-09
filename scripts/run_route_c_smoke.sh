#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/config/route_c_patho_r1.env}"

# shellcheck disable=SC1090
source "${CONFIG_PATH}"

slide_path="${ROUTE_C_SMOKE_SLIDE}"
slide_id="$(basename "${slide_path}")"
slide_id="${slide_id%.svs}"
select_dir="${ROUTE_C_OUTPUT_ROOT}/${slide_id}"
coords_h5="${ROUTE_C_COORDS_PATCH_DIR}/${slide_id}.h5"
patch_export_dir="${ROUTE_C_PATCH_EXPORT_ROOT}/${slide_id}"

mkdir -p "${ROUTE_C_OUTPUT_ROOT}" "${ROUTE_C_COORDS_PATCH_DIR}" "${ROUTE_C_PATCH_EXPORT_ROOT}"

patho_r1_command=(
  "${PATHO_R1_PYTHON}" "${PROJECT_ROOT}/scripts/patho_r1_route_c_select.py"
  --slide-path "${slide_path}"
  --output-dir "${select_dir}"
  --mode patho-r1
  --fallback-mode "${ROUTE_C_FALLBACK_MODE}"
  --model-id "${ROUTE_C_MODEL_ID}"
  --local-files-only
  --max-boxes "${ROUTE_C_MAX_BOXES}"
  --max-new-tokens "${ROUTE_C_MAX_NEW_TOKENS}"
  --thumbnail-max-size "${ROUTE_C_THUMB_MAX_SIZE}"
)

if [[ -n "${ROUTE_C_CUDA_VISIBLE_DEVICES:-}" ]]; then
  env CUDA_VISIBLE_DEVICES="${ROUTE_C_CUDA_VISIBLE_DEVICES}" "${patho_r1_command[@]}"
else
  "${patho_r1_command[@]}"
fi

"${CLAM_PYTHON}" "${PROJECT_ROOT}/scripts/route_c_boxes_to_h5.py" \
  --boxes-json "${select_dir}/${slide_id}_route_c_boxes.json" \
  --output-h5 "${coords_h5}" \
  --patch-size "${ROUTE_C_PATCH_SIZE}" \
  --step-size "${ROUTE_C_STEP_SIZE}" \
  --patch-level "${ROUTE_C_PATCH_LEVEL}"

"${CLAM_PYTHON}" "${PROJECT_ROOT}/scripts/export_patch_samples.py" \
  --slide-path "${slide_path}" \
  --coords-h5 "${coords_h5}" \
  --output-dir "${patch_export_dir}" \
  --max-patches 16

printf 'Route C smoke completed.\n'
printf 'Selection dir: %s\n' "${select_dir}"
printf 'Coords h5: %s\n' "${coords_h5}"
printf 'Patch export dir: %s\n' "${patch_export_dir}"
