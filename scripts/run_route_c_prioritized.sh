#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/config/route_c_patho_r1.env}"

# shellcheck disable=SC1090
source "${CONFIG_PATH}"

priority_manifest="${PROJECT_ROOT}/data/route_c_priority_manifest.csv"
log_file="${PROJECT_ROOT}/logs/route_c_prioritized.log"

"${PATHO_R1_PYTHON}" "${PROJECT_ROOT}/scripts/build_route_c_priority_manifest.py" \
  --manifest-csv "${ROUTE_C_MANIFEST_CSV}" \
  --label-csv "${ROUTE_C_LABEL_CSV}" \
  --fold-dir "${ROUTE_C_RAW_FOLD_DIR}" \
  --output-csv "${priority_manifest}"

printf 'Priority manifest: %s\n' "${priority_manifest}"
printf 'Log file: %s\n' "${log_file}"

patho_r1_batch_command=(
  "${PATHO_R1_PYTHON}" "${PROJECT_ROOT}/scripts/run_route_c_batch.py"
  --manifest-csv "${priority_manifest}"
  --output-root "${ROUTE_C_OUTPUT_ROOT}"
  --coords-root "${ROUTE_C_COORDS_ROOT}"
  --mode patho-r1
  --fallback-mode "${ROUTE_C_FALLBACK_MODE}"
  --model-id "${ROUTE_C_MODEL_ID}"
  --thumbnail-max-size "${ROUTE_C_THUMB_MAX_SIZE}"
  --patch-size "${ROUTE_C_PATCH_SIZE}"
  --step-size "${ROUTE_C_STEP_SIZE}"
  --patch-level "${ROUTE_C_PATCH_LEVEL}"
  --max-boxes "${ROUTE_C_MAX_BOXES}"
  --max-new-tokens "${ROUTE_C_MAX_NEW_TOKENS}"
  --local-files-only
  --resume
)

if [[ -n "${ROUTE_C_CUDA_VISIBLE_DEVICES:-}" ]]; then
  exec env CUDA_VISIBLE_DEVICES="${ROUTE_C_CUDA_VISIBLE_DEVICES}" "${patho_r1_batch_command[@]}"
else
  exec "${patho_r1_batch_command[@]}"
fi
