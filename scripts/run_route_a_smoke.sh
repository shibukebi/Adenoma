#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/config/route_a_20x.env}"

# shellcheck disable=SC1090
source "${CONFIG_PATH}"

"${PROJECT_ROOT}/scripts/build_smoke_feature_input.sh"
"${PROJECT_ROOT}/scripts/run_extract_features.sh" --smoke --pretrained-init --no-auto-skip

"${CLAM_PYTHON}" "${PROJECT_ROOT}/scripts/clam_pt_to_mag_npy.py" \
  --pt-dir "${SMOKE_FEATURE_DIR}/pt_files" \
  --coords-dir "${SMOKE_PREPROCESS_DIR}/patches" \
  --output-dir "${SMOKE_MAG_NPY_DIR}"

printf 'Route A smoke completed.\n'
printf 'Feature dir: %s\n' "${SMOKE_FEATURE_DIR}"
printf 'MAG npy dir: %s\n' "${SMOKE_MAG_NPY_DIR}"
