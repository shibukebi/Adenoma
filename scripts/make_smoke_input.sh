#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/config/adenoma_yx_preprocess.env}"

# shellcheck disable=SC1090
source "${CONFIG_PATH}"

dataset_tag="${DATASET_TAG:-adenoma_yx}"
smoke_dir="${SMOKE_INPUT_DIR:-${WORK_DIR}/${dataset_tag}_smoke_input}"
mkdir -p "${smoke_dir}"

slide_name="$(basename "${SMOKE_SLIDE}")"
ln -sfn "${SMOKE_SLIDE}" "${smoke_dir}/${slide_name}"

printf 'Smoke input ready: %s\n' "${smoke_dir}/${slide_name}"
