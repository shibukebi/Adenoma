#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/config/adenoma_yx_preprocess.env}"

# shellcheck disable=SC1090
source "${CONFIG_PATH}"

dataset_tag="${DATASET_TAG:-adenoma_yx}"
manifest_prefix="${MANIFEST_PREFIX:-${dataset_tag}}"
slide_ext="${SLIDE_EXT:-.svs}"

mkdir -p "${DATA_DIR}"

manifest_csv="${DATA_DIR}/${manifest_prefix}_manifest.csv"
feature_csv="${DATA_DIR}/${manifest_prefix}_feature_input.csv"
tmp_file="$(mktemp)"

find "${SOURCE_DIR}" -maxdepth 1 -type f -name "*${slide_ext}" | sort > "${tmp_file}"

{
    printf 'slide_id,slide_filename,slide_path\n'
    while IFS= read -r slide_path; do
        slide_filename="$(basename "${slide_path}")"
        slide_id="${slide_filename%"${slide_ext}"}"
        printf '%s,%s,%s\n' "${slide_id}" "${slide_filename}" "${slide_path}"
    done < "${tmp_file}"
} > "${manifest_csv}"

{
    printf 'slide_id\n'
    while IFS= read -r slide_path; do
        slide_filename="$(basename "${slide_path}")"
        slide_id="${slide_filename%"${slide_ext}"}"
        printf '%s\n' "${slide_id}"
    done < "${tmp_file}"
} > "${feature_csv}"

rm -f "${tmp_file}"

printf 'Wrote %s\n' "${manifest_csv}"
printf 'Wrote %s\n' "${feature_csv}"
