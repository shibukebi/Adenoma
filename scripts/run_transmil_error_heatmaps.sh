#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

magnification="${1:-}"
results_dir="${2:-}"

if [[ -z "${magnification}" || -z "${results_dir}" ]]; then
  printf 'Usage: %s <1x|2p5x|5x|20x> <results_dir>\n' "$0" >&2
  exit 1
fi

analysis_root="/data15/data15_5/yuexin2/adenoma/outputs/fold5_error_analysis/transmil/${magnification}"
selection_csv="${analysis_root}/selection.csv"
selection_summary_json="${analysis_root}/selection.summary.json"
heatmap_dir="${analysis_root}/heatmaps"

case "${magnification}" in
  1x)
    CONFIG_PATH="${PROJECT_ROOT}/config/transmil_ssl_1x.env"
    patch_dir="/data15/data15_5/yuexin2/adenoma/runs/adenoma_yx_preprocess_1x/patches"
    patch_size=640
    patch_level=2
    ;;
  2p5x)
    CONFIG_PATH="${PROJECT_ROOT}/config/transmil_ssl_2p5x.env"
    patch_dir="/data15/data15_5/yuexin2/adenoma/runs/adenoma_yx_preprocess_2p5x/patches"
    patch_size=256
    patch_level=2
    ;;
  5x)
    CONFIG_PATH="${PROJECT_ROOT}/config/transmil_ssl_5x.env"
    patch_dir="/data15/data15_5/yuexin2/adenoma/runs/adenoma_yx_preprocess_5x/patches"
    patch_size=256
    patch_level=2
    ;;
  20x)
    CONFIG_PATH="${PROJECT_ROOT}/config/transmil_ssl_20x.env"
    patch_dir="/data15/data15_5/yuexin2/adenoma/runs/adenoma_yx_preprocess/patches"
    patch_size=256
    patch_level=0
    ;;
  *)
    printf 'Unknown magnification: %s\n' "${magnification}" >&2
    exit 1
    ;;
esac

# shellcheck disable=SC1090
source "${CONFIG_PATH}"

mkdir -p "${analysis_root}" "${heatmap_dir}"

"${CLAM_PYTHON}" "${PROJECT_ROOT}/scripts/select_error_heatmap_samples.py" \
  --predictions-csv "${results_dir}/predictions.csv" \
  --output-csv "${selection_csv}" \
  --summary-json "${selection_summary_json}" \
  --top-k 3

"${CLAM_PYTHON}" "${PROJECT_ROOT}/scripts/generate_transmil_gradient_heatmaps.py" \
  --selection-csv "${selection_csv}" \
  --ready-csv "${TRANSMIL_READY_CSV}" \
  --slide-dir "${SOURCE_DIR}" \
  --checkpoint-path "${results_dir}/s_5_checkpoint.pt" \
  --feature-h5-dir "${TRANSMIL_FEATURE_DIR}/h5_files" \
  --patch-dir "${patch_dir}" \
  --output-dir "${heatmap_dir}" \
  --embed-dim "${EMBED_DIM}" \
  --model-dim "${TRANSMIL_MODEL_DIM}" \
  --drop-out "${DROPOUT}" \
  --patch-size "${patch_size}" \
  --patch-level "${patch_level}" \
  --exp-code "transmil_${magnification}_fold5_gradient_heatmaps" \
  --top-k 3
