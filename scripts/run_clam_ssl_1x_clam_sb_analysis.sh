#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/config/clam_ssl_1x_clam_sb.env}"

# shellcheck disable=SC1090
source "${CONFIG_PATH}"

default_results_dir="${RESULTS_ROOT}/fold-5"
results_dir="${1:-${default_results_dir}}"
analysis_dir="${results_dir}/analysis"
selection_csv="${analysis_dir}/heatmap_selection.csv"
selection_summary_json="${analysis_dir}/heatmap_selection.summary.json"

if [[ "${results_dir}" == "${default_results_dir}" ]]; then
  heatmap_dir="${HEATMAP_OUTPUT_DIR}"
  report_path="${REPORT_PATH}"
else
  heatmap_dir="${results_dir}/heatmaps"
  report_path="${analysis_dir}/clam_ssl_report.md"
fi

mkdir -p "${analysis_dir}" "${heatmap_dir}"

"${CLAM_PYTHON}" "${PROJECT_ROOT}/scripts/select_error_heatmap_samples.py" \
  --predictions-csv "${results_dir}/predictions.csv" \
  --output-csv "${selection_csv}" \
  --summary-json "${selection_summary_json}"

"${CLAM_PYTHON}" "${PROJECT_ROOT}/scripts/generate_clam_heatmaps.py" \
  --selection-csv "${selection_csv}" \
  --ready-csv "${CLAM_READY_CSV}" \
  --slide-dir "${SOURCE_DIR}" \
  --checkpoint-path "${results_dir}/s_5_checkpoint.pt" \
  --output-dir "${heatmap_dir}" \
  --config-template "${HEATMAP_TEMPLATE}" \
  --clam-python "${CLAM_PYTHON}" \
  --encoder-weights-path "${ENCODER_WEIGHTS_PATH}" \
  --exp-code "${HEATMAP_EXP_CODE}" \
  --patch-size 640 \
  --patch-level 2

"${CLAM_PYTHON}" "${PROJECT_ROOT}/scripts/build_clam_ssl_report.py" \
  --results-dir "${results_dir}" \
  --analysis-dir "${analysis_dir}" \
  --selection-csv "${selection_csv}" \
  --heatmap-dir "${heatmap_dir}" \
  --output-path "${report_path}"
