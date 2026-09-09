#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

"${PROJECT_ROOT}/scripts/build_fold5_confusion_matrices.py"

CUDA_VISIBLE_DEVICES=0 "${PROJECT_ROOT}/scripts/run_clam_sb_error_heatmaps.sh" \
  20x \
  /data15/data15_5/yuexin2/adenoma/outputs/clam_ssl_vs_others_20x_uni_clam_sb/fold-5_rerun_gpu0_20260424 &

CUDA_VISIBLE_DEVICES=5 "${PROJECT_ROOT}/scripts/run_transmil_error_heatmaps.sh" \
  20x \
  /data15/data15_5/yuexin2/adenoma/outputs/transmil_ssl_vs_others_20x_uni/fold-5 &

CUDA_VISIBLE_DEVICES=6 bash -lc "
  '${PROJECT_ROOT}/scripts/run_clam_sb_error_heatmaps.sh' 1x '/data15/data15_5/yuexin2/adenoma/outputs/clam_ssl_vs_others_1x_uni_clam_sb/fold-5' &&
  '${PROJECT_ROOT}/scripts/run_clam_sb_error_heatmaps.sh' 2p5x '/data15/data15_5/yuexin2/adenoma/outputs/clam_ssl_vs_others_2p5x_uni_clam_sb/fold-5' &&
  '${PROJECT_ROOT}/scripts/run_clam_sb_error_heatmaps.sh' 5x '/data15/data15_5/yuexin2/adenoma/outputs/clam_ssl_vs_others_5x_uni_clam_sb/fold-5'
" &

CUDA_VISIBLE_DEVICES=7 bash -lc "
  '${PROJECT_ROOT}/scripts/run_transmil_error_heatmaps.sh' 1x '/data15/data15_5/yuexin2/adenoma/outputs/transmil_ssl_vs_others_1x_uni/fold-5' &&
  '${PROJECT_ROOT}/scripts/run_transmil_error_heatmaps.sh' 2p5x '/data15/data15_5/yuexin2/adenoma/outputs/transmil_ssl_vs_others_2p5x_uni/fold-5' &&
  '${PROJECT_ROOT}/scripts/run_transmil_error_heatmaps.sh' 5x '/data15/data15_5/yuexin2/adenoma/outputs/transmil_ssl_vs_others_5x_uni/fold-5'
" &

wait
