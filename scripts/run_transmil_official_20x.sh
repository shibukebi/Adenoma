#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
gpu="${1:-auto}"
if [[ "${gpu}" == "auto" ]]; then
  gpu="$("${PROJECT_ROOT}/scripts/select_idle_gpu.sh")"
fi
"${PROJECT_ROOT}/scripts/check_transmil_env.py"
"${PROJECT_ROOT}/scripts/export_transmil_official_manifest.py" \
  --ready-csv /data15/data15_5/yuexin2/adenoma/data/clam_ssl_others_uni_ready.csv \
  --split-dir /data15/data15_5/yuexin2/adenoma/data/clam_ssl_splits_uni \
  --fold 5 \
  --output-csv /data15/data15_5/yuexin2/adenoma/transmil_official/manifests/adenoma_ssl_20x_fold5.csv
exec /data15/data15_5/yuexin2/anaconda3/envs/clam_latest/bin/python \
  /data15/data15_5/yuexin2/adenoma/transmil_official/train.py \
  --config /data15/data15_5/yuexin2/adenoma/transmil_official/configs/adenoma_ssl_20x_fold5.yaml \
  --gpu "${gpu}"
