#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/config/clam_ssl_2p5x_clam_sb.env}"
exec "${PROJECT_ROOT}/scripts/run_clam_ssl_2p5x.sh" \
  --smoke \
  --allow-small-splits \
  --results-dir /data15/data15_5/yuexin2/adenoma/outputs/clam_ssl_vs_others_2p5x_clam_sb_smoke/fold-0
