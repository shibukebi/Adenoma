#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

exec "${PROJECT_ROOT}/scripts/run_route_c_clam_ssl_20x.sh" \
  --smoke \
  --allow-small-splits \
  --results-dir /data15/data15_5/yuexin2/adenoma/outputs/clam_ssl_vs_others_20x_route_c_smoke/fold-0
