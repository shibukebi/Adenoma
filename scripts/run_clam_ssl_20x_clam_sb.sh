#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/config/clam_ssl_20x_clam_sb.env}"
exec "${PROJECT_ROOT}/scripts/run_clam_ssl_20x.sh" "$@"
