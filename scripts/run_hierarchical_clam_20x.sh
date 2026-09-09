#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/config/hierarchical_clam_20x.env}"
exec "${PROJECT_ROOT}/scripts/run_hierarchical_clam_like_20x.sh" "$@"
