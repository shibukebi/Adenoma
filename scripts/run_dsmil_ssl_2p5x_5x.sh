#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/config/dsmil_ssl_2p5x_5x.env}"
exec "${PROJECT_ROOT}/scripts/run_dsmil_ssl_pair.sh" "$@"
