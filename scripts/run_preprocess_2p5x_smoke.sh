#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

"${PROJECT_ROOT}/scripts/make_smoke_input.sh"
exec "${PROJECT_ROOT}/scripts/run_preprocess_2p5x.sh" --smoke
