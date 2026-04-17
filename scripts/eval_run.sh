#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_DIR="${1:-${PROJECT_ROOT}/artifacts/runs/smoke}"

PYTHONPATH="${PROJECT_ROOT}/src" \
python3 -m adenoma_agent.cli eval-run \
  --run-dir "${RUN_DIR}"
