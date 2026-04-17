#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHONPATH="${PROJECT_ROOT}/src" \
python3 -m adenoma_agent.cli run-case \
  --case-id 138189_751666001 \
  --trace-mode heuristic \
  --output-root "${PROJECT_ROOT}/artifacts/runs/smoke"
