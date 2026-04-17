#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LIMIT="${1:-20}"

PYTHONPATH="${PROJECT_ROOT}/src" \
python3 -m adenoma_agent.cli run-batch \
  --trace-mode heuristic \
  --limit "${LIMIT}" \
  --output-root "${PROJECT_ROOT}/artifacts/runs/batch"
