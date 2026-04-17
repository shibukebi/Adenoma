#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHONPATH="${PROJECT_ROOT}/src" \
python3 -m unittest discover -s "${PROJECT_ROOT}/tests" -p "test_*.py" -v
