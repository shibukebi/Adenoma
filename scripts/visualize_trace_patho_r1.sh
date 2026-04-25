#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHONPATH="${PROJECT_ROOT}/src" \
"${PYTHON:-/data1/yuexin/.conda/envs/patho-r1/bin/python}" \
"${SCRIPT_DIR}/visualize_trace_patho_r1.py" "$@"
