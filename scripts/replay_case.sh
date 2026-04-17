#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CASE_DIR="${1:?usage: replay_case.sh /path/to/case_dir}"

PYTHONPATH="${PROJECT_ROOT}/src" \
python3 -m adenoma_agent.cli replay-case \
  --case-dir "${CASE_DIR}" \
  --write
