#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

POSITIVES="${1:-12}"
NEGATIVES="${2:-12}"

PYTHONPATH="${PROJECT_ROOT}/src" \
python3 -m adenoma_agent.cli build-pilot \
  --positives "${POSITIVES}" \
  --negatives "${NEGATIVES}"
