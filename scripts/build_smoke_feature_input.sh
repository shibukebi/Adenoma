#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/config/route_a_20x.env}"

# shellcheck disable=SC1090
source "${CONFIG_PATH}"

mkdir -p "${DATA_DIR}"

cat > "${SMOKE_FEATURE_INPUT_CSV}" <<'EOF'
slide_id
138189_751666001
EOF

printf 'Wrote %s\n' "${SMOKE_FEATURE_INPUT_CSV}"
