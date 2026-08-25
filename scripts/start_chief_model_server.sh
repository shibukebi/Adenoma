#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CHIEF_SERVER_PYTHON="${CHIEF_SERVER_PYTHON:-}"
CHIEF_SERVER_GPU="${CHIEF_SERVER_GPU:-1,2}"
CHIEF_SERVER_HOST="${CHIEF_SERVER_HOST:-127.0.0.1}"
CHIEF_SERVER_PORT="${CHIEF_SERVER_PORT:-8100}"
CHIEF_MODEL_PATH="${CHIEF_MODEL_PATH:-$ROOT_DIR/models/DeepSeek-R1-Distill-Qwen-14B}"
CHIEF_FALLBACK_MODEL_PATH="${CHIEF_FALLBACK_MODEL_PATH:-$ROOT_DIR/models/DeepSeek-R1-Distill-Qwen-14B}"
CHIEF_MAX_NEW_TOKENS="${CHIEF_MAX_NEW_TOKENS:-4096}"
CHIEF_MAX_MODEL_LEN="${CHIEF_MAX_MODEL_LEN:-16384}"
CHIEF_LOG_PATH="${CHIEF_LOG_PATH:-$ROOT_DIR/artifacts/chief_model_server_latest.log}"
CHIEF_ERROR_LOG_PATH="${CHIEF_ERROR_LOG_PATH:-$ROOT_DIR/artifacts/chief_model_server_errors.log}"

if [[ -z "$CHIEF_SERVER_PYTHON" && -f "$ROOT_DIR/configs/runtime.yaml" ]]; then
  CHIEF_SERVER_PYTHON="$(python3 - <<'PY'
import yaml
from pathlib import Path
path = Path("configs/runtime.yaml")
data = yaml.safe_load(path.read_text(encoding="utf-8"))
print(((data or {}).get("paths") or {}).get("chief_python", ""))
PY
)"
fi

if [[ -z "$CHIEF_SERVER_PYTHON" || "$CHIEF_SERVER_PYTHON" == "/path/to/chief-deepseek-serving/bin/python" ]]; then
  echo "CHIEF_SERVER_PYTHON is not set." >&2
  echo "Set it to the dedicated DeepSeek Chief conda environment python, for example:" >&2
  echo "  export CHIEF_SERVER_PYTHON=/path/to/chief-deepseek-serving/bin/python" >&2
  exit 1
fi

if [[ ! -x "$CHIEF_SERVER_PYTHON" ]]; then
  echo "CHIEF_SERVER_PYTHON is not executable: $CHIEF_SERVER_PYTHON" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$CHIEF_SERVER_GPU"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export CHIEF_ERROR_LOG_PATH

mkdir -p "$(dirname "$CHIEF_LOG_PATH")" "$(dirname "$CHIEF_ERROR_LOG_PATH")"
echo "[$(date -Is)] starting Chief server on ${CHIEF_SERVER_HOST}:${CHIEF_SERVER_PORT}" >> "$CHIEF_LOG_PATH"

exec "$CHIEF_SERVER_PYTHON" "$ROOT_DIR/scripts/chief_model_server.py" \
  --host "$CHIEF_SERVER_HOST" \
  --port "$CHIEF_SERVER_PORT" \
  --model-path "$CHIEF_MODEL_PATH" \
  --fallback-model-path "$CHIEF_FALLBACK_MODEL_PATH" \
  --max-new-tokens "$CHIEF_MAX_NEW_TOKENS" \
  --max-model-len "$CHIEF_MAX_MODEL_LEN" \
  --error-log-path "$CHIEF_ERROR_LOG_PATH" \
  >> "$CHIEF_LOG_PATH" 2>&1
