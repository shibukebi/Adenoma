#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

QWEN_SERVER_PYTHON="${QWEN_SERVER_PYTHON:-/data1/yuexin/.conda/envs/patho-r1/bin/python}"
QWEN_SERVER_GPU="${QWEN_SERVER_GPU:-0}"
QWEN_SERVER_HOST="${QWEN_SERVER_HOST:-127.0.0.1}"
QWEN_SERVER_PORT="${QWEN_SERVER_PORT:-8000}"
QWEN_MODEL_ID="${QWEN_MODEL_ID:-$ROOT_DIR/models/Qwen2.5-VL-7B-Instruct}"
QWEN_ADAPTER_PATH="${QWEN_ADAPTER_PATH:-$ROOT_DIR/models/PathReasoner-R1}"
QWEN_VISION_ENCODER_ID="${QWEN_VISION_ENCODER_ID:-built_in_from_checkpoint}"
QWEN_PROJECTOR_TYPE="${QWEN_PROJECTOR_TYPE:-checkpoint_native}"

if [[ ! -x "$QWEN_SERVER_PYTHON" ]]; then
  echo "QWEN_SERVER_PYTHON is not executable: $QWEN_SERVER_PYTHON" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$QWEN_SERVER_GPU"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$QWEN_SERVER_PYTHON" "$ROOT_DIR/scripts/model_server.py" \
  --host "$QWEN_SERVER_HOST" \
  --port "$QWEN_SERVER_PORT" \
  --model-id "$QWEN_MODEL_ID" \
  --adapter-path "$QWEN_ADAPTER_PATH" \
  --vision-encoder-id "$QWEN_VISION_ENCODER_ID" \
  --projector-type "$QWEN_PROJECTOR_TYPE"
