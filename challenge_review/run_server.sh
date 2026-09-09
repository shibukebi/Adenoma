#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADENOMA_DIR="$(cd "$APP_DIR/.." && pwd)"
HOST="${CHALLENGE_REVIEW_HOST:-0.0.0.0}"
PORT="${CHALLENGE_REVIEW_PORT:-8765}"

cd "$ADENOMA_DIR"
export PYTHONPATH="$ADENOMA_DIR${PYTHONPATH:+:$PYTHONPATH}"

child_pid=""
stop_server() {
  if [[ -n "$child_pid" ]]; then
    kill "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
  exit 0
}
trap stop_server INT TERM

while true; do
  "$APP_DIR/.venv/bin/uvicorn" challenge_review.main:app --host "$HOST" --port "$PORT" --workers 1 &
  child_pid=$!
  set +e
  wait "$child_pid"
  status=$?
  set -e
  child_pid=""
  if [[ "$status" -eq 0 || "$status" -eq 130 || "$status" -eq 143 ]]; then
    exit 0
  fi
  printf 'WSI review server exited with status %s; restarting in 2 seconds\n' "$status" >&2
  sleep 2
done
