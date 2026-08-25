#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DASHBOARD_DIR="${DASHBOARD_DIR:-$ROOT_DIR/artifacts/real_case_dashboard/real_10sample_harness_v3_20260612}"
DASHBOARD_HOST="${DASHBOARD_HOST:-0.0.0.0}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8000}"
DASHBOARD_SERVER_MODE="${DASHBOARD_SERVER_MODE:-python}"
DASHBOARD_LOG_DIR="${DASHBOARD_LOG_DIR:-$ROOT_DIR/artifacts/.dashboard_server_logs}"

if [[ ! -d "$DASHBOARD_DIR" ]]; then
  echo "DASHBOARD_DIR does not exist: $DASHBOARD_DIR" >&2
  exit 1
fi

mkdir -p "$DASHBOARD_LOG_DIR"

DASHBOARD_NAME="$(basename "$DASHBOARD_DIR")"
SAFE_NAME="$(printf '%s' "$DASHBOARD_NAME" | sed 's/[^A-Za-z0-9._-]/_/g')"
PID_FILE="$DASHBOARD_LOG_DIR/${SAFE_NAME}_${DASHBOARD_PORT}.pid"
LOG_FILE="$DASHBOARD_LOG_DIR/${SAFE_NAME}_${DASHBOARD_PORT}.log"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") start
  $(basename "$0") stop
  $(basename "$0") restart
  $(basename "$0") status
  $(basename "$0") logs
  $(basename "$0") foreground

Environment overrides:
  DASHBOARD_DIR=$DASHBOARD_DIR
  DASHBOARD_HOST=$DASHBOARD_HOST
  DASHBOARD_PORT=$DASHBOARD_PORT
  DASHBOARD_SERVER_MODE=$DASHBOARD_SERVER_MODE   # python | tmux-python | serve | npx-serve
  DASHBOARD_LOG_DIR=$DASHBOARD_LOG_DIR
EOF
}

running_pid() {
  local pid
  if [[ -f "$PID_FILE" ]]; then
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      printf '%s\n' "$pid"
      return 0
    fi
  fi
  if command -v lsof >/dev/null 2>&1; then
    local cwd
    for pid in $(lsof -tiTCP:"$DASHBOARD_PORT" -sTCP:LISTEN -n -P 2>/dev/null || true); do
      cwd="$(readlink "/proc/$pid/cwd" 2>/dev/null || true)"
      if [[ "$cwd" == "$DASHBOARD_DIR" ]]; then
        printf '%s\n' "$pid"
        printf '%s\n' "$pid" >"$PID_FILE"
        return 0
      fi
    done
  fi
  return 1
}

build_command() {
  local quoted_dir quoted_host quoted_port server_code quoted_server_code
  printf -v quoted_dir '%q' "$DASHBOARD_DIR"
  printf -v quoted_host '%q' "$DASHBOARD_HOST"
  printf -v quoted_port '%q' "$DASHBOARD_PORT"
  server_code='import http.server,socketserver,sys; host=sys.argv[1]; port=int(sys.argv[2]); Server=type("ThreadingHTTPServer",(socketserver.ThreadingMixIn,http.server.HTTPServer),{"daemon_threads":True,"allow_reuse_address":True}); httpd=Server((host,port),http.server.SimpleHTTPRequestHandler); print("Serving threaded dashboard on %s:%s"%(host,port),flush=True); httpd.serve_forever()'
  printf -v quoted_server_code '%q' "$server_code"

  case "$DASHBOARD_SERVER_MODE" in
    python)
      printf 'cd %s && exec python3 -u -c %s %s %s' "$quoted_dir" "$quoted_server_code" "$quoted_host" "$quoted_port"
      ;;
    tmux-python)
      printf 'cd %s && exec python3 -u -c %s %s %s' "$quoted_dir" "$quoted_server_code" "$quoted_host" "$quoted_port"
      ;;
    serve)
      if ! command -v serve >/dev/null 2>&1; then
        echo "The 'serve' command is not installed. Install it with: npm install -g serve" >&2
        exit 1
      fi
      printf 'exec serve %s -l tcp://%s:%s' "$quoted_dir" "$quoted_host" "$quoted_port"
      ;;
    npx-serve)
      if ! command -v npx >/dev/null 2>&1; then
        echo "The 'npx' command is not available on this server." >&2
        exit 1
      fi
      printf 'exec npx serve %s -l tcp://%s:%s' "$quoted_dir" "$quoted_host" "$quoted_port"
      ;;
    *)
      echo "Unsupported DASHBOARD_SERVER_MODE: $DASHBOARD_SERVER_MODE" >&2
      echo "Use one of: python, serve, npx-serve" >&2
      exit 1
      ;;
  esac
}

start_server() {
  if pid="$(running_pid)"; then
    echo "Dashboard server is already running."
    echo "PID: $pid"
    echo "URL: http://$DASHBOARD_HOST:$DASHBOARD_PORT/"
    echo "Log: $LOG_FILE"
    return 0
  fi

  local command_string
  command_string="$(build_command)"
  if [[ "$DASHBOARD_SERVER_MODE" == "tmux-python" ]]; then
    if ! command -v tmux >/dev/null 2>&1; then
      echo "The 'tmux' command is not available. Use DASHBOARD_SERVER_MODE=python instead." >&2
      exit 1
    fi
    local session_name="adenoma_dashboard_${SAFE_NAME}_${DASHBOARD_PORT}"
    tmux kill-session -t "$session_name" >/dev/null 2>&1 || true
    tmux new-session -d -s "$session_name" "bash -lc '$command_string'" 2>"$LOG_FILE"
    sleep 1
    local pid
    pid="$(tmux display-message -p -t "$session_name" '#{pane_pid}' 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "$pid" >"$PID_FILE"
    else
      echo "Dashboard tmux session failed to start. Check the log:" >&2
      echo "  $LOG_FILE" >&2
      rm -f "$PID_FILE"
      exit 1
    fi
  elif command -v setsid >/dev/null 2>&1; then
    nohup setsid bash -lc "$command_string" >"$LOG_FILE" 2>&1 < /dev/null &
    local pid="$!"
    echo "$pid" >"$PID_FILE"
  else
    nohup bash -lc "$command_string" >"$LOG_FILE" 2>&1 < /dev/null &
    local pid="$!"
    echo "$pid" >"$PID_FILE"
  fi
  sleep 1

  if kill -0 "$pid" 2>/dev/null; then
    echo "Dashboard server started."
    echo "PID: $pid"
    echo "URL: http://$DASHBOARD_HOST:$DASHBOARD_PORT/"
    echo "Dashboard dir: $DASHBOARD_DIR"
    echo "Mode: $DASHBOARD_SERVER_MODE"
    echo "Log: $LOG_FILE"
  else
    echo "Dashboard server failed to start. Check the log:" >&2
    echo "  $LOG_FILE" >&2
    rm -f "$PID_FILE"
    exit 1
  fi
}

stop_server() {
  if ! pid="$(running_pid)"; then
    echo "Dashboard server is not running."
    rm -f "$PID_FILE"
    return 0
  fi
  kill "$pid"
  sleep 1
  if kill -0 "$pid" 2>/dev/null; then
    echo "PID $pid did not exit after SIGTERM; sending SIGKILL."
    kill -9 "$pid"
  fi
  rm -f "$PID_FILE"
  echo "Dashboard server stopped."
}

status_server() {
  if pid="$(running_pid)"; then
    echo "Dashboard server is running."
    echo "PID: $pid"
    echo "URL: http://$DASHBOARD_HOST:$DASHBOARD_PORT/"
    echo "Dashboard dir: $DASHBOARD_DIR"
    echo "Mode: $DASHBOARD_SERVER_MODE"
    echo "Log: $LOG_FILE"
  else
    echo "Dashboard server is not running."
    echo "Configured URL: http://$DASHBOARD_HOST:$DASHBOARD_PORT/"
    echo "Dashboard dir: $DASHBOARD_DIR"
    echo "Mode: $DASHBOARD_SERVER_MODE"
    echo "Log: $LOG_FILE"
    rm -f "$PID_FILE"
    return 1
  fi
}

foreground_server() {
  local command_string
  command_string="$(build_command)"
  echo "Starting dashboard server in foreground..."
  echo "URL: http://$DASHBOARD_HOST:$DASHBOARD_PORT/"
  echo "Dashboard dir: $DASHBOARD_DIR"
  echo "Mode: $DASHBOARD_SERVER_MODE"
  exec bash -lc "$command_string"
}

tail_logs() {
  if [[ ! -f "$LOG_FILE" ]]; then
    echo "Log file does not exist yet: $LOG_FILE" >&2
    exit 1
  fi
  exec tail -f "$LOG_FILE"
}

ACTION="${1:-start}"

case "$ACTION" in
  start)
    start_server
    ;;
  stop)
    stop_server
    ;;
  restart)
    stop_server || true
    start_server
    ;;
  status)
    status_server
    ;;
  logs)
    tail_logs
    ;;
  foreground)
    foreground_server
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    usage >&2
    exit 1
    ;;
esac
