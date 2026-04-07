#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$ROOT_DIR/chart-incident-management"
LIVE_JSON="$APP_DIR/dashboard/data/live_incidents.json"
VIDEO_DIR="$APP_DIR/test_videos"

mkdir -p "$(dirname "$LIVE_JSON")"
printf "[]\n" > "$LIVE_JSON"
echo "Cleared $LIVE_JSON"

cd "$APP_DIR"
python dashboard/app.py &
DASH_PID=$!
echo "Dash started (pid=$DASH_PID) at http://127.0.0.1:8050"

cleanup() {
  if kill -0 "$DASH_PID" >/dev/null 2>&1; then
    kill "$DASH_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

python accident_guard_yolo26.py --video_dir "$VIDEO_DIR"
