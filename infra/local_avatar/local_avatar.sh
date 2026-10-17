#!/usr/bin/env bash
# Start, stop or inspect the Tokkio-free avatar conversation.
# Usage: local_avatar.sh start|stop|status
#
# Managed here: Audio2Face-3D container, Pixel Streaming signalling, the animation
# relay, the Unreal renderer and the conversation server. Not managed (shared with
# Tokkio): the LLM container and the Irodori TTS service; start checks them only.
# Do not run this while Tokkio is up: both use the renderer GPU and the TTS service.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${LOCAL_AVATAR_ROOT:-/data/ACE/ace_kagawa/local_avatar}"
LOGS="${ROOT}/logs"
PY="${ROOT}/.venv/bin/python"
SIGNALLING_DIR="${ROOT}/ps-signalling/SignallingWebServer"
export LOCAL_AVATAR_RENDERER_DIR="${LOCAL_AVATAR_RENDERER_DIR:-${ROOT}/renderer/cc5_tokkio-2026-07-15}"
STREAM_ID="${LOCAL_AVATAR_STREAM_ID:-kagawa-local}"
LLM_URL="${LOCAL_AVATAR_LLM_URL:-http://127.0.0.1:8011/v1/models}"
TTS_URL="${LOCAL_AVATAR_TTS_URL:-http://127.0.0.1:8021/healthz}"
A2F_CONTAINER="${LOCAL_AVATAR_A2F_CONTAINER:-kagawa-local-a2f}"

mkdir -p "${LOGS}"
info() { printf '%s\n' "$*"; }

# pid_alive NAME PATTERN: the PID file exists and its process command line contains PATTERN.
pid_alive() {
  local file="${LOGS}/$1.pid" pid
  [[ -f "${file}" ]] || return 1
  pid="$(cat "${file}")"
  [[ -n "${pid}" ]] && ps -o args= -p "${pid}" 2>/dev/null | grep -q -- "$2"
}

stop_pid() {
  local name="$1" pattern="$2"
  if pid_alive "${name}" "${pattern}"; then
    kill "$(cat "${LOGS}/${name}.pid")"
    info "stopped ${name}"
  fi
  rm -f "${LOGS}/${name}.pid"
}

wait_for() {  # wait_for SECONDS COMMAND...
  local deadline=$((SECONDS + $1)); shift
  until "$@" >/dev/null 2>&1; do
    (( SECONDS < deadline )) || return 1
    sleep 2
  done
}

start() {
  if systemctl is-active --quiet kubelet; then
    info "Tokkio (kubelet) is running; stop it first (infra/tokkio/manage_tokkio.sh stop)"
    exit 1
  fi
  curl -fsS -m 5 "${LLM_URL}" >/dev/null || { info "LLM is not answering at ${LLM_URL} (docker start ace-vllm-llama31-swallow-8b)"; exit 1; }
  curl -fsS -m 5 "${TTS_URL}" >/dev/null || { info "TTS is not answering at ${TTS_URL} (sudo systemctl start ace-irodori-tts)"; exit 1; }

  if ! docker ps --format '{{.Names}}' | grep -qx "${A2F_CONTAINER}"; then
    "${HERE}/run_a2f.sh"
  fi
  a2f_ready() { docker logs "${A2F_CONTAINER}" 2>&1 | grep -q "Starting HTTP Inference server"; }
  wait_for 180 a2f_ready || { info "Audio2Face-3D did not become ready (docker logs ${A2F_CONTAINER})"; exit 1; }
  if ! pid_alive signalling "dist/index.js"; then
    (cd "${SIGNALLING_DIR}" && exec nohup node dist/index.js --serve --streamer_port=18888 --player_port=18080 \
      --peer_options='{"iceServers":[]}' --http_root="${SIGNALLING_DIR}/www" --homepage=player.html \
      >"${LOGS}/signalling.log" 2>&1) &
    echo $! >"${LOGS}/signalling.pid"
  fi
  # The relay and the renderer always restart together: the renderer keeps per-stream
  # audio counters that a fresh relay's time codes would not match.
  stop_pid renderer "cc5_tokkio"
  stop_pid mini_animgraph "mini_animgraph.py"
  sleep 2
  nohup "${PY}" "${HERE}/mini_animgraph.py" >"${LOGS}/mini_animgraph.log" 2>&1 &
  echo $! >"${LOGS}/mini_animgraph.pid"
  wait_for 30 curl -fsS -m 2 http://127.0.0.1:18102/status
  "${HERE}/run_renderer.sh"
  wait_for 120 curl -fsS -m 2 http://127.0.0.1:18021/health || { info "renderer did not start"; exit 1; }
  curl -fsS -m 10 -X POST -d '{}' "http://127.0.0.1:18021/streams/${STREAM_ID}" >/dev/null
  if ! pid_alive server "server.py"; then
    (cd "${HERE}" && exec nohup "${PY}" server.py >"${LOGS}/server.log" 2>&1) &
    echo $! >"${LOGS}/server.pid"
  fi
  wait_for 180 curl -fsSk -m 2 https://127.0.0.1:18443/healthz || { info "server did not start"; exit 1; }
  info "ready: https://$(hostname -I | awk '{print $1}'):18443/"
}

stop() {
  stop_pid server "server.py"
  stop_pid renderer "cc5_tokkio"
  stop_pid mini_animgraph "mini_animgraph.py"
  stop_pid signalling "dist/index.js"
  if docker ps --format '{{.Names}}' | grep -qx "${A2F_CONTAINER}"; then
    docker stop "${A2F_CONTAINER}" >/dev/null && info "stopped ${A2F_CONTAINER}"
  fi
}

status() {
  for item in "server:server.py" "renderer:cc5_tokkio" "mini_animgraph:mini_animgraph.py" "signalling:dist/index.js"; do
    if pid_alive "${item%%:*}" "${item#*:}"; then info "${item%%:*}: running ($(cat "${LOGS}/${item%%:*}.pid"))"
    else info "${item%%:*}: stopped"; fi
  done
  info "a2f: $(docker ps --filter "name=^${A2F_CONTAINER}$" --format '{{.Status}}' | grep . || echo stopped)"
  info "llm: $(curl -fsS -m 3 "${LLM_URL}" >/dev/null 2>&1 && echo up || echo down)"
  info "tts: $(curl -fsS -m 3 "${TTS_URL}" >/dev/null 2>&1 && echo up || echo down)"
  info "relay: $(curl -fsS -m 3 http://127.0.0.1:18102/status 2>/dev/null || echo down)"
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  *) echo "usage: $0 start|stop|status" >&2; exit 2 ;;
esac
