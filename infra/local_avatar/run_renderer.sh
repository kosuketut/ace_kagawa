#!/usr/bin/env bash
# Start the packaged Tokkio Unreal renderer without Tokkio (spike).
# It renders off-screen, streams to a local Pixel Streaming signalling server,
# and pulls animation from mini_animgraph.py instead of Animation Graph.
set -euo pipefail

RENDERER_DIR="${LOCAL_AVATAR_RENDERER_DIR:?set LOCAL_AVATAR_RENDERER_DIR to the unpacked renderer package}"
PROJECT="${LOCAL_AVATAR_RENDERER_PROJECT:-cc5_tokkio}"
# Run the binary itself, not the packaged launcher script: the script does
# not exec, so its PID would be a wrapper and stopping it would orphan UE.
BINARY="${PROJECT}/Binaries/Linux/${PROJECT}"
LOG_DIR="${LOCAL_AVATAR_LOG_DIR:-/data/ACE/ace_kagawa/local_avatar/logs}"
SIGNALLING_URL="${LOCAL_AVATAR_SIGNALLING_URL:-ws://127.0.0.1:18888}"
ANIMGRAPH_URL="${LOCAL_AVATAR_ANIMGRAPH_URL:-http://127.0.0.1:18100}"
HTTP_PORT="${LOCAL_AVATAR_RENDERER_HTTP_PORT:-18021}"
RES_X="${LOCAL_AVATAR_RES_X:-1280}"
RES_Y="${LOCAL_AVATAR_RES_Y:-720}"

mkdir -p "${LOG_DIR}"
ace='[/Script/ACECore.ACESettings]'
ini="${ace}:ACEAnimgraphURL=${ANIMGRAPH_URL}"
ini+=",${ace}:NumConnectionAttempts=900,${ace}:TimeBetweenRetrySeconds=1.0,${ace}:ConnectionTimeout=1.0"
ini+=",[/Script/ACEUnrealRendererHttpAPI.Settings]:Port=${HTTP_PORT}"

cd "${RENDERER_DIR}"
# DISPLAY may point at an SSH-forwarded X server; off-screen rendering must not use it.
env -u DISPLAY nohup "./${BINARY}" "${PROJECT}" \
  -RenderOffScreen -Windowed -ForceRes -ResX="${RES_X}" -ResY="${RES_Y}" \
  -Unattended -log \
  -PixelStreamingURL="${SIGNALLING_URL}" -PixelStreamingEncoderCodec=VP8 \
  "-ini:Engine:${ini}" \
  >"${LOG_DIR}/renderer.log" 2>&1 &
echo $! >"${LOG_DIR}/renderer.pid"
echo "renderer pid $(cat "${LOG_DIR}/renderer.pid"), log ${LOG_DIR}/renderer.log"
