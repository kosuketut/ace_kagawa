#!/usr/bin/env bash
# Run Audio2Face-3D 1.3.16 for the Tokkio-free conversation.
# Bidirectional gRPC is published on the host's 127.0.0.1:18200 (container port 52000).
# The image's own deployment and advanced configs are used; only the face style is
# replaced with the values of the kagawa Tokkio deployment (a2f/stylization_config.yaml).
# The first start downloads the model profile from NGC: set NGC_API_KEY for that run
# (see infra/local_avatar/README.md); later starts reuse LOCAL_AVATAR_A2F_CACHE.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${LOCAL_AVATAR_A2F_IMAGE:-nvcr.io/nim/nvidia/audio2face-3d:1.3.16}"
NAME="${LOCAL_AVATAR_A2F_CONTAINER:-kagawa-local-a2f}"
GPU="${LOCAL_AVATAR_A2F_GPU:-0}"
PORT="${LOCAL_AVATAR_A2F_PORT:-18200}"
STYLE="${LOCAL_AVATAR_A2F_STYLE:-${HERE}/a2f/stylization_config.yaml}"
ROOT="${LOCAL_AVATAR_A2F_ROOT:-/data/ACE/ace_kagawa/local_avatar}"

mkdir -p "${ROOT}/a2f-engines" "${ROOT}/a2f-cache" "${ROOT}/a2f-workspace"
env_args=()
if [[ -n "${NGC_API_KEY:-}" ]]; then
  env_args=(-e NGC_API_KEY)
fi

# Not --network host: the NIM also serves HTTP on 8000, which is taken on this host.
# It runs as the host user, so the NIM's cache, workspace and engine paths are mounted writable.
docker run -d --rm --name "${NAME}" --gpus "\"device=${GPU}\"" -p "127.0.0.1:${PORT}:52000" \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e NIM_CACHE_PATH=/nim-cache "${env_args[@]}" \
  -v "${STYLE}:/apps/configs/stylization_config.yaml:ro" \
  -v "${ROOT}/a2f-engines:/tmp/a2x" -v "${ROOT}/a2f-cache:/nim-cache" \
  -v "${ROOT}/a2f-workspace:/opt/nim/workspace" \
  "${IMAGE}"
echo "started ${NAME} on GPU ${GPU}, gRPC 127.0.0.1:${PORT}; logs: docker logs -f ${NAME}"
