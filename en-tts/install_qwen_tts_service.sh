#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="ace-qwen-tts.service"

[[ -x "${SCRIPT_DIR}/.venv/bin/python" ]] || { echo "[ERROR] missing ${SCRIPT_DIR}/.venv" >&2; exit 1; }
[[ -f "${SCRIPT_DIR}/voice/kagawa_EN.MP3" ]] || { echo "[ERROR] missing reference audio" >&2; exit 1; }
[[ -f "/data/ACE/en-tts/models/Qwen3-TTS-12Hz-1.7B-Base/model.safetensors" ]] || { echo "[ERROR] missing Qwen model" >&2; exit 1; }

sudo install -m 0644 "${SCRIPT_DIR}/${SERVICE_NAME}" "/etc/systemd/system/${SERVICE_NAME}"
sudo systemctl daemon-reload
sudo systemctl enable --now "${SERVICE_NAME}"
echo "[INFO] Installed and started ${SERVICE_NAME}"
