#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="ace-indextts-tts.service"
REPO_DIR="/data/ACE/en-tts/repos/IndexTTS"
MODEL_DIR="/data/ACE/en-tts/models/IndexTTS-2.5"
REFERENCE_AUDIO="/data/ACE/en-tts/reference/kagawa_en_sentence1.wav"

[[ -x "${REPO_DIR}/.venv/bin/python" ]] || { echo "[ERROR] missing ${REPO_DIR}/.venv" >&2; exit 1; }
[[ -f "${MODEL_DIR}/config.yaml" ]] || { echo "[ERROR] missing IndexTTS model" >&2; exit 1; }
[[ -f "${MODEL_DIR}/gpt.pth" ]] || { echo "[ERROR] missing IndexTTS GPT weights" >&2; exit 1; }
[[ -f "${REFERENCE_AUDIO}" ]] || { echo "[ERROR] missing reference audio" >&2; exit 1; }
[[ -f "${SCRIPT_DIR}/voice/symposium_opening_script.md" ]] || { echo "[ERROR] missing prewarm text" >&2; exit 1; }

sudo install -m 0644 "${SCRIPT_DIR}/${SERVICE_NAME}" "/etc/systemd/system/${SERVICE_NAME}"
sudo systemctl daemon-reload
# Not enabled at boot: the English voice holds GPU1 even in Japanese mode.
sudo systemctl start "${SERVICE_NAME}"
echo "[INFO] Installed and started ${SERVICE_NAME}"
