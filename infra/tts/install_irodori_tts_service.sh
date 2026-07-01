#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SERVICE_NAME="${IRODORI_TTS_SERVICE_NAME:-ace-irodori-tts.service}"
SERVICE_SRC="${SCRIPT_DIR}/${SERVICE_NAME}"
SERVICE_DST="/etc/systemd/system/${SERVICE_NAME}"
REFERENCE_SRC="${IRODORI_TTS_REFERENCE_SOURCE:-${ROOT_DIR}/Irodori-TTS/data/kagawa_voice.m4a}"
REFERENCE_WAV="${IRODORI_TTS_REFERENCE_WAV:-/data/ACE/irodori/reference/kagawa_voice_ref_48k_mono.wav}"
DATA_ROOT="${IRODORI_TTS_DATA_ROOT:-/data/ACE/irodori}"

if [[ ! -f "${SERVICE_SRC}" ]]; then
  echo "[ERROR] service unit not found: ${SERVICE_SRC}" >&2
  exit 1
fi

if [[ ! -f "${REFERENCE_SRC}" ]]; then
  echo "[ERROR] reference audio not found: ${REFERENCE_SRC}" >&2
  exit 1
fi

sudo install -d -o "${USER}" -g "${USER}" "${DATA_ROOT}" "$(dirname "${REFERENCE_WAV}")"

if [[ ! -f "${REFERENCE_WAV}" ]]; then
  ffmpeg -hide_banner -loglevel error -y \
    -i "${REFERENCE_SRC}" \
    -ac 1 \
    -ar 48000 \
    "${REFERENCE_WAV}"
fi

sudo install -m 0644 "${SERVICE_SRC}" "${SERVICE_DST}"
sudo systemctl daemon-reload

echo "[INFO] Installed ${SERVICE_DST}"
echo "[INFO] Reference wav: ${REFERENCE_WAV}"
echo "[INFO] Start with: sudo systemctl start ${SERVICE_NAME}"
