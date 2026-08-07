#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/infra/tokkio/.env"
SYNC_CONTROLLER=true

usage() {
  cat <<EOF
Usage: $0 [--env-file PATH] [--no-sync]

Atomically rebuild the local SQLite index, verify corpus freshness, regenerate
Tokkio artifacts, and sync the verified bundle into the running controller.

Options:
  --env-file PATH  Tokkio env file (default: infra/tokkio/.env)
  --no-sync        Stop after rebuilding, verification, and Tokkio generation
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      [[ $# -ge 2 ]] || { echo "[ERROR] --env-file requires a path" >&2; exit 2; }
      ENV_FILE="$2"
      shift 2
      ;;
    --no-sync)
      SYNC_CONTROLLER=false
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "${PROJECT_ROOT}"

python3 infra/rag/build_local_index.py \
  --corpus data/rag/corpus \
  --db data/rag/local/local_rag.sqlite

python3 infra/rag/verify_local_index.py \
  --corpus data/rag/corpus \
  --db data/rag/local/local_rag.sqlite

python3 infra/tokkio/prepare_tokkio_workspace.py --env-file "${ENV_FILE}"

if [[ "${SYNC_CONTROLLER}" == "true" ]]; then
  infra/tokkio/manage_tokkio.sh sync-controller --env-file "${ENV_FILE}"
else
  echo "[INFO] Controller sync skipped by --no-sync"
fi
