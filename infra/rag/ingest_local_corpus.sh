#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CORPUS_DIR="${ACE_RAG_CORPUS_DIR:-${REPO_ROOT}/data/rag/corpus}"
INGESTOR_URL="${ACE_RAG_INGESTOR_URL:-http://127.0.0.1:8082}"
COLLECTION_NAME="${ACE_RAG_COLLECTION_NAME:-ace_kagawa}"
POLL_ATTEMPTS="${ACE_RAG_INGEST_POLL_ATTEMPTS:-120}"
POLL_SECONDS="${ACE_RAG_INGEST_POLL_SECONDS:-2}"
DRY_RUN=false
RESET_COLLECTION=false

usage() {
  cat <<EOF
Usage: $0 [--folder PATH] [--collection NAME] [--ingestor-url URL] [--reset-collection] [--dry-run]

Uploads local files into a NVIDIA RAG Blueprint collection.
Use --reset-collection after deleting local corpus files so stale documents are removed from the vector DB.

Defaults:
  Folder:       ${CORPUS_DIR}
  Collection:   ${COLLECTION_NAME}
  Ingestor URL: ${INGESTOR_URL}
EOF
}

info() {
  echo "[INFO] $*"
}

warn() {
  echo "[WARN] $*" >&2
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

json_field() {
  local field="$1"
  python3 -c '
import json
import sys

field = sys.argv[1]
try:
    payload = json.load(sys.stdin)
except json.JSONDecodeError:
    print("")
    raise SystemExit(0)

value = payload
for part in field.split("."):
    if isinstance(value, dict):
        value = value.get(part)
    else:
        value = None
        break
print("" if value is None else value)
' "${field}"
}

json_collection_array() {
  python3 -c 'import json, sys; print(json.dumps([sys.argv[1]], ensure_ascii=False))' "${COLLECTION_NAME}"
}

delete_collection() {
  local response_file
  local status_code
  local payload

  info "Deleting collection before re-ingest: ${COLLECTION_NAME}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    return 0
  fi

  response_file="$(mktemp)"
  payload="$(json_collection_array)"
  status_code="$(
    curl -sS \
      -o "${response_file}" \
      -w "%{http_code}" \
      -X DELETE \
      -H "accept: application/json" \
      -H "Content-Type: application/json" \
      -d "${payload}" \
      "${INGESTOR_URL%/}/v1/collections"
  )" || {
    local curl_status=$?
    warn "Collection deletion request failed: $(cat "${response_file}")"
    rm -f "${response_file}"
    return "${curl_status}"
  }

  case "${status_code}" in
    200|202|204|404)
      info "Collection delete response (${status_code}): $(cat "${response_file}")"
      ;;
    *)
      warn "Collection delete failed (${status_code}): $(cat "${response_file}")"
      rm -f "${response_file}"
      return 1
      ;;
  esac
  rm -f "${response_file}"
}

upload_file() {
  local file="$1"
  local response
  local task_id
  local attempt
  local status_response
  local status

  info "Uploading ${file}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    return 0
  fi

  response="$(
    curl -fsS \
      -X POST \
      -F "documents=@${file}" \
      -F "data={\"collection_name\":\"${COLLECTION_NAME}\",\"blocking\":false,\"split_options\":{\"chunk_size\":512,\"chunk_overlap\":150},\"custom_metadata\":[],\"generate_summary\":false};type=application/json" \
      "${INGESTOR_URL%/}/v1/documents"
  )"
  task_id="$(printf '%s' "${response}" | json_field task_id)"
  if [[ -z "${task_id}" ]]; then
    task_id="$(printf '%s' "${response}" | json_field id)"
  fi
  [[ -n "${task_id}" ]] || die "Upload did not return task_id for ${file}: ${response}"

  for attempt in $(seq 1 "${POLL_ATTEMPTS}"); do
    status_response="$(curl -fsS "${INGESTOR_URL%/}/v1/status?task_id=${task_id}")"
    status="$(printf '%s' "${status_response}" | json_field state)"
    if [[ -z "${status}" ]]; then
      status="$(printf '%s' "${status_response}" | json_field status)"
    fi
    case "${status}" in
      FINISHED|finished|completed|COMPLETED)
        info "Ingest finished for ${file}"
        return 0
        ;;
      FAILED|failed)
        die "Ingest failed for ${file}: ${status_response}"
        ;;
      *)
        info "Waiting for ${file} ingestion (${attempt}/${POLL_ATTEMPTS}, status=${status:-unknown})"
        sleep "${POLL_SECONDS}"
        ;;
    esac
  done

  die "Timed out waiting for ingestion of ${file}"
}

ensure_collection() {
  local response

  info "Ensuring collection exists: ${COLLECTION_NAME}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    return 0
  fi

  response="$(
    curl -fsS \
      -X POST \
      -H "Content-Type: application/json" \
      -d "{\"collection_name\":\"${COLLECTION_NAME}\",\"embedding_dimension\":2048,\"metadata_schema\":[]}" \
      "${INGESTOR_URL%/}/v1/collection"
  )" || {
    warn "Collection creation failed. If it already exists, ingestion may still continue."
    return 0
  }
  info "Collection response: ${response}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --folder)
      [[ $# -ge 2 ]] || die "--folder requires a path"
      CORPUS_DIR="$2"
      shift 2
      ;;
    --collection)
      [[ $# -ge 2 ]] || die "--collection requires a value"
      COLLECTION_NAME="$2"
      shift 2
      ;;
    --ingestor-url)
      [[ $# -ge 2 ]] || die "--ingestor-url requires a URL"
      INGESTOR_URL="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --reset-collection)
      RESET_COLLECTION=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ -d "${CORPUS_DIR}" ]] || die "Corpus directory not found: ${CORPUS_DIR}"

if [[ "${RESET_COLLECTION}" == "true" ]]; then
  delete_collection
fi
ensure_collection

found=0
while IFS= read -r -d '' file; do
  found=1
  upload_file "${file}"
done < <(find "${CORPUS_DIR}" -type f ! -name '.*' -print0 | sort -z)

if [[ "${found}" == "0" ]]; then
  warn "No files found under ${CORPUS_DIR}"
fi
