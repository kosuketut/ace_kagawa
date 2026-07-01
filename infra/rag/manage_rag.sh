#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RAG_ROOT="${ACE_RAG_ROOT:-/data/ACE/rag}"
BLUEPRINT_DIR="${ACE_RAG_BLUEPRINT_DIR:-${RAG_ROOT}/blueprint}"
CORPUS_DIR="${ACE_RAG_CORPUS_DIR:-${REPO_ROOT}/data/rag/corpus}"
RAG_REPO_URL="${ACE_RAG_REPO_URL:-https://github.com/NVIDIA-AI-Blueprints/rag.git}"
RAG_ENV_FILE="${ACE_RAG_ENV_FILE:-${BLUEPRINT_DIR}/deploy/compose/nvdev.env}"
TOKKIO_ENV_FILE="${ACE_RAG_TOKKIO_ENV_FILE:-${REPO_ROOT}/infra/tokkio/.env}"
RAG_HEALTH_URL="${ACE_RAG_HEALTH_URL:-http://127.0.0.1:8081/v1/health?check_dependencies=true}"
RAG_INGESTOR_HEALTH_URL="${ACE_RAG_INGESTOR_HEALTH_URL:-http://127.0.0.1:8082/v1/health?check_dependencies=true}"
RAG_LLM_MODEL="${ACE_RAG_LLM_MODEL:-stockmark/stockmark-2-100b-instruct}"
RAG_LLM_SERVER_URL="${ACE_RAG_LLM_SERVER_URL:-https://integrate.api.nvidia.com/v1}"

usage() {
  cat <<EOF
Usage: $0 <init|start|stop|status|health|logs|help> [logs-target]

Commands:
  init          Create /data/ACE/rag directories and clone NVIDIA RAG Blueprint if needed
  start         Start NVIDIA-hosted Docker RAG services: vector DB, ingestor, RAG server
  stop          Stop RAG server and ingestor without deleting persisted Docker volumes
  status        Show Docker Compose status for RAG services
  health        Check RAG and ingestor health endpoints
  logs [target] Show logs for rag-server|ingestor|vectordb (default: rag-server)
  help          Show this help

Defaults:
  Blueprint: ${BLUEPRINT_DIR}
  Corpus:    ${CORPUS_DIR}
  Env file:  ${RAG_ENV_FILE}
  Tokkio env: ${TOKKIO_ENV_FILE}
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

ensure_dirs() {
  mkdir -p "${RAG_ROOT}" "${CORPUS_DIR}"
}

ensure_blueprint() {
  ensure_dirs
  if [[ -d "${BLUEPRINT_DIR}/.git" ]]; then
    return 0
  fi
  if [[ -e "${BLUEPRINT_DIR}" ]]; then
    die "Blueprint path exists but is not a git checkout: ${BLUEPRINT_DIR}"
  fi
  info "Cloning NVIDIA RAG Blueprint into ${BLUEPRINT_DIR}"
  git clone "${RAG_REPO_URL}" "${BLUEPRINT_DIR}"
}

ensure_docker() {
  if docker info >/dev/null 2>&1; then
    return 0
  fi
  if command -v systemctl >/dev/null 2>&1; then
    info "Docker daemon is not reachable; starting docker.service"
    sudo systemctl start docker
  fi
  docker info >/dev/null 2>&1 || die "Docker daemon is not reachable"
}

compose_env() {
  [[ -f "${RAG_ENV_FILE}" ]] || die "RAG env file not found: ${RAG_ENV_FILE}. Run init after the Blueprint checkout exists."

  if [[ -f "${TOKKIO_ENV_FILE}" ]]; then
    set -a
    set +u
    # shellcheck disable=SC1090
    source "${TOKKIO_ENV_FILE}"
    set -u
    set +a
  fi

  if [[ -z "${NGC_API_KEY:-}" && -n "${TOKKIO_NGC_CLI_API_KEY:-}" ]]; then
    export NGC_API_KEY="${TOKKIO_NGC_CLI_API_KEY}"
  fi
  if [[ -z "${NVIDIA_API_KEY:-}" && -n "${TOKKIO_NVIDIA_API_KEY:-}" ]]; then
    export NVIDIA_API_KEY="${TOKKIO_NVIDIA_API_KEY}"
  fi
  if [[ -z "${NGC_API_KEY:-}" && -n "${NVIDIA_API_KEY:-}" ]]; then
    export NGC_API_KEY="${NVIDIA_API_KEY}"
  fi
  if [[ -z "${NVIDIA_API_KEY:-}" && -n "${NGC_API_KEY:-}" ]]; then
    export NVIDIA_API_KEY="${NGC_API_KEY}"
  fi
  [[ -n "${NGC_API_KEY:-}" || -n "${NVIDIA_API_KEY:-}" ]] || die "Set NGC_API_KEY or NVIDIA_API_KEY before starting RAG"

  set -a
  set +u
  # shellcheck disable=SC1090
  source "${RAG_ENV_FILE}"
  set -u
  set +a

  export APP_LLM_MODELNAME="${RAG_LLM_MODEL}"
  export APP_LLM_SERVERURL="${RAG_LLM_SERVER_URL}"
  export APP_LLM_APIKEY="${APP_LLM_APIKEY:-${NVIDIA_API_KEY:-${NGC_API_KEY:-}}}"
  export COMPONENTS_TO_READY_CHECK="${COMPONENTS_TO_READY_CHECK:-}"
}

compose_run() {
  local compose_file="$1"
  shift
  (
    cd "${BLUEPRINT_DIR}"
    compose_env
    docker compose -f "${compose_file}" "$@"
  )
}

start_rag() {
  ensure_blueprint
  ensure_docker
  info "Starting RAG vector DB"
  compose_run deploy/compose/vectordb.yaml up -d
  info "Starting RAG ingestor"
  compose_run deploy/compose/docker-compose-ingestor-server.yaml up -d
  info "Starting RAG server"
  compose_run deploy/compose/docker-compose-rag-server.yaml up -d
  health_rag
}

stop_rag() {
  [[ -d "${BLUEPRINT_DIR}" ]] || return 0
  ensure_docker
  compose_run deploy/compose/docker-compose-rag-server.yaml down || true
  compose_run deploy/compose/docker-compose-ingestor-server.yaml down || true
}

status_rag() {
  [[ -d "${BLUEPRINT_DIR}" ]] || die "Blueprint checkout not found: ${BLUEPRINT_DIR}"
  ensure_docker
  compose_run deploy/compose/vectordb.yaml ps || true
  compose_run deploy/compose/docker-compose-ingestor-server.yaml ps || true
  compose_run deploy/compose/docker-compose-rag-server.yaml ps || true
}

health_rag() {
  info "RAG health: ${RAG_HEALTH_URL}"
  curl -fsS "${RAG_HEALTH_URL}"
  echo
  info "Ingestor health: ${RAG_INGESTOR_HEALTH_URL}"
  curl -fsS "${RAG_INGESTOR_HEALTH_URL}"
  echo
}

logs_rag() {
  local target="${1:-rag-server}"
  case "${target}" in
    rag-server)
      compose_run deploy/compose/docker-compose-rag-server.yaml logs --tail=200
      ;;
    ingestor)
      compose_run deploy/compose/docker-compose-ingestor-server.yaml logs --tail=200
      ;;
    vectordb)
      compose_run deploy/compose/vectordb.yaml logs --tail=200
      ;;
    *)
      die "logs target must be one of: rag-server, ingestor, vectordb"
      ;;
  esac
}

COMMAND="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${COMMAND}" in
  init)
    ensure_blueprint
    info "RAG directories ready"
    echo "  Blueprint: ${BLUEPRINT_DIR}"
    echo "  Corpus: ${CORPUS_DIR}"
    ;;
  start)
    start_rag
    ;;
  stop)
    stop_rag
    ;;
  status)
    status_rag
    ;;
  health)
    health_rag
    ;;
  logs)
    logs_rag "${1:-rag-server}"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage
    exit 1
    ;;
esac
