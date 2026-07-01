#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
DEFAULT_UI_URL="https://10.209.1.12:30111"
DEFAULT_API_URL="http://10.209.1.12:30888"
DEFAULT_GRAFANA_URL="http://10.209.1.12:32300"
DEFAULT_IRODORI_TTS_HEALTH_URL="http://127.0.0.1:8021/healthz"
DEFAULT_IRODORI_TTS_SERVICE="ace-irodori-tts.service"
DEFAULT_RAG_HEALTH_URL="http://127.0.0.1:8081/v1/health?check_dependencies=true"

usage() {
  cat <<EOF
Usage: $0 <start|stop|status|restart|reapply|restart-controller|sync-controller|logs|help> [options] [logs-target]

Commands:
  start                Start host services, restore app workload replicas if saved, then verify Tokkio
  stop                 Scale app workloads in namespace app to zero, then stop host services without uninstalling Tokkio
  status               Show systemd state, app pods, endpoint checks, and controller pod status
  restart              Run stop, wait briefly, then run start
  reapply              Refresh generated env, rerun deploy_tokkio.sh install, then show status
  restart-controller   Recreate ace-controller after source/config/secret updates
  sync-controller      Copy generated ace-controller source/config into the running controller pod
  logs [target]        Show last 200 lines for controller|riva|a2f (default: controller)
  help                 Show this help

Options:
  --env-file PATH      Path to Tokkio env file (default: ${SCRIPT_DIR}/.env)

Verification URLs:
  UI: ${DEFAULT_UI_URL}
  API: ${DEFAULT_API_URL}
  Grafana: ${DEFAULT_GRAFANA_URL}
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

ensure_workspace_artifacts() {
  [[ -f "${ENV_FILE}" ]] || die "env file not found: ${ENV_FILE}"
  info "Refreshing generated Tokkio workspace artifacts from ${ENV_FILE}"
  python3 "${SCRIPT_DIR}/prepare_tokkio_workspace.py" --env-file "${ENV_FILE}" >/dev/null
}

load_env() {
  ensure_workspace_artifacts

  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a

  TOKKIO_WORKSPACE_DIR="${TOKKIO_WORKSPACE_DIR:-${SCRIPT_DIR}/workspace}"
  TOKKIO_CONTROLLER_DIR="${TOKKIO_CONTROLLER_DIR:-${TOKKIO_WORKSPACE_DIR}/controller}"
  TOKKIO_ENV_FILE_NAME="${TOKKIO_ENV_FILE_NAME:-my-config.env}"
  TOKKIO_CONFIG_FILE_NAME="${TOKKIO_CONFIG_FILE_NAME:-ace-app-config.yml}"
  TOKKIO_ACE_BRANCH="${TOKKIO_ACE_BRANCH:-5.0.0-ga}"
  TOKKIO_PROFILE="${TOKKIO_PROFILE:-tokkio-1stream}"
  TOKKIO_ACE_REPO_DIR="${TOKKIO_ACE_REPO_DIR:-${TOKKIO_WORKSPACE_DIR}/NVIDIA-ACE}"

  GENERATED_ENV_FILE="${TOKKIO_CONTROLLER_DIR}/generated/${TOKKIO_ENV_FILE_NAME}"
  CONFIG_FILE="${TOKKIO_CONTROLLER_DIR}/${TOKKIO_CONFIG_FILE_NAME}"
  ONE_CLICK_DIR="${TOKKIO_ACE_REPO_DIR}/workflows/tokkio/${TOKKIO_ACE_BRANCH}/scripts/one-click/baremetal"
  APP_NAMESPACE="${TOKKIO_K8S_NAMESPACE:-app}"
  APP_WORKLOAD_REPLICAS_FILE="${TOKKIO_CONTROLLER_DIR}/generated/app-workload-replicas.tsv"
  APP_WORKLOAD_WAIT_ATTEMPTS="${TOKKIO_APP_WORKLOAD_WAIT_ATTEMPTS:-24}"
  APP_WORKLOAD_WAIT_SECONDS="${TOKKIO_APP_WORKLOAD_WAIT_SECONDS:-5}"
  TOKKIO_CONTROLLER_SOURCE_DIR="${TOKKIO_CONTROLLER_SOURCE_DIR:-${TOKKIO_ACE_REPO_DIR}/workflows/tokkio/${TOKKIO_ACE_BRANCH}/src/llm-rag}"
  TOKKIO_CONTROLLER_POD_PREFIX="${TOKKIO_CONTROLLER_POD_PREFIX:-ace-controller-ace-controller-deployment-}"
  TOKKIO_CONTROLLER_READY_TIMEOUT="${TOKKIO_CONTROLLER_READY_TIMEOUT:-300s}"
  TOKKIO_HELM_RELEASE="${TOKKIO_HELM_RELEASE:-tokkio-app}"
  UI_URL="${TOKKIO_UI_URL:-${DEFAULT_UI_URL}}"
  API_URL="${TOKKIO_API_URL:-${DEFAULT_API_URL}}"
  GRAFANA_URL="${TOKKIO_GRAFANA_URL:-${DEFAULT_GRAFANA_URL}}"
  IRODORI_TTS_ENABLED="${TOKKIO_IRODORI_TTS_ENABLED:-true}"
  IRODORI_TTS_SERVICE="${TOKKIO_IRODORI_TTS_SERVICE:-${DEFAULT_IRODORI_TTS_SERVICE}}"
  IRODORI_TTS_HEALTH_URL="${TOKKIO_IRODORI_TTS_HEALTH_URL:-${DEFAULT_IRODORI_TTS_HEALTH_URL}}"
  IRODORI_TTS_WAIT_ATTEMPTS="${TOKKIO_IRODORI_TTS_WAIT_ATTEMPTS:-60}"
  IRODORI_TTS_WAIT_SECONDS="${TOKKIO_IRODORI_TTS_WAIT_SECONDS:-2}"
  RAG_ENABLED="${TOKKIO_RAG_ENABLED:-false}"
  RAG_HEALTH_URL="${TOKKIO_RAG_HEALTH_URL:-${DEFAULT_RAG_HEALTH_URL}}"
  RAG_WAIT_ATTEMPTS="${TOKKIO_RAG_WAIT_ATTEMPTS:-60}"
  RAG_WAIT_SECONDS="${TOKKIO_RAG_WAIT_SECONDS:-2}"
}

show_urls() {
  info "Verification URLs"
  echo "  UI: ${UI_URL}"
  echo "  API: ${API_URL}"
  echo "  Grafana: ${GRAFANA_URL}"
}

show_service_status() {
  info "systemctl is-active containerd kubelet nginx coturn"
  systemctl is-active containerd kubelet nginx coturn || true
}

is_irodori_tts_enabled() {
  case "${IRODORI_TTS_ENABLED,,}" in
    ""|0|false|no|off)
      return 1
      ;;
    *)
      return 0
      ;;
  esac
}

check_irodori_tts_health() {
  curl -fsS "${IRODORI_TTS_HEALTH_URL}" >/dev/null
}

wait_for_irodori_tts_health() {
  local attempt

  for attempt in $(seq 1 "${IRODORI_TTS_WAIT_ATTEMPTS}"); do
    if check_irodori_tts_health; then
      info "Irodori TTS health check passed: ${IRODORI_TTS_HEALTH_URL}"
      return 0
    fi
    info "Waiting for Irodori TTS health (${attempt}/${IRODORI_TTS_WAIT_ATTEMPTS})"
    sleep "${IRODORI_TTS_WAIT_SECONDS}"
  done

  warn "Timed out waiting for Irodori TTS health: ${IRODORI_TTS_HEALTH_URL}"
  return 1
}

start_irodori_tts_service() {
  if ! is_irodori_tts_enabled; then
    info "Irodori TTS service integration disabled."
    return 0
  fi

  info "Starting Irodori TTS service: ${IRODORI_TTS_SERVICE}"
  sudo systemctl start "${IRODORI_TTS_SERVICE}"
  wait_for_irodori_tts_health
}

stop_irodori_tts_service() {
  if ! is_irodori_tts_enabled; then
    info "Irodori TTS service integration disabled."
    return 0
  fi

  info "Stopping Irodori TTS service: ${IRODORI_TTS_SERVICE}"
  sudo systemctl stop "${IRODORI_TTS_SERVICE}" || true
}

show_irodori_tts_status() {
  if ! is_irodori_tts_enabled; then
    info "Irodori TTS service integration disabled."
    return 0
  fi

  info "systemctl is-active ${IRODORI_TTS_SERVICE}"
  systemctl is-active "${IRODORI_TTS_SERVICE}" || true
  if check_irodori_tts_health; then
    info "Irodori TTS health: ok (${IRODORI_TTS_HEALTH_URL})"
  else
    warn "Irodori TTS health check failed: ${IRODORI_TTS_HEALTH_URL}"
  fi
}

is_rag_enabled() {
  case "${RAG_ENABLED,,}" in
    ""|0|false|no|off)
      return 1
      ;;
    *)
      return 0
      ;;
  esac
}

check_rag_health() {
  curl -fsS "${RAG_HEALTH_URL}" >/dev/null
}

wait_for_rag_health() {
  local attempt

  if ! is_rag_enabled; then
    info "RAG integration disabled."
    return 0
  fi

  for attempt in $(seq 1 "${RAG_WAIT_ATTEMPTS}"); do
    if check_rag_health; then
      info "RAG health check passed: ${RAG_HEALTH_URL}"
      return 0
    fi
    info "Waiting for RAG health (${attempt}/${RAG_WAIT_ATTEMPTS})"
    sleep "${RAG_WAIT_SECONDS}"
  done

  warn "Timed out waiting for RAG health: ${RAG_HEALTH_URL}"
  return 1
}

show_rag_status() {
  if ! is_rag_enabled; then
    info "RAG integration disabled."
    return 0
  fi

  if check_rag_health; then
    info "RAG health: ok (${RAG_HEALTH_URL})"
  else
    warn "RAG health check failed: ${RAG_HEALTH_URL}"
  fi
}

run_kubectl_get_pods() {
  info "kubectl get pods -n ${APP_NAMESPACE}"
  if ! kubectl get pods -n "${APP_NAMESPACE}"; then
    warn "kubectl get pods -n ${APP_NAMESPACE} failed. Check kubelet and containerd first: systemctl is-active containerd kubelet"
    return 1
  fi
}

warn_on_problem_pods() {
  local pod_table
  if ! pod_table="$(kubectl get pods -n "${APP_NAMESPACE}" 2>/dev/null)"; then
    return 0
  fi

  if printf '%s\n' "${pod_table}" | grep -Eq 'Pending|CrashLoopBackOff|ImagePullBackOff'; then
    warn "Detected Pending/CrashLoopBackOff/ImagePullBackOff in namespace ${APP_NAMESPACE}. Inspect the pod list above."
  fi
}

show_controller_status_line() {
  local controller_line
  controller_line="$(kubectl get pods -n "${APP_NAMESPACE}" --no-headers 2>/dev/null | grep "^${TOKKIO_CONTROLLER_POD_PREFIX}" | head -n 1 || true)"
  if [[ -z "${controller_line}" ]]; then
    warn "Failed to inspect ace-controller pod status. Check kubelet and containerd first."
    return 0
  fi

  info "ace-controller: ${controller_line}"
}

check_endpoints() {
  info "python3 infra/tokkio/check_tokkio_endpoints.py --insecure --kubectl ..."
  python3 "${SCRIPT_DIR}/check_tokkio_endpoints.py" \
    --insecure \
    --kubectl \
    --ui-url "${UI_URL}" \
    --api-url "${API_URL}" \
    --grafana-url "${GRAFANA_URL}"
}

start_services() {
  info "Starting services: containerd kubelet nginx coturn"
  info "This start path only brings the existing installation back up. It does not run deploy_tokkio.sh install."
  sudo systemctl start containerd kubelet nginx coturn
}

workload_snapshot_has_positive_replicas() {
  local snapshot_file="$1"

  [[ -f "${snapshot_file}" ]] || return 1
  awk -F '\t' '($3 ~ /^[0-9]+$/ && $3 > 0) { found = 1 } END { exit found ? 0 : 1 }' "${snapshot_file}"
}

rebuild_app_workload_replicas_from_helm_manifest() {
  local tmp_file

  if ! command -v helm >/dev/null 2>&1; then
    warn "Cannot rebuild workload replica snapshot because helm is not installed."
    return 1
  fi

  tmp_file="$(mktemp)"
  if ! helm get manifest "${TOKKIO_HELM_RELEASE}" -n "${APP_NAMESPACE}" | python3 -c '
import sys

try:
    import yaml
except ImportError as exc:
    raise SystemExit(f"PyYAML is required to parse Helm manifest: {exc}")

for doc in yaml.safe_load_all(sys.stdin):
    if not isinstance(doc, dict):
        continue
    kind = doc.get("kind")
    if kind not in {"Deployment", "StatefulSet"}:
        continue
    metadata = doc.get("metadata") or {}
    spec = doc.get("spec") or {}
    name = metadata.get("name")
    replicas = spec.get("replicas", 1)
    if replicas is None:
        replicas = 1
    if name:
        print(f"{kind}\t{name}\t{replicas}")
' > "${tmp_file}"; then
    rm -f "${tmp_file}"
    warn "Failed to rebuild workload replica snapshot from Helm release ${TOKKIO_HELM_RELEASE}."
    return 1
  fi

  if ! workload_snapshot_has_positive_replicas "${tmp_file}"; then
    rm -f "${tmp_file}"
    warn "Helm release ${TOKKIO_HELM_RELEASE} did not yield any positive workload replicas."
    return 1
  fi

  mkdir -p "$(dirname "${APP_WORKLOAD_REPLICAS_FILE}")"
  mv "${tmp_file}" "${APP_WORKLOAD_REPLICAS_FILE}"
  info "Rebuilt app workload replicas from Helm release ${TOKKIO_HELM_RELEASE}: ${APP_WORKLOAD_REPLICAS_FILE}"
}

save_app_workload_replicas() {
  local tmp_file

  tmp_file="$(mktemp)"
  if ! kubectl get deployment,statefulset -n "${APP_NAMESPACE}" \
    -o jsonpath='{range .items[*]}{.kind}{"\t"}{.metadata.name}{"\t"}{.spec.replicas}{"\n"}{end}' > "${tmp_file}"; then
    rm -f "${tmp_file}"
    warn "Failed to snapshot workload replicas in namespace ${APP_NAMESPACE}."
    return 1
  fi

  if ! workload_snapshot_has_positive_replicas "${tmp_file}"; then
    if workload_snapshot_has_positive_replicas "${APP_WORKLOAD_REPLICAS_FILE}"; then
      rm -f "${tmp_file}"
      warn "Current workload snapshot contains no positive replicas; keeping existing snapshot ${APP_WORKLOAD_REPLICAS_FILE}."
      return 0
    fi
    warn "Current workload snapshot contains no positive replicas; saved snapshot may require Helm manifest recovery on start."
  fi

  mkdir -p "$(dirname "${APP_WORKLOAD_REPLICAS_FILE}")"
  mv "${tmp_file}" "${APP_WORKLOAD_REPLICAS_FILE}"
  info "Saved app workload replicas to ${APP_WORKLOAD_REPLICAS_FILE}"
}

wait_for_app_workloads_to_stop() {
  local attempt
  local pod_table
  local pod_count

  for attempt in $(seq 1 "${APP_WORKLOAD_WAIT_ATTEMPTS}"); do
    if ! pod_table="$(kubectl get pods -n "${APP_NAMESPACE}" --no-headers 2>/dev/null)"; then
      warn "Failed to query pods in namespace ${APP_NAMESPACE} while waiting for scale-down."
      return 1
    fi

    pod_count="$(printf '%s\n' "${pod_table}" | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' ')"
    if [[ "${pod_count}" == "0" ]]; then
      info "All app workloads in namespace ${APP_NAMESPACE} are stopped."
      return 0
    fi

    info "Waiting for app workloads to stop (${pod_count} pod(s) remaining, attempt ${attempt}/${APP_WORKLOAD_WAIT_ATTEMPTS})"
    sleep "${APP_WORKLOAD_WAIT_SECONDS}"
  done

  warn "Timed out waiting for pods in namespace ${APP_NAMESPACE} to stop."
  kubectl get pods -n "${APP_NAMESPACE}" || true
  return 1
}

quiesce_app_workloads() {
  if ! save_app_workload_replicas; then
    return 1
  fi

  info "Scaling deployments in namespace ${APP_NAMESPACE} to zero"
  kubectl scale deployment --all -n "${APP_NAMESPACE}" --replicas=0
  info "Scaling statefulsets in namespace ${APP_NAMESPACE} to zero"
  kubectl scale statefulset --all -n "${APP_NAMESPACE}" --replicas=0
  wait_for_app_workloads_to_stop
}

wait_for_kubectl_access() {
  local attempt

  for attempt in $(seq 1 "${APP_WORKLOAD_WAIT_ATTEMPTS}"); do
    if kubectl get namespace "${APP_NAMESPACE}" >/dev/null 2>&1; then
      return 0
    fi

    info "Waiting for Kubernetes API access in namespace ${APP_NAMESPACE} (attempt ${attempt}/${APP_WORKLOAD_WAIT_ATTEMPTS})"
    sleep "${APP_WORKLOAD_WAIT_SECONDS}"
  done

  warn "Timed out waiting for Kubernetes API access in namespace ${APP_NAMESPACE}."
  return 1
}

restore_app_workloads() {
  local kind
  local name
  local replicas
  local resource
  local restored=0

  if [[ ! -f "${APP_WORKLOAD_REPLICAS_FILE}" ]]; then
    info "No saved app workload replica snapshot found at ${APP_WORKLOAD_REPLICAS_FILE}; skipping restore."
    return 0
  fi

  if ! workload_snapshot_has_positive_replicas "${APP_WORKLOAD_REPLICAS_FILE}"; then
    warn "Saved workload snapshot contains no positive replicas: ${APP_WORKLOAD_REPLICAS_FILE}"
    rebuild_app_workload_replicas_from_helm_manifest || return 1
  fi

  if ! wait_for_kubectl_access; then
    warn "Skipping app workload restore because Kubernetes API is not reachable yet."
    return 1
  fi

  while IFS=$'\t' read -r kind name replicas; do
    [[ -n "${kind}" && -n "${name}" && -n "${replicas}" ]] || continue

    case "${kind}" in
      Deployment)
        resource="deployment/${name}"
        ;;
      StatefulSet)
        resource="statefulset/${name}"
        ;;
      *)
        warn "Skipping unsupported workload kind '${kind}' for ${name}"
        continue
        ;;
    esac

    info "Restoring ${resource} to ${replicas} replica(s)"
    kubectl scale "${resource}" -n "${APP_NAMESPACE}" --replicas="${replicas}"
    restored=1
  done < "${APP_WORKLOAD_REPLICAS_FILE}"

  if [[ "${restored}" == "0" ]]; then
    warn "Saved workload snapshot ${APP_WORKLOAD_REPLICAS_FILE} was empty."
  fi
}

stop_services() {
  info "Stopping services: nginx coturn kubelet containerd"
  warn "This stop path assumes the host is dedicated enough for Tokkio that stopping containerd and kubelet is acceptable."
  sudo systemctl stop nginx coturn kubelet containerd
}

show_status() {
  show_service_status
  show_irodori_tts_status
  show_rag_status
  run_kubectl_get_pods || true
  show_controller_status_line
  check_endpoints
  warn_on_problem_pods
  show_urls
}

ensure_controller_runtime_source_files() {
  local source_dir="${TOKKIO_CONTROLLER_SOURCE_DIR}"
  local required_file
  local required_files=(
    "${source_dir}/src/bot.py"
    "${source_dir}/src/config.py"
    "${source_dir}/src/tokkio_irodori_tts.py"
    "${source_dir}/src/tokkio_rag.py"
    "${source_dir}/configs/config.yaml"
  )

  for required_file in "${required_files[@]}"; do
    [[ -f "${required_file}" ]] || die "required controller runtime file not found: ${required_file}"
  done
}

wait_for_controller_ready() {
  local attempt
  local pod_name

  for attempt in $(seq 1 "${APP_WORKLOAD_WAIT_ATTEMPTS}"); do
    pod_name="$(kubectl get pods -n "${APP_NAMESPACE}" --no-headers -o custom-columns='NAME:.metadata.name' 2>/dev/null | awk -v prefix="${TOKKIO_CONTROLLER_POD_PREFIX}" 'index($0, prefix) == 1 { print; exit }')"
    if [[ -n "${pod_name}" ]]; then
      break
    fi
    info "Waiting for ace-controller pod to be recreated (attempt ${attempt}/${APP_WORKLOAD_WAIT_ATTEMPTS})"
    sleep "${APP_WORKLOAD_WAIT_SECONDS}"
  done

  [[ -n "${pod_name}" ]] || die "could not find controller pod with prefix ${TOKKIO_CONTROLLER_POD_PREFIX}"
  info "Waiting for ${pod_name} to become Ready"
  kubectl wait pod/"${pod_name}" -n "${APP_NAMESPACE}" --for=condition=Ready --timeout="${TOKKIO_CONTROLLER_READY_TIMEOUT}"
}

sync_controller_runtime_files() {
  local pod_name
  local source_dir="${TOKKIO_CONTROLLER_SOURCE_DIR}"
  local src_file
  local config_file="${source_dir}/configs/config.yaml"
  local src_files=(
    "bot.py"
    "config.py"
    "tokkio_irodori_tts.py"
    "tokkio_rag.py"
  )

  ensure_controller_runtime_source_files
  wait_for_controller_ready
  pod_name="$(resolve_pod_by_prefix "${TOKKIO_CONTROLLER_POD_PREFIX}")"

  info "Syncing generated ace-controller files from ${source_dir} to ${pod_name}"
  kubectl exec -n "${APP_NAMESPACE}" "${pod_name}" -- mkdir -p /app/src /code/src /code/configs

  for src_file in "${src_files[@]}"; do
    kubectl cp "${source_dir}/src/${src_file}" "${APP_NAMESPACE}/${pod_name}:/app/src/${src_file}"
    kubectl cp "${source_dir}/src/${src_file}" "${APP_NAMESPACE}/${pod_name}:/code/src/${src_file}"
  done
  kubectl cp "${config_file}" "${APP_NAMESPACE}/${pod_name}:/code/configs/config.yaml"
  kubectl exec -n "${APP_NAMESPACE}" "${pod_name}" -- touch /code/src/bot.py
  info "ace-controller source/config sync complete"
}

resolve_pod_by_prefix() {
  local prefix="$1"
  local pod_names
  local pod_name

  if ! pod_names="$(kubectl get pods -n "${APP_NAMESPACE}" --no-headers -o custom-columns='NAME:.metadata.name')"; then
    warn "Failed to list pods in namespace ${APP_NAMESPACE}. Check kubelet and containerd first: systemctl is-active containerd kubelet"
    return 1
  fi

  pod_name="$(printf '%s\n' "${pod_names}" | awk -v prefix="${prefix}" 'index($0, prefix) == 1 { print; exit }')"
  if [[ -n "${pod_name}" ]]; then
    printf '%s\n' "${pod_name}"
    return 0
  fi

  warn "Could not resolve a pod with prefix '${prefix}' in namespace ${APP_NAMESPACE}."
  kubectl get pods -n "${APP_NAMESPACE}" || true
  return 1
}

show_logs() {
  local target="${1:-controller}"
  local prefix
  local pod_name

  case "${target}" in
    controller)
      prefix="${TOKKIO_CONTROLLER_POD_PREFIX}"
      ;;
    riva)
      prefix="riva-speech-"
      ;;
    a2f)
      prefix="a2f-a2f-deployment-"
      ;;
    *)
      die "logs target must be one of: controller, riva, a2f"
      ;;
  esac

  pod_name="$(resolve_pod_by_prefix "${prefix}")"
  info "kubectl logs -n ${APP_NAMESPACE} ${pod_name} --tail=200"
  kubectl logs -n "${APP_NAMESPACE}" "${pod_name}" --tail=200
}

COMMAND=""
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      [[ $# -ge 2 ]] || die "--env-file requires a path"
      ENV_FILE="$2"
      shift 2
      ;;
    start|stop|status|restart|reapply|restart-controller|sync-controller|logs|help)
      if [[ -n "${COMMAND}" ]]; then
        POSITIONAL_ARGS+=("$1")
      else
        COMMAND="$1"
      fi
      shift
      ;;
    -h|--help)
      if [[ -z "${COMMAND}" ]]; then
        COMMAND="help"
      else
        POSITIONAL_ARGS+=("$1")
      fi
      shift
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${COMMAND}" ]]; then
  COMMAND="help"
fi

case "${COMMAND}" in
  help)
    [[ ${#POSITIONAL_ARGS[@]} -eq 0 ]] || die "help does not accept positional arguments"
    usage
    ;;
  start)
    [[ ${#POSITIONAL_ARGS[@]} -eq 0 ]] || die "start does not accept positional arguments"
    load_env
    start_services
    start_irodori_tts_service
    wait_for_rag_health
    restore_app_workloads || true
    sync_controller_runtime_files || true
    show_status
    ;;
  stop)
    [[ ${#POSITIONAL_ARGS[@]} -eq 0 ]] || die "stop does not accept positional arguments"
    load_env
    quiesce_app_workloads
    stop_irodori_tts_service
    stop_services
    show_service_status
    show_irodori_tts_status
    show_urls
    ;;
  status)
    [[ ${#POSITIONAL_ARGS[@]} -eq 0 ]] || die "status does not accept positional arguments"
    load_env
    show_status
    ;;
  restart)
    [[ ${#POSITIONAL_ARGS[@]} -eq 0 ]] || die "restart does not accept positional arguments"
    load_env
    quiesce_app_workloads
    stop_irodori_tts_service
    stop_services
    info "Waiting 3 seconds before restart"
    sleep 3
    start_services
    start_irodori_tts_service
    wait_for_rag_health
    restore_app_workloads || true
    sync_controller_runtime_files || true
    show_status
    ;;
  reapply)
    [[ ${#POSITIONAL_ARGS[@]} -eq 0 ]] || die "reapply does not accept positional arguments"
    load_env
    wait_for_rag_health
    info "Reapplying Tokkio deployment from ${ENV_FILE}"
    "${SCRIPT_DIR}/deploy_tokkio.sh" install --env-file "${ENV_FILE}"
    sync_controller_runtime_files || true
    show_status
    ;;
  restart-controller)
    [[ ${#POSITIONAL_ARGS[@]} -eq 0 ]] || die "restart-controller does not accept positional arguments"
    load_env
    wait_for_rag_health
    info "Restarting ace-controller to pick up refreshed source, config, and secrets"
    kubectl delete pod -n "${APP_NAMESPACE}" "${TOKKIO_CONTROLLER_POD_PREFIX}0"
    sync_controller_runtime_files
    run_kubectl_get_pods || true
    ;;
  sync-controller)
    [[ ${#POSITIONAL_ARGS[@]} -eq 0 ]] || die "sync-controller does not accept positional arguments"
    load_env
    wait_for_rag_health
    sync_controller_runtime_files
    run_kubectl_get_pods || true
    ;;
  logs)
    [[ ${#POSITIONAL_ARGS[@]} -le 1 ]] || die "logs accepts at most one target: controller, riva, or a2f"
    load_env
    show_logs "${POSITIONAL_ARGS[0]:-controller}"
    ;;
  *)
    usage
    exit 1
    ;;
esac
