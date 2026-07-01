#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

usage() {
  echo "Usage: $0 <init-config|install|info|uninstall|print> [--env-file PATH]" >&2
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

COMMAND="$1"
shift

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

python3 "${SCRIPT_DIR}/prepare_tokkio_workspace.py" --env-file "${ENV_FILE}" >/dev/null

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

ONE_CLICK_DIR="${TOKKIO_ACE_REPO_DIR}/workflows/tokkio/${TOKKIO_ACE_BRANCH}/scripts/one-click/baremetal"
GENERATED_ENV_FILE="${TOKKIO_CONTROLLER_DIR}/generated/${TOKKIO_ENV_FILE_NAME}"
CONFIG_FILE="${TOKKIO_CONTROLLER_DIR}/${TOKKIO_CONFIG_FILE_NAME}"
EXAMPLE_CONFIG_FILE="${ONE_CLICK_DIR}/config-template-examples/${TOKKIO_PROFILE}/config-template.yml"

require_path() {
  if [[ ! -e "$1" ]]; then
    echo "Required path not found: $1" >&2
    exit 1
  fi
}

require_path "${GENERATED_ENV_FILE}"
require_path "${TOKKIO_ACE_REPO_DIR}"
require_path "${ONE_CLICK_DIR}"

case "${COMMAND}" in
  init-config)
    require_path "${EXAMPLE_CONFIG_FILE}"
    mkdir -p "${TOKKIO_CONTROLLER_DIR}"
    if [[ -e "${CONFIG_FILE}" ]]; then
      echo "Config already exists: ${CONFIG_FILE}"
    else
      cp "${EXAMPLE_CONFIG_FILE}" "${CONFIG_FILE}"
      echo "Created: ${CONFIG_FILE}"
    fi
    ;;
  install|info|uninstall)
    require_path "${CONFIG_FILE}"
    (
      cd "${ONE_CLICK_DIR}"
      set -a
      # shellcheck disable=SC1090
      source "${GENERATED_ENV_FILE}"
      set +a
      ./envbuild.sh "${COMMAND}" --tf-binary terraform --component all --config-file "${CONFIG_FILE}"
    )
    ;;
  print)
    cat <<EOF
Tokkio ACE repo: ${TOKKIO_ACE_REPO_DIR}
One-click dir: ${ONE_CLICK_DIR}
Generated env: ${GENERATED_ENV_FILE}
Config file: ${CONFIG_FILE}
EOF
    ;;
  *)
    usage
    exit 1
    ;;
esac
