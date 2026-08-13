#!/usr/bin/env bash
# Install / undeploy only the reindex service (Docling + Qdrant indexer).
#
# Production (needs root):
#   sudo ./deploy/install-reindex.sh
#   sudo ./deploy/install-reindex.sh --enable
#   sudo ./deploy/install-reindex.sh --undeploy
#
# Safe fake-root (no systemd, no user creation, does not touch host /):
#   ./deploy/install-reindex.sh --dest-dir /tmp/itc-reindex
#   ./deploy/install-reindex.sh --dest-dir /tmp/itc-reindex --layout-only
#   ./deploy/install-reindex.sh --dest-dir /tmp/itc-reindex --undeploy
#
# Environment:
#   DESTDIR  — same as --dest-dir (packaging convention)
#   PYTHON   — python interpreter for venv (default: python3)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OPT_DIR="/opt/it-consultant"
ETC_DIR="/etc/it-consultant"
VAR_DIR="/var/lib/it-consultant"
SERVICE_USER="it-consultant"
SERVICE_GROUP="it-consultant"
UNIT="reindex.service"

# Interactive configure asks only these keys; EWS_* stay as dummy defaults
# so common.Settings still constructs when running reindex alone.
REINDEX_ENV_KEYS=(
  OLLAMA_BASE_URL
  EMBEDDING_MODEL
  EMBEDDING_TIMEOUT_SEC
  QDRANT_URL
  QDRANT_COLLECTION
  CHUNK_SIZE
  CHUNK_OVERLAP
  LOG_LEVEL
  WATCH_PATH
  DEBOUNCE_SECONDS
  INDEX_EXTENSIONS
)

DEST_DIR="${DESTDIR:-}"
LAYOUT_ONLY=0
ENABLE=0
CREATE_USER=1
UNDEPLOY=0
CONFIGURE=auto
PYTHON="${PYTHON:-python3}"

usage() {
  cat <<'EOF'
Usage: deploy/install-reindex.sh [options]

Installs reindex only: copies app sources, creates venv, pip install -e '.[reindex]'
(docling, watchdog, qdrant-client), writes .env, installs reindex.service.

Does not install or start mail-gateway.

Options:
  --dest-dir DIR   Install/undeploy under DIR as fake root (sets DESTDIR).
                   Skips systemctl and useradd/userdel. Safe for local/CI tests.
  --layout-only    Create dirs, .env, systemd unit only (no copy/venv/pip).
  --enable         systemctl daemon-reload && enable --now reindex.service
                   (ignored when --dest-dir is set).
  --undeploy       Stop/disable reindex.service, remove app/env/data/unit and
                   service user (production). With --dest-dir only removes the
                   fake tree. Does not remove mail-gateway.service if present.
  --configure      Prompt for reindex-related .env keys (Enter keeps current).
  --no-configure   Do not prompt for .env values (default for non-TTY / CI).
  --no-create-user Do not create service user/group (real install only).
  -h, --help       Show this help.

Paths (always absolute on the target system; prefixed by --dest-dir when set):
  /opt/it-consultant         application + .venv
  /etc/it-consultant/.env    secrets / settings
  /var/lib/it-consultant/db  WATCH_PATH data
  /etc/systemd/system/reindex.service
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest-dir)
      DEST_DIR="${2:?--dest-dir requires a path}"
      shift 2
      ;;
    --layout-only)
      LAYOUT_ONLY=1
      shift
      ;;
    --enable)
      ENABLE=1
      shift
      ;;
    --undeploy)
      UNDEPLOY=1
      shift
      ;;
    --configure)
      CONFIGURE=yes
      shift
      ;;
    --no-configure)
      CONFIGURE=no
      shift
      ;;
    --no-create-user)
      CREATE_USER=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

root() {
  printf '%s%s' "${DEST_DIR}" "$1"
}

log() {
  printf 'install-reindex: %s\n' "$*"
}

is_fake_root() {
  [[ -n "${DEST_DIR}" ]]
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "install-reindex: required command not found: $1" >&2
    exit 1
  }
}

create_service_user() {
  if is_fake_root || [[ "${CREATE_USER}" -eq 0 ]]; then
    return 0
  fi
  if [[ "$(id -u)" -ne 0 ]]; then
    echo "install-reindex: root required to create user (or pass --dest-dir / --no-create-user)" >&2
    exit 1
  fi
  if ! getent group "${SERVICE_GROUP}" >/dev/null; then
    log "creating group ${SERVICE_GROUP}"
    groupadd --system "${SERVICE_GROUP}"
  fi
  if ! getent passwd "${SERVICE_USER}" >/dev/null; then
    log "creating user ${SERVICE_USER}"
    useradd --system --gid "${SERVICE_GROUP}" --home-dir "${OPT_DIR}" \
      --shell /usr/sbin/nologin "${SERVICE_USER}"
  fi
}

should_configure_env() {
  case "${CONFIGURE}" in
    yes) return 0 ;;
    no) return 1 ;;
    auto)
      [[ -t 0 ]]
      ;;
    *)
      return 1
      ;;
  esac
}

get_env_value() {
  local file="$1" key="$2" line=""
  line="$(grep -E "^${key}=" "${file}" 2>/dev/null | tail -n1 || true)"
  if [[ -z "${line}" ]]; then
    printf ''
    return 0
  fi
  printf '%s' "${line#"${key}="}"
}

set_env_value() {
  local file="$1" key="$2" value="$3"
  local tmp
  tmp="$(mktemp)"
  if grep -qE "^${key}=" "${file}"; then
    while IFS= read -r line || [[ -n "${line}" ]]; do
      if [[ "${line}" == "${key}="* ]]; then
        printf '%s=%s\n' "${key}" "${value}"
      else
        printf '%s\n' "${line}"
      fi
    done <"${file}" >"${tmp}"
  elif grep -qE "^[[:space:]]*#[[:space:]]*${key}=" "${file}"; then
    while IFS= read -r line || [[ -n "${line}" ]]; do
      if [[ "${line}" =~ ^[[:space:]]*#[[:space:]]*${key}= ]]; then
        printf '%s=%s\n' "${key}" "${value}"
      else
        printf '%s\n' "${line}"
      fi
    done <"${file}" >"${tmp}"
  else
    cat "${file}" >"${tmp}"
    printf '%s=%s\n' "${key}" "${value}" >>"${tmp}"
  fi
  cat "${tmp}" >"${file}"
  rm -f "${tmp}"
}

prompt_env_value() {
  local key="$1" current="$2" secret="${3:-0}"
  local display input=""
  if [[ "${secret}" -eq 1 ]]; then
    if [[ -n "${current}" ]]; then
      display="****"
    else
      display="empty"
    fi
    printf '  %s [%s]: ' "${key}" "${display}" >&2
    IFS= read -r -s input || true
    printf '\n' >&2
  else
    display="${current}"
    printf '  %s [%s]: ' "${key}" "${display}" >&2
    IFS= read -r input || true
  fi
  if [[ -z "${input}" ]]; then
    printf '%s' "${current}"
  else
    printf '%s' "${input}"
  fi
}

is_secret_env_key() {
  local key="$1"
  [[ "${key}" == *PASSWORD* || "${key}" == *SECRET* || "${key}" == *TOKEN* ]]
}

configure_env_file() {
  local env_file="$1"
  local key current value

  if ! should_configure_env; then
    log "skipping interactive .env configure"
    return 0
  fi

  log "configure .env (${#REINDEX_ENV_KEYS[@]} reindex vars; Enter keeps current)"
  for key in "${REINDEX_ENV_KEYS[@]}"; do
    current="$(get_env_value "${env_file}" "${key}")"
    if is_secret_env_key "${key}"; then
      value="$(prompt_env_value "${key}" "${current}" 1)"
    else
      value="$(prompt_env_value "${key}" "${current}" 0)"
    fi
    if [[ -z "${value}" ]] && ! grep -qE "^${key}=" "${env_file}"; then
      continue
    fi
    set_env_value "${env_file}" "${key}" "${value}"
  done
  log "wrote reindex settings into ${ETC_DIR}/.env"
}

install_layout() {
  log "creating directories under ${DEST_DIR:-/}"
  mkdir -p "$(root "${OPT_DIR}")"
  mkdir -p "$(root "${ETC_DIR}")"
  mkdir -p "$(root "${VAR_DIR}/db")"
  mkdir -p "$(root /etc/systemd/system)"

  local env_dst
  env_dst="$(root "${ETC_DIR}/.env")"
  if [[ ! -f "${env_dst}" ]]; then
    log "writing ${ETC_DIR}/.env from .env.example"
    sed "s|^WATCH_PATH=.*|WATCH_PATH=${VAR_DIR}/db|" \
      "${REPO_ROOT}/.env.example" >"${env_dst}"
    chmod 600 "${env_dst}"
  else
    log "keeping existing ${ETC_DIR}/.env"
  fi

  configure_env_file "${env_dst}"

  log "installing ${UNIT}"
  install -m 644 \
    "${REPO_ROOT}/deploy/systemd/${UNIT}" \
    "$(root /etc/systemd/system)/"
}

copy_app_sources() {
  local dest
  dest="$(root "${OPT_DIR}")"
  mkdir -p "${dest}"
  log "copying application sources to ${OPT_DIR}"
  local item
  for item in pyproject.toml README.md LICENSE common mail_gateway reindex; do
    if [[ -e "${REPO_ROOT}/${item}" ]]; then
      rm -rf "${dest}/${item}"
      cp -a "${REPO_ROOT}/${item}" "${dest}/"
    fi
  done
}

install_venv() {
  require_cmd "${PYTHON}"
  copy_app_sources

  local venv_dir dest
  dest="$(root "${OPT_DIR}")"
  venv_dir="${dest}/.venv"
  if [[ ! -x "${venv_dir}/bin/python" ]]; then
    log "creating venv at ${OPT_DIR}/.venv"
    if ! "${PYTHON}" -m venv "${venv_dir}"; then
      echo "install-reindex: failed to create venv (is python3-venv installed?)" >&2
      echo "install-reindex: tip: use --layout-only for a filesystem-only dry run" >&2
      exit 1
    fi
  fi

  log "installing package extras [reindex] (docling, watchdog) into venv"
  "${venv_dir}/bin/pip" install -U pip
  "${venv_dir}/bin/pip" install -e "${dest}[reindex]"
}

fix_ownership() {
  if is_fake_root || [[ "${CREATE_USER}" -eq 0 ]]; then
    return 0
  fi
  if [[ "$(id -u)" -ne 0 ]]; then
    return 0
  fi
  if getent passwd "${SERVICE_USER}" >/dev/null; then
    log "fixing ownership to ${SERVICE_USER}:${SERVICE_GROUP}"
    chown -R "${SERVICE_USER}:${SERVICE_GROUP}" \
      "$(root "${OPT_DIR}")" \
      "$(root "${ETC_DIR}")" \
      "$(root "${VAR_DIR}")"
  fi
}

enable_systemd() {
  if is_fake_root; then
    log "skipping systemctl (fake root / DESTDIR)"
    return 0
  fi
  if [[ "${ENABLE}" -ne 1 ]]; then
    log "unit installed; enable later with: systemctl enable --now ${UNIT}"
    return 0
  fi
  if [[ "$(id -u)" -ne 0 ]]; then
    echo "install-reindex: --enable requires root" >&2
    exit 1
  fi
  if ! command -v systemctl >/dev/null 2>&1; then
    echo "install-reindex: systemctl not found (WSL without systemd?)" >&2
    echo "install-reindex: start manually:" >&2
    echo "  sudo -u ${SERVICE_USER} ${OPT_DIR}/.venv/bin/python -m reindex" >&2
    exit 1
  fi
  log "daemon-reload and enable --now ${UNIT}"
  systemctl daemon-reload
  systemctl enable --now "${UNIT}"
}

stop_systemd() {
  if is_fake_root; then
    log "skipping systemctl (fake root / DESTDIR)"
    return 0
  fi
  if [[ "$(id -u)" -ne 0 ]]; then
    echo "install-reindex: --undeploy requires root on a real system" >&2
    exit 1
  fi
  if ! command -v systemctl >/dev/null 2>&1; then
    log "systemctl not found; skipping stop/disable"
    return 0
  fi
  log "disable --now ${UNIT}"
  systemctl disable --now "${UNIT}" 2>/dev/null || true
  systemctl daemon-reload || true
  systemctl reset-failed 2>/dev/null || true
}

remove_paths() {
  local path
  for path in \
    "$(root "${OPT_DIR}")" \
    "$(root "${ETC_DIR}")" \
    "$(root "${VAR_DIR}")"
  do
    if [[ -e "${path}" ]]; then
      log "removing ${path#"${DEST_DIR}"}"
      rm -rf "${path}"
    fi
  done

  local unit_path
  unit_path="$(root "/etc/systemd/system/${UNIT}")"
  if [[ -e "${unit_path}" || -L "${unit_path}" ]]; then
    log "removing /etc/systemd/system/${UNIT}"
    rm -f "${unit_path}"
  fi
}

remove_service_user() {
  if is_fake_root; then
    return 0
  fi
  if [[ "$(id -u)" -ne 0 ]]; then
    return 0
  fi
  local mail_unit
  mail_unit="/etc/systemd/system/mail-gateway.service"
  if [[ -e "${mail_unit}" || -L "${mail_unit}" ]]; then
    log "keeping user ${SERVICE_USER} (mail-gateway.service still present)"
    return 0
  fi
  if getent passwd "${SERVICE_USER}" >/dev/null; then
    log "removing user ${SERVICE_USER}"
    userdel "${SERVICE_USER}" 2>/dev/null || true
  fi
  if getent group "${SERVICE_GROUP}" >/dev/null; then
    log "removing group ${SERVICE_GROUP}"
    groupdel "${SERVICE_GROUP}" 2>/dev/null || true
  fi
}

undeploy() {
  if is_fake_root; then
    log "fake-root undeploy DESTDIR=${DEST_DIR}"
    if [[ ! -d "${DEST_DIR}" ]]; then
      log "nothing to remove (${DEST_DIR} missing)"
      return 0
    fi
    DEST_DIR="$(cd "${DEST_DIR}" && pwd)"
  elif [[ "$(id -u)" -ne 0 ]]; then
    echo "install-reindex: production undeploy requires root, or use --dest-dir for a test root" >&2
    exit 1
  else
    log "production undeploy (reindex only)"
  fi

  stop_systemd
  remove_paths
  remove_service_user

  if ! is_fake_root && command -v systemctl >/dev/null 2>&1; then
    systemctl daemon-reload || true
  fi

  log "undeploy done"
}

main() {
  if [[ "${UNDEPLOY}" -eq 1 ]]; then
    undeploy
    return 0
  fi

  if is_fake_root; then
    log "fake-root install DESTDIR=${DEST_DIR}"
    mkdir -p "${DEST_DIR}"
    DEST_DIR="$(cd "${DEST_DIR}" && pwd)"
  elif [[ "$(id -u)" -ne 0 ]]; then
    echo "install-reindex: production install requires root, or use --dest-dir for a test root" >&2
    exit 1
  fi

  create_service_user
  install_layout

  if [[ "${LAYOUT_ONLY}" -eq 0 ]]; then
    install_venv
  else
    log "layout-only: skipping copy/venv/pip"
  fi

  fix_ownership
  enable_systemd

  log "done"
  log "  app:   ${DEST_DIR}${OPT_DIR}"
  log "  env:   ${DEST_DIR}${ETC_DIR}/.env"
  log "  data:  ${DEST_DIR}${VAR_DIR}/db"
  log "  unit:  ${DEST_DIR}/etc/systemd/system/${UNIT}"
}

main
