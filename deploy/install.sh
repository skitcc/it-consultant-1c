#!/usr/bin/env bash
# Install IT Consultant services into a filesystem tree.
#
# Production (needs root):
#   sudo ./deploy/install.sh --enable
# Then edit the docs mount What= / credentials and restart the mount unit.
#
# Safe fake-root (no systemd, no user creation, does not touch host /):
#   ./deploy/install.sh --dest-dir /tmp/itc-root --layout-only
#
# Environment:
#   DESTDIR  — same as --dest-dir (packaging convention)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OPT_DIR="/opt/it-consultant"
ETC_DIR="/etc/it-consultant"
VAR_DIR="/var/lib/it-consultant"
DOCS_DIR="${VAR_DIR}/db"
SERVICE_USER="it-consultant"
SERVICE_GROUP="it-consultant"
# systemd-escape -p --suffix=mount /var/lib/it-consultant/db
DOCS_MOUNT_UNIT='var-lib-it\x2dconsultant-db.mount'

DEST_DIR="${DESTDIR:-}"
LAYOUT_ONLY=0
ENABLE=0
CREATE_USER=1
PYTHON="${PYTHON:-python3}"

usage() {
  cat <<'EOF'
Usage: deploy/install.sh [options]

Options:
  --dest-dir DIR   Install under DIR as fake root (sets DESTDIR). Skips
                   systemctl and useradd. Safe for local/CI tests.
  --layout-only    Create dirs, .env, systemd units only (no venv/pip).
  --enable         systemctl daemon-reload && enable mount + it-consultant.target
                   (ignored when --dest-dir is set).
  --no-create-user Do not create service user/group (real install only).
  -h, --help       Show this help.

Paths (always absolute on the target system; prefixed by --dest-dir when set):
  /opt/it-consultant                 application + .venv
  /etc/it-consultant/.env            secrets / settings
  /etc/it-consultant/docs-credentials  CIFS credentials for docs.mount
  /var/lib/it-consultant/db          documents mount point (WATCH_PATH)
  /etc/systemd/system/               unit files (+ docs .mount)
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

# Packaging root: files go to ${DEST_DIR}${absolute_path}
root() {
  printf '%s%s' "${DEST_DIR}" "$1"
}

log() {
  printf 'install: %s\n' "$*"
}

is_fake_root() {
  [[ -n "${DEST_DIR}" ]]
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "install: required command not found: $1" >&2
    exit 1
  }
}

create_service_user() {
  if is_fake_root || [[ "${CREATE_USER}" -eq 0 ]]; then
    return 0
  fi
  if [[ "$(id -u)" -ne 0 ]]; then
    echo "install: root required to create user (or pass --dest-dir / --no-create-user)" >&2
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

install_layout() {
  log "creating directories under ${DEST_DIR:-/}"
  mkdir -p "$(root "${OPT_DIR}")"
  mkdir -p "$(root "${ETC_DIR}")"
  mkdir -p "$(root "${DOCS_DIR}")"
  mkdir -p "$(root /etc/systemd/system)"

  local env_dst
  env_dst="$(root "${ETC_DIR}/.env")"
  if [[ ! -f "${env_dst}" ]]; then
    log "writing ${ETC_DIR}/.env from .env.example"
    sed "s|^WATCH_PATH=.*|WATCH_PATH=${DOCS_DIR}|" \
      "${REPO_ROOT}/.env.example" >"${env_dst}"
    chmod 600 "${env_dst}"
  else
    log "keeping existing ${ETC_DIR}/.env"
  fi

  local creds_dst
  creds_dst="$(root "${ETC_DIR}/docs-credentials")"
  if [[ ! -f "${creds_dst}" ]]; then
    log "writing ${ETC_DIR}/docs-credentials from example (edit username/password)"
    install -m 600 \
      "${REPO_ROOT}/deploy/systemd/docs-credentials.example" \
      "${creds_dst}"
  else
    log "keeping existing ${ETC_DIR}/docs-credentials"
  fi

  log "installing systemd units (services + docs mount)"
  install -m 644 \
    "${REPO_ROOT}/deploy/systemd/mail-gateway.service" \
    "${REPO_ROOT}/deploy/systemd/reindex.service" \
    "${REPO_ROOT}/deploy/systemd/it-consultant.target" \
    "$(root /etc/systemd/system)/"
  install -m 644 \
    "${REPO_ROOT}/deploy/systemd/docs.mount" \
    "$(root /etc/systemd/system)/${DOCS_MOUNT_UNIT}"
}

install_venv() {
  require_cmd "${PYTHON}"
  local venv_dir
  venv_dir="$(root "${OPT_DIR}/.venv")"
  if [[ ! -x "${venv_dir}/bin/python" ]]; then
    log "creating venv at ${OPT_DIR}/.venv"
    if ! "${PYTHON}" -m venv "${venv_dir}"; then
      echo "install: failed to create venv (is python3-venv installed?)" >&2
      echo "install: tip: use --layout-only for a filesystem-only dry run" >&2
      exit 1
    fi
  fi
  log "installing Python package into venv (mail_gateway + reindex + common)"
  "${venv_dir}/bin/pip" install --upgrade pip
  "${venv_dir}/bin/pip" install "${REPO_ROOT}[reindex]"
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
    log "units installed; next:"
    log "  1) edit What= in /etc/systemd/system/${DOCS_MOUNT_UNIT}"
    log "  2) edit /etc/it-consultant/docs-credentials"
    log "  3) systemctl daemon-reload"
    log "  4) systemctl enable --now '${DOCS_MOUNT_UNIT}' it-consultant.target"
    return 0
  fi
  if [[ "$(id -u)" -ne 0 ]]; then
    echo "install: --enable requires root" >&2
    exit 1
  fi
  require_cmd systemctl
  log "daemon-reload"
  systemctl daemon-reload

  log "enable docs mount unit (${DOCS_MOUNT_UNIT})"
  systemctl enable "${DOCS_MOUNT_UNIT}"
  # Start may fail until What=/credentials are fixed — that is expected right after deploy.
  if systemctl start "${DOCS_MOUNT_UNIT}"; then
    log "docs mount started OK"
  else
    log "WARN: docs mount failed to start (edit What= / docs-credentials, then restart the mount)"
  fi

  log "enable --now it-consultant.target"
  systemctl enable --now it-consultant.target
}

main() {
  if is_fake_root; then
    log "fake-root install DESTDIR=${DEST_DIR}"
    mkdir -p "${DEST_DIR}"
    DEST_DIR="$(cd "${DEST_DIR}" && pwd)"
  elif [[ "$(id -u)" -ne 0 ]]; then
    echo "install: production install requires root, or use --dest-dir for a test root" >&2
    exit 1
  fi

  create_service_user
  install_layout

  if [[ "${LAYOUT_ONLY}" -eq 0 ]]; then
    install_venv
  else
    log "layout-only: skipping venv/pip"
  fi

  fix_ownership
  enable_systemd

  log "done"
  log "  app:     ${DEST_DIR}${OPT_DIR}"
  log "  env:     ${DEST_DIR}${ETC_DIR}/.env"
  log "  creds:   ${DEST_DIR}${ETC_DIR}/docs-credentials"
  log "  docs:    ${DEST_DIR}${DOCS_DIR}  (WATCH_PATH / mount Where=)"
  log "  mount:   ${DEST_DIR}/etc/systemd/system/${DOCS_MOUNT_UNIT}"
  log "  units:   ${DEST_DIR}/etc/systemd/system/"
  log "After deploy: set What= in the mount unit + docs-credentials, then:"
  log "  systemctl daemon-reload && systemctl restart '${DOCS_MOUNT_UNIT}' reindex"
}

main
