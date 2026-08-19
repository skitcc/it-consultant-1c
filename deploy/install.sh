#!/usr/bin/env bash
# Install / undeploy IT Consultant services into a filesystem tree.
#
# Production (needs root):
#   sudo ./deploy/install.sh
#   sudo ./deploy/install.sh --enable
#   sudo ./deploy/install.sh --only api-gateway --enable
#   sudo ./deploy/install.sh --only reindex --enable
#   sudo ./deploy/install.sh --undeploy
#
# Safe fake-root (no systemd, no user creation, does not touch host /):
#   ./deploy/install.sh --dest-dir /tmp/itc-root
#   ./deploy/install.sh --dest-dir /tmp/itc-root --layout-only
#   ./deploy/install.sh --dest-dir /tmp/itc-root --undeploy
#
# Environment:
#   DESTDIR  — same as --dest-dir (packaging convention)
#   PYTHON   — interpreter for venv (default: python3)
#   TORCH_CPU_INDEX — PyTorch CPU wheel index (default: pytorch.org cpu)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OPT_DIR="/opt/it-consultant"
ETC_DIR="/etc/it-consultant"
VAR_DIR="/var/lib/it-consultant"
SERVICE_USER="it-consultant"
SERVICE_GROUP="it-consultant"
UNITS=(
  api-gateway.service
  reindex.service
  mail-gateway.service
  it-consultant.target
  knowledge-sync.service
)

DEST_DIR="${DESTDIR:-}"
LAYOUT_ONLY=0
ENABLE=0
CREATE_USER=1
UNDEPLOY=0
# empty | api-gateway | reindex | mail-gateway
ONLY=""
# configure: auto | yes | no  (auto = prompt when stdin is a TTY)
CONFIGURE=auto
PYTHON="${PYTHON:-python3}"
TORCH_CPU_INDEX="${TORCH_CPU_INDEX:-https://download.pytorch.org/whl/cpu}"

usage() {
  cat <<'EOF'
Usage: deploy/install.sh [options]

Options:
  --dest-dir DIR   Install/undeploy under DIR as fake root (sets DESTDIR).
                   Skips systemctl and useradd/userdel. Safe for local/CI tests.
  --layout-only    Create dirs, .env, systemd units only (no copy/venv/pip).
  --only NAME      Enable/start only: api-gateway, reindex or mail-gateway.
                   Still copies the app, creates venv, and installs all units
                   and API/indexing dependencies. Ignored with --undeploy.
  --enable         systemctl daemon-reload && enable --now
                   it-consultant.target (or NAME.service when --only is set).
                   Ignored when --dest-dir is set.
  --undeploy       Stop/disable units, remove app/env/data/units and service
                   user (production). With --dest-dir only removes the fake tree.
  --configure      Prompt for every variable from .env.example (Enter keeps
                   current). Forced even when stdin is not a TTY.
  --no-configure   Do not prompt for .env values (default for non-TTY / CI).
  --no-create-user Do not create service user/group (real install only).
  -h, --help       Show this help.

Paths (always absolute on the target system; prefixed by --dest-dir when set):
  /opt/it-consultant       application + .venv
  /etc/it-consultant/.env  secrets / settings
  /var/lib/it-consultant     persistent document registry and OWUI uploads
  /etc/systemd/system/     unit files

Interactive configure (TTY by default, or --configure) asks for all KEY=
entries parsed from .env.example (including commented optional keys).
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
    --only)
      ONLY="${2:?--only requires a service name}"
      case "${ONLY}" in
        api-gateway|reindex|mail-gateway) ;;
        *)
          echo "install: unsupported --only service: ${ONLY}" >&2
          usage >&2
          exit 2
          ;;
      esac
      shift 2
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

# Parse KEY names from .env.example (active KEY= and commented # KEY= lines).
list_env_keys_from_example() {
  local file="${REPO_ROOT}/.env.example"
  local line key
  declare -A seen=()
  while IFS= read -r line || [[ -n "${line}" ]]; do
    if [[ "${line}" =~ ^[[:space:]]*#[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)= ]]; then
      key="${BASH_REMATCH[1]}"
    elif [[ "${line}" =~ ^([A-Za-z_][A-Za-z0-9_]*)= ]]; then
      key="${BASH_REMATCH[1]}"
    else
      continue
    fi
    if [[ -z "${seen[${key}]+x}" ]]; then
      seen["${key}"]=1
      printf '%s\n' "${key}"
    fi
  done <"${file}"
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
    # -r keeps DOMAIN\user backslashes; -s hides password input.
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
  local -a keys=()

  if ! should_configure_env; then
    log "skipping interactive .env configure"
    return 0
  fi

  mapfile -t keys < <(list_env_keys_from_example)
  if [[ "${#keys[@]}" -eq 0 ]]; then
    log "no keys found in .env.example; skipping configure"
    return 0
  fi

  log "configure .env (${#keys[@]} vars from .env.example; Enter keeps current)"
  for key in "${keys[@]}"; do
    current="$(get_env_value "${env_file}" "${key}")"
    if is_secret_env_key "${key}"; then
      value="$(prompt_env_value "${key}" "${current}" 1)"
    else
      value="$(prompt_env_value "${key}" "${current}" 0)"
    fi
    # Optional commented keys: do not create empty KEY= on Enter.
    if [[ -z "${value}" ]] && ! grep -qE "^${key}=" "${env_file}"; then
      continue
    fi
    set_env_value "${env_file}" "${key}" "${value}"
  done
  log "wrote settings into ${ETC_DIR}/.env"
}

install_layout() {
  log "creating directories under ${DEST_DIR:-/}"
  mkdir -p "$(root "${OPT_DIR}")"
  mkdir -p "$(root "${ETC_DIR}")"
  mkdir -p "$(root "${VAR_DIR}")"
  mkdir -p "$(root "${VAR_DIR}/owui-data/uploads")"
  mkdir -p "$(root /etc/systemd/system)"

  local env_dst
  env_dst="$(root "${ETC_DIR}/.env")"
  if [[ ! -f "${env_dst}" ]]; then
    log "writing ${ETC_DIR}/.env from .env.example"
    cp "${REPO_ROOT}/.env.example" "${env_dst}"
    chmod 600 "${env_dst}"
  else
    log "keeping existing ${ETC_DIR}/.env"
  fi

  configure_env_file "${env_dst}"

  log "installing systemd units"
  install -m 644 \
    "${REPO_ROOT}/deploy/systemd/api-gateway.service" \
    "${REPO_ROOT}/deploy/systemd/reindex.service" \
    "${REPO_ROOT}/deploy/systemd/mail-gateway.service" \
    "${REPO_ROOT}/deploy/systemd/it-consultant.target" \
    "$(root /etc/systemd/system)/"
  if ! is_fake_root && command -v systemctl >/dev/null 2>&1; then
    systemctl disable --now knowledge-sync.service 2>/dev/null || true
  fi
  rm -f "$(root /etc/systemd/system/knowledge-sync.service)"
}

copy_app_sources() {
  local dest
  dest="$(root "${OPT_DIR}")"
  mkdir -p "${dest}"
  log "copying application sources to ${OPT_DIR}"
  local item
  for item in pyproject.toml README.md LICENSE common knowledge api_gateway knowledge_sync mail_gateway reindex integrations; do
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
      echo "install: failed to create venv (is python3-venv installed?)" >&2
      echo "install: tip: use --layout-only for a filesystem-only dry run" >&2
      exit 1
    fi
  fi

  "${venv_dir}/bin/pip" install -U pip
  # Docling needs torch; default PyPI wheels are CUDA. Pin CPU wheels first so
  # the follow-up install does not pull multi-GB NVIDIA packages on a CPU host.
  log "installing CPU-only PyTorch from ${TORCH_CPU_INDEX}"
  "${venv_dir}/bin/pip" install \
    --index-url "${TORCH_CPU_INDEX}" \
    torch torchvision
  log "installing Python package into venv (.[api,reindex])"
  "${venv_dir}/bin/pip" install \
    --extra-index-url "${TORCH_CPU_INDEX}" \
    -e "${dest}[api,reindex]"
}

enable_unit() {
  if [[ -n "${ONLY}" ]]; then
    printf '%s.service' "${ONLY}"
  else
    printf 'it-consultant.target'
  fi
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
  local unit
  unit="$(enable_unit)"

  if is_fake_root; then
    log "skipping systemctl (fake root / DESTDIR)"
    if [[ "${ENABLE}" -eq 1 ]]; then
      log "would enable: ${unit}"
    else
      log "units installed; enable later with: systemctl enable --now ${unit}"
    fi
    return 0
  fi
  if [[ "${ENABLE}" -ne 1 ]]; then
    log "units installed; enable later with: systemctl enable --now ${unit}"
    return 0
  fi
  if [[ "$(id -u)" -ne 0 ]]; then
    echo "install: --enable requires root" >&2
    exit 1
  fi
  if ! command -v systemctl >/dev/null 2>&1; then
    echo "install: systemctl not found (WSL without systemd?)" >&2
    echo "install: start manually, e.g.:" >&2
    echo "  sudo -u ${SERVICE_USER} ${OPT_DIR}/.venv/bin/python -m api_gateway" >&2
    exit 1
  fi
  log "daemon-reload and enable --now ${unit}"
  systemctl daemon-reload
  systemctl enable --now "${unit}"
}

stop_systemd() {
  if is_fake_root; then
    log "skipping systemctl (fake root / DESTDIR)"
    return 0
  fi
  if [[ "$(id -u)" -ne 0 ]]; then
    echo "install: --undeploy requires root on a real system" >&2
    exit 1
  fi
  if ! command -v systemctl >/dev/null 2>&1; then
    log "systemctl not found; skipping stop/disable"
    return 0
  fi
  log "disable --now it-consultant.target (and related units)"
  systemctl disable --now it-consultant.target 2>/dev/null || true
  local unit
  for unit in "${UNITS[@]}"; do
    systemctl disable --now "${unit}" 2>/dev/null || true
  done
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

  local unit unit_path
  for unit in "${UNITS[@]}"; do
    unit_path="$(root "/etc/systemd/system/${unit}")"
    if [[ -e "${unit_path}" || -L "${unit_path}" ]]; then
      log "removing /etc/systemd/system/${unit}"
      rm -f "${unit_path}"
    fi
  done
}

remove_service_user() {
  if is_fake_root; then
    return 0
  fi
  if [[ "$(id -u)" -ne 0 ]]; then
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
    echo "install: production undeploy requires root, or use --dest-dir for a test root" >&2
    exit 1
  else
    log "production undeploy"
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
  log "  app:   ${DEST_DIR}${OPT_DIR}"
  log "  env:   ${DEST_DIR}${ETC_DIR}/.env"
  log "  data:  ${DEST_DIR}${VAR_DIR}"
  log "  units: ${DEST_DIR}/etc/systemd/system/"
}

main
