#!/bin/zsh
# Privacy-safe macOS launcher. It prepares only the application runtime and GUI
# dependencies. OCR/ML resources stay deferred to explicit in-app confirmation.
set -u
setopt PIPE_FAIL
setopt NULL_GLOB

BUNDLE_CONTENTS=${0:A:h:h}
APP_BUNDLE=${BUNDLE_CONTENTS:h}
SUPPORT_DIR="$HOME/Library/Application Support/Manga HD Transfer Studio"
RUNTIME_DIR="$SUPPORT_DIR/native-runtime-v1"
DOWNLOAD_DIR="$SUPPORT_DIR/downloads"
LOG_DIR="$SUPPORT_DIR/logs"
LOG_FILE="$LOG_DIR/launcher.log"
PYTHON_RELEASE="20251010"
PYTHON_VERSION="3.12.12"

mkdir -p "$SUPPORT_DIR" "$DOWNLOAD_DIR" "$LOG_DIR"
exec >>"$LOG_FILE" 2>&1
print "\n==== $(date '+%Y-%m-%d %H:%M:%S') Manga HD Transfer launch ===="

show_error() {
  local message="$1"
  local safe_message="${message//\"/\\\"}"
  print "ERROR: $message"
  /usr/bin/osascript -e "display dialog \"$safe_message\" buttons {\"OK\"} default button 1 with icon stop" >/dev/null 2>&1 || true
}

show_notice() {
  local message="$1"
  local safe_message="${message//\"/\\\"}"
  /usr/bin/osascript -e "display notification \"$safe_message\" with title \"Manga HD Transfer Studio\"" >/dev/null 2>&1 || true
}

if [[ "$(/usr/bin/uname -s)" != "Darwin" ]]; then
  show_error "This launcher supports macOS only."
  exit 2
fi

ARCH="$(/usr/bin/uname -m)"
case "$ARCH" in
  arm64) PY_ARCH="aarch64" ;;
  x86_64) PY_ARCH="x86_64" ;;
  *) show_error "Unsupported Mac architecture."; exit 2 ;;
esac

EMBEDDED_SOURCE="$BUNDLE_CONTENTS/Resources/app"
APP_SOURCE="$SUPPORT_DIR/app-source"
if [[ ! -f "$EMBEDDED_SOURCE/run_gui.py" || ! -f "$EMBEDDED_SOURCE/bootstrap.py" ]]; then
  show_error "The application package is incomplete. Please download it again."
  exit 2
fi
mkdir -p "$APP_SOURCE"
/usr/bin/rsync -a --delete \
  --exclude '.git/' \
  --exclude '.runtime/' \
  --exclude '.venv*/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '*.pyc' \
  --exclude '*.log' \
  "$EMBEDDED_SOURCE/" "$APP_SOURCE/" || {
    show_error "Unable to prepare the writable application source directory."
    exit 1
  }

clean_injected_environment() {
  unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONUSERBASE
  unset QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH QML2_IMPORT_PATH QML_IMPORT_PATH
  unset DYLD_LIBRARY_PATH DYLD_FRAMEWORK_PATH DYLD_FALLBACK_LIBRARY_PATH DYLD_FALLBACK_FRAMEWORK_PATH
  export PYTHONNOUSERSITE=1
  export PYTHONUNBUFFERED=1
  export QT_API=pyside6
  export PIP_DISABLE_PIP_VERSION_CHECK=1
  export PIP_NO_INPUT=1
  export MHD_APP_BUNDLE="$APP_BUNDLE"
}

fetch_file() {
  local output="$1"
  shift
  local url
  for url in "$@"; do
    if /usr/bin/curl --fail --location --retry 3 --connect-timeout 20 --speed-time 30 --speed-limit 1024 --output "$output.part" "$url"; then
      /bin/mv -f "$output.part" "$output"
      return 0
    fi
    /bin/rm -f "$output.part"
  done
  return 1
}

ARCHIVE="cpython-${PYTHON_VERSION}+${PYTHON_RELEASE}-${PY_ARCH}-apple-darwin-install_only_stripped.tar.gz"
ARCHIVE_PATH="$DOWNLOAD_DIR/$ARCHIVE"
SUMS_PATH="$DOWNLOAD_DIR/SHA256SUMS-${PYTHON_RELEASE}"
MIRROR_ROOT="https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone/${PYTHON_RELEASE}"
OFFICIAL_ROOT="https://github.com/astral-sh/python-build-standalone/releases/download/${PYTHON_RELEASE}"
PY="$RUNTIME_DIR/python/bin/python3"

install_python_runtime() {
  show_notice "First launch prepares Python and main GUI dependencies only. OCR/ML models remain on-demand."
  /bin/rm -rf "$RUNTIME_DIR.new"
  /bin/mkdir -p "$RUNTIME_DIR.new"
  if [[ ! -s "$ARCHIVE_PATH" ]]; then
    fetch_file "$ARCHIVE_PATH" "$MIRROR_ROOT/$ARCHIVE" "$OFFICIAL_ROOT/$ARCHIVE" || {
      show_error "Unable to download the standalone Python runtime."
      return 1
    }
  fi
  if [[ ! -s "$SUMS_PATH" ]]; then
    fetch_file "$SUMS_PATH" "$MIRROR_ROOT/SHA256SUMS" "$OFFICIAL_ROOT/SHA256SUMS" || {
      show_error "Unable to download the Python checksum file."
      return 1
    }
  fi
  local expected actual
  expected=$(/usr/bin/awk -v filename="$ARCHIVE" '$2 == filename || $2 == "*" filename {print $1; exit}' "$SUMS_PATH")
  actual=$(/usr/bin/shasum -a 256 "$ARCHIVE_PATH" | /usr/bin/awk '{print $1}')
  if [[ -z "$expected" || "$expected" != "$actual" ]]; then
    /bin/rm -f "$ARCHIVE_PATH" "$SUMS_PATH"
    show_error "Standalone Python SHA-256 verification failed; downloaded files were removed."
    return 1
  fi
  /usr/bin/tar -xzf "$ARCHIVE_PATH" -C "$RUNTIME_DIR.new" || {
    /bin/rm -f "$ARCHIVE_PATH"
    show_error "The standalone Python archive is invalid and was removed."
    return 1
  }
  if [[ ! -x "$RUNTIME_DIR.new/python/bin/python3" ]]; then
    show_error "Standalone Python is unavailable after extraction."
    return 1
  fi
  /bin/rm -rf "$RUNTIME_DIR"
  /bin/mv "$RUNTIME_DIR.new" "$RUNTIME_DIR"
}

clean_injected_environment
if [[ ! -x "$PY" ]]; then
  install_python_runtime || exit 1
fi

PY_ARCH_ACTUAL=$("$PY" -c 'import platform; print(platform.machine())' 2>/dev/null || true)
if [[ "$ARCH" == "arm64" && "$PY_ARCH_ACTUAL" != "arm64" ]]; then
  /bin/rm -rf "$RUNTIME_DIR"
  show_error "Standalone Python architecture mismatch; runtime was cleared."
  exit 1
fi

ARGS=("$APP_SOURCE/bootstrap.py" --install-main-deps)
if [[ "${MHD_PREPARE_ONLY:-0}" != "1" ]]; then
  ARGS+=(--launch)
fi

cd "$APP_SOURCE" || exit 1
if [[ "$ARCH" == "arm64" ]]; then
  /usr/bin/arch -arm64 "$PY" -X faulthandler "${ARGS[@]}"
else
  /usr/bin/arch -x86_64 "$PY" -X faulthandler "${ARGS[@]}"
fi
STATUS=$?
if [[ $STATUS -ne 0 ]]; then
  show_error "Application preparation or startup failed (code $STATUS). See the local Application Support log."
fi
exit $STATUS
