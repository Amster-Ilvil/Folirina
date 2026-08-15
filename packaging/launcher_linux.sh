#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
SUPPORT_DIR="$DATA_HOME/MangaHDTransferStudio"
RUNTIME_DIR="$SUPPORT_DIR/native-runtime-v1"
VENV_DIR="$SUPPORT_DIR/app-venv-v1"
DOWNLOAD_DIR="$SUPPORT_DIR/downloads"
LOG_DIR="$SUPPORT_DIR/logs"
LOG_FILE="$LOG_DIR/launcher.log"
PYTHON_RELEASE="20251010"
PYTHON_VERSION="3.12.12"

mkdir -p "$SUPPORT_DIR" "$DOWNLOAD_DIR" "$LOG_DIR"
exec >> >(tee -a "$LOG_FILE") 2>&1

supported_python() {
  local exe="$1"
  [[ -x "$exe" ]] || return 1
  "$exe" -c 'import sys; raise SystemExit(0 if sys.version_info.major == 3 and 11 <= sys.version_info.minor <= 13 else 8)' >/dev/null 2>&1
}

find_system_python() {
  local name
  if [[ -n "${MHD_TRANSFER_PYTHON:-}" ]] && supported_python "$MHD_TRANSFER_PYTHON"; then
    printf '%s\n' "$MHD_TRANSFER_PYTHON"; return 0
  fi
  for name in python3.13 python3.12 python3.11 python3; do
    if command -v "$name" >/dev/null 2>&1 && supported_python "$(command -v "$name")"; then
      command -v "$name"; return 0
    fi
  done
  return 1
}

fetch_file() {
  local output="$1"; shift
  local url
  for url in "$@"; do
    rm -f "$output.part"
    if command -v curl >/dev/null 2>&1; then
      if curl --fail --location --retry 3 --connect-timeout 20 --output "$output.part" "$url"; then
        mv -f "$output.part" "$output"; return 0
      fi
    elif command -v wget >/dev/null 2>&1; then
      if wget -O "$output.part" "$url"; then
        mv -f "$output.part" "$output"; return 0
      fi
    else
      echo "curl or wget is required to download the fallback Python runtime." >&2
      return 1
    fi
    rm -f "$output.part"
  done
  return 1
}

install_portable_python() {
  local arch triple archive archive_path sums_path mirror official expected actual
  arch="$(uname -m)"
  case "$arch" in
    x86_64|amd64) triple="x86_64-unknown-linux-gnu" ;;
    aarch64|arm64) triple="aarch64-unknown-linux-gnu" ;;
    *) echo "Unsupported Linux architecture: $arch" >&2; return 2 ;;
  esac
  archive="cpython-${PYTHON_VERSION}+${PYTHON_RELEASE}-${triple}-install_only_stripped.tar.gz"
  archive_path="$DOWNLOAD_DIR/$archive"
  sums_path="$DOWNLOAD_DIR/SHA256SUMS-$PYTHON_RELEASE"
  mirror="https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone/$PYTHON_RELEASE"
  official="https://github.com/astral-sh/python-build-standalone/releases/download/$PYTHON_RELEASE"

  [[ -s "$archive_path" ]] || fetch_file "$archive_path" "$mirror/$archive" "$official/$archive"
  [[ -s "$sums_path" ]] || fetch_file "$sums_path" "$mirror/SHA256SUMS" "$official/SHA256SUMS"
  expected="$(awk -v filename="$archive" '$2 == filename || $2 == "*" filename {print $1; exit}' "$sums_path")"
  actual="$(sha256sum "$archive_path" | awk '{print $1}')"
  if [[ -z "$expected" || "$expected" != "$actual" ]]; then
    rm -f "$archive_path" "$sums_path"
    echo "Standalone Python SHA-256 verification failed; downloaded files were removed." >&2
    return 1
  fi

  rm -rf "$RUNTIME_DIR.new"
  mkdir -p "$RUNTIME_DIR.new"
  tar -xzf "$archive_path" -C "$RUNTIME_DIR.new"
  [[ -x "$RUNTIME_DIR.new/python/bin/python3" ]] || { echo "Standalone Python extraction failed." >&2; return 1; }
  rm -rf "$RUNTIME_DIR"
  mv "$RUNTIME_DIR.new" "$RUNTIME_DIR"
}

BASE_PY="$(find_system_python || true)"
if [[ -z "$BASE_PY" ]]; then
  if [[ ! -x "$RUNTIME_DIR/python/bin/python3" ]]; then
    echo "No compatible Python found; downloading a verified standalone runtime."
    install_portable_python
  fi
  BASE_PY="$RUNTIME_DIR/python/bin/python3"
fi

VENV_PY="$VENV_DIR/bin/python"
if ! supported_python "$VENV_PY"; then
  rm -rf "$VENV_DIR"
  "$BASE_PY" -m venv "$VENV_DIR"
fi

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1
cd "$ROOT"
ARGS=("$ROOT/bootstrap.py" --install-main-deps)
if [[ "${MHD_PREPARE_ONLY:-0}" != "1" ]]; then
  ARGS+=(--launch)
fi
"$VENV_PY" "${ARGS[@]}"
