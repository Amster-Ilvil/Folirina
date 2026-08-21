#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then
  echo "The local environment is not installed. Run scripts/install_macos.command first."
  exit 1
fi
exec .venv/bin/python -m manga_hd_transfer.launcher
