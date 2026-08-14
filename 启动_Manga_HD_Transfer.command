#!/bin/zsh
set -e
cd "$(dirname "$0")"
printf '\033]0;Manga HD Transfer Studio\007'

PY=""
for CAND in python3.13 python3.12 python3.11 python3; do
  if command -v "$CAND" >/dev/null 2>&1; then
    if "$CAND" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3,11) else 1)
PY
    then PY="$CAND"; break; fi
  fi
done

if [[ -z "$PY" ]]; then
  echo "需要 Python 3.11 或更高版本。建议安装 python.org 官方 Python 3.12/3.13。"
  read "?按回车退出…"
  exit 1
fi


if [[ ! -d .venv ]]; then
  echo "首次启动：创建本地虚拟环境（不会下载 OCR / LightGlue / LoFTR / LaMa / MangaLens / SAM 2 模型）…"
  "$PY" -m venv .venv
fi
source .venv/bin/activate

if ! python - <<'PY' >/dev/null 2>&1
import cv2, numpy, PIL, scipy, pydantic, typer, PySide6
PY
then
  echo "正在安装核心运行依赖。这里只安装 OpenCV / Pillow / SciPy / PySide6 等核心与界面包，不安装或下载 OCR 模型。"
  python -m pip install --upgrade pip
  python -m pip install -e ".[gui]"
fi

python run_gui.py
