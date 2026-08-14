#!/bin/zsh
set -e
cd "$(dirname "$0")"
if [[ "$(uname -s)" != "Darwin" ]]; then echo "此脚本仅用于 macOS。"; read "?按回车退出…"; exit 1; fi
if [[ ! -d .venv ]]; then echo "请先运行 启动_Manga_HD_Transfer.command 创建环境。"; read "?按回车退出…"; exit 1; fi
source .venv/bin/activate
ARCH="$(uname -m)"
echo "准备 Apple MPS 可选运行环境（$ARCH）。不会下载任何漫画模型权重。"
python -m pip install -U pip
python -m pip install 'torch>=2.2' 'torchvision>=0.17' 'kornia>=0.8' 'spandrel>=0.4' 'ultralytics>=8.3'
echo "安装 LightGlue 代码（不含模型权重）…"
python -m pip install 'git+https://github.com/cvg/LightGlue.git'
python - <<'PY'
import json, torch
print(json.dumps({
  "torch": torch.__version__,
  "mps_built": bool(torch.backends.mps.is_built()),
  "mps_available": bool(torch.backends.mps.is_available()),
}, ensure_ascii=False, indent=2))
PY
echo "完成。模型仍按需由你显式选择/准备。"
read "?按回车退出…"
