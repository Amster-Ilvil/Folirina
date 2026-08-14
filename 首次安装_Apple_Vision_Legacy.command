#!/bin/zsh
set -e
cd "$(dirname "$0")"
if [[ ! -d .venv ]]; then
  echo "请先运行：启动_Manga_HD_Transfer.command"
  read "?按回车退出…"; exit 1
fi
source .venv/bin/activate
if [[ "$(uname -s)" != "Darwin" ]]; then echo "Apple Vision Legacy 仅支持 macOS。"; exit 1; fi
echo "仅安装旧 PyObjC VNRecognizeTextRequest 兼容路线。正常使用不需要安装。"
read "?确认安装？按回车继续，Ctrl+C 取消…"
python -m pip install -U 'pyobjc-core>=12.2.1' 'pyobjc-framework-Cocoa>=12.2.1' 'pyobjc-framework-Vision>=12.2.1' 'pyobjc-framework-Quartz>=12.2.1'
python -m pip install -e '.[apple_legacy]'
echo "安装完成。GUI 中请选择 Apple Vision Legacy。"
read "?按回车退出…"
