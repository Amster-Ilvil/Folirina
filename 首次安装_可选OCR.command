#!/bin/zsh
set -e
cd "$(dirname "$0")"
if [[ ! -d .venv ]]; then
  echo "请先运行：启动_Manga_HD_Transfer.command"
  read "?按回车退出…"; exit 1
fi
source .venv/bin/activate
echo "将安装 PaddleOCR / PaddlePaddle Python 依赖。OCR 模型权重仍由 PaddleOCR 在首次真正使用时按其机制获取。"
read "?确认安装？按回车继续，Ctrl+C 取消…"
python -m pip install -e '.[ocr]'
echo "安装完成。"
read "?按回车退出…"
