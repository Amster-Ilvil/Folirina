#!/bin/zsh
set -e
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo "请先运行：启动_Manga_HD_Transfer.command"
  read "?按回车退出…"
  exit 1
fi
source .venv/bin/activate
echo "将安装 Ultralytics 运行依赖，用于 MangaLens / YOLO11 气泡实例分割。"
echo "不会自动下载 MangaLens 模型权重；请自行准备有权使用的本地 .pt 模型，并在配置中设置 bubbles.mangalens_model_path。"
python -m pip install 'ultralytics>=8.3'
echo "安装完成。"
read "?按回车关闭…"
