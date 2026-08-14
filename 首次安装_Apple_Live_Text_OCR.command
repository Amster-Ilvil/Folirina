#!/bin/zsh
set -e
cd "$(dirname "$0")"
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Apple Live Text OCR 仅支持 macOS。"
  read "?按回车退出…"; exit 1
fi

echo "Manga HD Transfer v0.8.11 Apple OCR"
echo "默认路线已改为与 Novel Formatter 相同的系统 OCR 家族："
echo "  1) Swift VisionKit ImageAnalyzer / Live Text"
echo "  2) 失败时自动回退 macOS ExtractText 快捷指令"
echo

if ! command -v xcrun >/dev/null 2>&1; then
  echo "⚠️ 未找到 xcrun。VisionKit Helper 暂时不能编译。"
  echo "   可执行：xcode-select --install"
  echo "   如果已经有 ExtractText 快捷指令，程序仍可直接走快捷指令 OCR。"
else
  echo "正在编译 Swift VisionKit Live Text Helper…"
  ./build_apple_live_text_helper.command || {
    echo "⚠️ Swift Helper 编译失败；程序仍会尝试 ExtractText 快捷指令。"
  }
fi

echo
if command -v shortcuts >/dev/null 2>&1; then
  if shortcuts list 2>/dev/null | grep -Fxq "ExtractText"; then
    echo "✅ 已检测到 ExtractText 快捷指令。"
  else
    echo "⚠️ 未检测到名为 ExtractText 的快捷指令。"
    echo "   如果 VisionKit Live Text 在本机不可用，请在‘快捷指令’App 新建："
    echo "   名称：ExtractText"
    echo "   动作：从图像中提取文字（输入设为快捷指令输入）"
  fi
else
  echo "⚠️ 未检测到 shortcuts 命令。"
fi

echo
echo "完成。Apple Live Text OCR 不下载 OCR 模型权重。"
read "?按回车退出…"
