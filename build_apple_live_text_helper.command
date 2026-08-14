#!/bin/zsh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
SOURCE="$ROOT/tools/apple_live_text_helper/AppleLiveTextOCRHelper.swift"
OUTPUT_DIR="$ROOT/tools/apple_live_text_helper/bin"
OUTPUT="$OUTPUT_DIR/apple_live_text_helper"
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Apple Live Text Helper 只能在 macOS 上编译。" >&2
  exit 1
fi
if ! command -v xcrun >/dev/null 2>&1; then
  echo "未找到 xcrun。请先运行 xcode-select --install 安装 Xcode Command Line Tools。" >&2
  exit 1
fi
if [[ ! -f "$SOURCE" ]]; then
  echo "缺少源码：$SOURCE" >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR"
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-13.0}"
xcrun swiftc "$SOURCE" \
  -parse-as-library \
  -O \
  -framework Foundation \
  -framework VisionKit \
  -framework ImageIO \
  -framework AppKit \
  -o "$OUTPUT"
chmod +x "$OUTPUT"
echo "Apple Live Text Helper 已生成：$OUTPUT"
