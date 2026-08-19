# Folirina

Folirina 是一个用于漫画页面文字迁移、清理、重排和人工复核的桌面工具。

## 安装

```bash
python -m pip install -e .
```

## 启动

```bash
python run_gui.py
```

也可以使用命令行：

```bash
mhd-transfer --help
```

## 支持模式

- `auto`
- `direct_patch`
- `mask_replace`
- `aligned_overlay_reveal`
- `transparent_bubble_reveal`
- `hybrid`
- `reletter`

程序运行时生成的模型、缓存、日志和项目结果保存在本地工作区，不属于代码仓库内容。
