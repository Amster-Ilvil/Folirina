from __future__ import annotations

"""Pre-run publication settings, separate from renderer and review controls."""

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QCheckBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QVBoxLayout, QWidget,
)

from .gui_components import StableComboBox, StableSpinBox
from .publication_runtime import find_caesium, find_upscayl


class DeliveryOptions(QWidget):
    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.window = window
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self._save)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.compress = QCheckBox("处理完成后用 Caesium 压缩最终图片")
        root.addWidget(self.compress)
        form = QFormLayout()
        self.format = StableComboBox()
        for label, value in (("WebP（小体积）", "webp"), ("JPEG（兼容阅读器）", "jpeg"), ("PNG（无损）", "png")):
            self.format.addItem(label, value)
        form.addRow("导出格式", self.format)
        self.quality = StableSpinBox(); self.quality.setRange(1, 100)
        form.addRow("质量 1–100", self.quality)
        self.lossless = QCheckBox("WebP 无损（忽略质量值）")
        form.addRow("", self.lossless)
        root.addLayout(form)
        self.upscale = QCheckBox("处理完成后用 Upscayl AI 超分")
        root.addWidget(self.upscale)
        up = QFormLayout()
        self.scale = StableComboBox()
        self.scale.addItem("2×（像素量约 4 倍）", 2); self.scale.addItem("4×（像素量约 16 倍）", 4)
        up.addRow("导出倍率", self.scale)
        self.model = StableComboBox()
        for label, value in (("Digital Art · 漫画 / 线条", "digital-art-4x"), ("Upscayl Standard", "upscayl-standard-4x"),
                             ("Upscayl Lite", "upscayl-lite-4x"), ("High Fidelity", "high-fidelity-4x"),
                             ("UltraSharp", "ultrasharp-4x"), ("Remacri", "remacri-4x"), ("Ultramix Balanced", "ultramix-balanced-4x")):
            self.model.addItem(label, value)
        up.addRow("超分模型", self.model)
        self.tile = StableSpinBox(); self.tile.setRange(32, 1024); self.tile.setSingleStep(32)
        up.addRow("分块边长（省内存）", self.tile)
        root.addLayout(up)
        hint = QLabel("顺序：完成文字替换 → 可选 AI 超分 → 可选压缩 → final/。\n编辑底稿保留原分辨率无损 PNG；复核保存会自动更新导出。超分会增加体积与耗时。Mac 上 Upscayl 使用 NCNN / MoltenVK 调用 Metal GPU。")
        hint.setWordWrap(True); hint.setObjectName("hint"); root.addWidget(hint)

        advanced = QGroupBox("工具路径（留空自动探测）")
        layout = QFormLayout(advanced)
        self.caesium = QLineEdit(); self.upscayl = QLineEdit(); self.models = QLineEdit()
        for label, edit, folder in (("Caesium CLT", self.caesium, False), ("Upscayl 引擎", self.upscayl, False), ("Upscayl 模型目录", self.models, True)):
            row = QHBoxLayout(); row.addWidget(edit, 1)
            button = QPushButton("选择…"); row.addWidget(button)
            button.clicked.connect(lambda _checked=False, field=edit, directory=folder: self._choose(field, directory))
            layout.addRow(label, row)
        root.addWidget(advanced)
        self.runtime_status = QLabel(); self.runtime_status.setWordWrap(True); self.runtime_status.setObjectName("quiet")
        root.addWidget(self.runtime_status)
        self._controls = [self.compress, self.format, self.quality, self.lossless, self.upscale, self.scale, self.model, self.tile, self.caesium, self.upscayl, self.models]
        for control in (self.compress, self.lossless, self.upscale):
            control.toggled.connect(self._sync)
        for control in (self.format, self.scale, self.model):
            control.currentIndexChanged.connect(self._sync)
        for control in (self.quality, self.tile):
            control.valueChanged.connect(self._sync)
        for control in (self.caesium, self.upscayl, self.models):
            control.editingFinished.connect(self._sync)
        self.refresh()

    def _choose(self, field, directory):
        value = QFileDialog.getExistingDirectory(self, "选择模型目录", field.text()) if directory else QFileDialog.getOpenFileName(self, "选择命令行程序", field.text())[0]
        if value:
            field.setText(value); self._sync()

    def _save(self):
        save = getattr(self.window, "_save_workspace_session", None)
        if callable(save):
            save()

    def _sync(self, *_):
        cfg = self.window.state.config.export.delivery
        cfg.compression_enabled = self.compress.isChecked()
        cfg.format = self.format.currentData()
        cfg.quality = self.quality.value()
        cfg.lossless = self.lossless.isChecked()
        cfg.upscale_enabled = self.upscale.isChecked()
        cfg.upscale_scale = self.scale.currentData()
        cfg.upscale_model = self.model.currentData()
        cfg.tile_size = self.tile.value()
        cfg.caesium_path = self.caesium.text().strip()
        cfg.upscayl_path = self.upscayl.text().strip()
        cfg.upscayl_models = self.models.text().strip()
        self._enabled_fields()
        self._status()
        self._save_timer.start(400)

    def _enabled_fields(self):
        compress = self.compress.isChecked()
        self.format.setEnabled(compress)
        self.lossless.setEnabled(compress and self.format.currentData() == "webp")
        self.quality.setEnabled(compress and self.format.currentData() != "png" and not (self.format.currentData() == "webp" and self.lossless.isChecked()))
        for control in (self.scale, self.model, self.tile):
            control.setEnabled(self.upscale.isChecked())

    def _status(self):
        caesium = find_caesium(self.caesium.text().strip())
        upscayl, models = find_upscayl(self.upscayl.text().strip(), self.models.text().strip())
        model = str(self.model.currentData() or "")
        ready = upscayl and models and (models / (model + ".bin")).is_file() and (models / (model + ".param")).is_file()
        self.runtime_status.setText(
            ("Caesium 已就绪" if caesium else "Caesium 未安装：运行 scripts/install_output_tools.command")
            + "\n" + ("Upscayl 及所选模型已就绪" if ready else "Upscayl 或所选模型未就绪，请指定已有程序和模型目录")
        )

    def refresh(self):
        cfg = self.window.state.config.export.delivery
        for control in self._controls:
            control.blockSignals(True)
        try:
            self.compress.setChecked(cfg.compression_enabled)
            self.format.setCurrentIndex(self.format.findData(cfg.format))
            self.quality.setValue(cfg.quality); self.lossless.setChecked(cfg.lossless)
            self.upscale.setChecked(cfg.upscale_enabled)
            self.scale.setCurrentIndex(self.scale.findData(cfg.upscale_scale))
            index = self.model.findData(cfg.upscale_model)
            if index < 0:
                self.model.addItem(cfg.upscale_model, cfg.upscale_model); index = self.model.count() - 1
            self.model.setCurrentIndex(index); self.tile.setValue(cfg.tile_size)
            self.caesium.setText(cfg.caesium_path); self.upscayl.setText(cfg.upscayl_path); self.models.setText(cfg.upscayl_models)
        finally:
            for control in self._controls:
                control.blockSignals(False)
        self._enabled_fields(); self._status()
