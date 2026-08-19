from __future__ import annotations

"""Publication/export page isolated from the StudioWindow shell."""

from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QCheckBox, QMessageBox

from .gui_components import Card, PathRow
from .workspace_cleanup import cleanup_output_workspace

class ExportPage(QWidget):
    def __init__(self, window: "StudioWindow"):
        super().__init__(); self.window=window
        root=QVBoxLayout(self); root.setContentsMargins(16,16,16,16); root.setSpacing(12)
        content=QHBoxLayout(); content.setSpacing(12); root.addLayout(content,1)
        left=Card("输出选项", "选择输出目录和需要保留的诊断/编辑文件。")
        self.out=PathRow("输出目录","选择目录"); left.layout.addWidget(self.out)
        compact_hint=QLabel("默认使用精简工作区：保留恢复/人工补漏必需文件；Debug、逐气泡 Mask、ORA/PSD 改为按需输出。")
        compact_hint.setObjectName("hint"); compact_hint.setWordWrap(True); left.layout.addWidget(compact_hint)
        self.debug=QCheckBox("保存配准 / 结构 / 蒙版 Debug 图（占空间）")
        self.component_masks=QCheckBox("保存逐气泡 / 逐文本单元 Mask（诊断用）")
        self.layers=QCheckBox("输出 OpenRaster / PSD 可编辑图层（占空间）")
        left.layout.addWidget(self.debug); left.layout.addWidget(self.component_masks); left.layout.addWidget(self.layers)
        self.cleanup=QPushButton("清理 pages 冗余诊断文件…"); self.cleanup.setObjectName("softPrimary")
        self.cleanup.setToolTip("删除可重新生成的 Debug 图、自动 editable.ora/.psd、inpainted.png、逐气泡/逐单元 Mask；不会删除 final、原图、中文图层、Review 或人工补漏文件。")
        left.layout.addWidget(self.cleanup)
        self.run=QPushButton("开始整册处理"); self.run.setObjectName("primary"); left.layout.addWidget(self.run); left.layout.addStretch(1)
        content.addWidget(left,4)
        right=Card("输出结构", "默认只保留正式结果与继续编辑所需文件。")
        for title,desc in [
            ("final/","最终高清页面"),("pages/*/project.json","配准、气泡、匹配与 QA 工程数据"),("mask_transfer_layer.png","旧中文版气泡/文本框迁移图层"),("mask_transfer_mask.png","实际覆盖高清母版的精确蒙版"),("editable.ora / .psd","可选；默认关闭以节省空间")]:
            row=QHBoxLayout(); t=QLabel(title); t.setStyleSheet("font-weight:600;"); d=QLabel(desc); d.setObjectName("hint"); row.addWidget(t); row.addStretch(1); row.addWidget(d); right.layout.addLayout(row)
        right.layout.addStretch(1); content.addWidget(right,6)
        self.out.button.clicked.connect(lambda:self.window.choose_directory("output")); self.run.clicked.connect(self.window.run_book)
        self.cleanup.clicked.connect(self._cleanup_pages)
        self.debug.toggled.connect(self._sync); self.component_masks.toggled.connect(self._sync); self.layers.toggled.connect(self._sync)
    def _sync(self):
        self.window.state.config.export.save_debug=self.debug.isChecked()
        self.window.state.config.export.save_component_masks=self.component_masks.isChecked()
        self.window.state.config.export.layer_bundle=self.layers.isChecked()
    def _cleanup_pages(self):
        out=str(self.window.state.output_dir or "").strip()
        if not out:
            QMessageBox.information(self,"没有输出目录","请先选择已有输出目录。")
            return
        answer=QMessageBox.question(self,"清理冗余文件","将删除 pages 中可重新生成的 Debug、自动 ORA/PSD、inpainted 和逐组件 Mask。不会删除 final、原始页、Review/人工补漏结果。继续吗？")
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            stats=cleanup_output_workspace(out)
            freed=float(stats.get("bytes_freed",0))/(1024**3)
            QMessageBox.information(self,"清理完成",f"扫描 {stats.get('pages_scanned',0)} 页，删除 {stats.get('files_removed',0)} 个文件，释放约 {freed:.2f} GB。")
        except Exception as exc:
            QMessageBox.critical(self,"清理失败",str(exc))
    def refresh(self):
        self.out.set_path(self.window.state.output_dir)
        self.debug.setChecked(self.window.state.config.export.save_debug)
        self.component_masks.setChecked(bool(getattr(self.window.state.config.export,"save_component_masks",False)))
        self.layers.setChecked(self.window.state.config.export.layer_bundle)

__all__ = ["ExportPage"]
