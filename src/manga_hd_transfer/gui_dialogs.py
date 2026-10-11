from __future__ import annotations

"""Small application-owned confirmation helpers.

The module deliberately imports Qt lazily so headless release/audit environments
can import the whole non-GUI package surface without PySide6 installed.  The
actual dialog is only constructed when the desktop GUI calls ``confirm_action``.
"""


def confirm_action(
    parent, title: str, message: str, *,
    confirm_text: str = "确认", cancel_text: str = "取消", destructive: bool = True,
) -> bool:
    # Native QMessageBox.question sheets can render as an empty translucent
    # panel on macOS when Folirina lives inside the responsive graphics proxy.
    # Keep the confirmation in the normal Qt widget hierarchy.
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

    dialog = QDialog(parent)
    dialog.setWindowTitle(str(title))
    dialog.setModal(True)
    dialog.setObjectName("confirmDialog")
    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(24, 22, 24, 20)
    layout.setSpacing(18)
    head = QHBoxLayout(); head.setSpacing(16)
    icon = QLabel("!" if destructive else "?")
    icon.setObjectName("confirmIconDanger" if destructive else "confirmIconInfo")
    icon.setFixedSize(42, 42); icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
    head.addWidget(icon, 0, Qt.AlignmentFlag.AlignTop)
    text = QVBoxLayout(); text.setSpacing(6)
    heading = QLabel(str(title)); heading.setObjectName("confirmTitle"); text.addWidget(heading)
    body = QLabel(str(message))
    body.setWordWrap(True)
    body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    body.setMinimumWidth(360)
    text.addWidget(body)
    head.addLayout(text, 1)
    layout.addLayout(head)
    buttons = QHBoxLayout()
    buttons.addStretch(1)
    cancel = QPushButton(str(cancel_text))
    confirm = QPushButton(str(confirm_text))
    confirm.setObjectName("danger" if destructive else "primary")
    cancel.clicked.connect(dialog.reject)
    confirm.clicked.connect(dialog.accept)
    buttons.addWidget(cancel)
    buttons.addWidget(confirm)
    layout.addLayout(buttons)
    # 破坏性操作默认停在“取消”上，防止回车误删；普通确认则回车即确认
    (cancel if destructive else confirm).setDefault(True)
    dialog.setMinimumWidth(460)
    dialog.adjustSize()
    return dialog.exec() == QDialog.DialogCode.Accepted


__all__ = ["confirm_action"]
