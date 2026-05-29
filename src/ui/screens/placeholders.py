"""Placeholder screens for planned desktop UI sections."""

from __future__ import annotations

from PySide6.QtWidgets import QVBoxLayout, QWidget

from ui.widgets.screen import ScreenFrame


class PlaceholderScreen(QWidget):
    """Simple screen that reserves a UI section without combat duplication."""

    def __init__(
        self,
        title: str,
        description: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(0)

        frame = ScreenFrame(title, description)
        frame.add_body_text(
            "Этот раздел будет подключаться к существующим модулям combat, "
            "rules, agents и replay. Новая боевая логика здесь не дублируется."
        )
        frame.content_layout.addStretch(1)
        layout.addWidget(frame)
