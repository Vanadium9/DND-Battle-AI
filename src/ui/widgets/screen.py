"""Reusable screen layout helpers."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class ScreenFrame(QWidget):
    """Base white content surface with a title and optional subtitle."""

    def __init__(
        self,
        title: str,
        subtitle: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("screenSurface")
        self.content_layout = QVBoxLayout(self)
        self.content_layout.setContentsMargins(28, 26, 28, 26)
        self.content_layout.setSpacing(14)

        title_label = QLabel(title)
        title_label.setObjectName("screenTitle")
        title_label.setWordWrap(True)
        self.content_layout.addWidget(title_label)

        if subtitle:
            subtitle_label = QLabel(subtitle)
            subtitle_label.setObjectName("screenSubtitle")
            subtitle_label.setWordWrap(True)
            self.content_layout.addWidget(subtitle_label)

    def add_body_text(self, text: str) -> QLabel:
        """Add a body label and return it for further customization."""

        label = QLabel(text)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.content_layout.addWidget(label)
        return label
