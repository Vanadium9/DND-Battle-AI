"""Application theme for the PySide6 desktop UI."""

from __future__ import annotations

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication


def apply_theme(app: QApplication) -> None:
    """Apply a calm, readable application-wide style."""

    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(
        """
        QMainWindow {
            background: #f4f6f8;
        }

        QWidget {
            color: #1f2933;
            font-size: 10pt;
        }

        QLabel#screenTitle {
            color: #17202a;
            font-size: 22pt;
            font-weight: 700;
        }

        QLabel#screenSubtitle {
            color: #52616f;
            font-size: 11pt;
        }

        QWidget#navigationPanel {
            background: #182433;
            border-right: 1px solid #111927;
        }

        QLabel#navigationTitle {
            color: #f7fafc;
            font-size: 14pt;
            font-weight: 700;
            padding: 8px 10px;
        }

        QListWidget#navigationList {
            background: transparent;
            border: none;
            outline: none;
            color: #c9d4df;
        }

        QListWidget#navigationList::item {
            min-height: 36px;
            padding: 8px 12px;
            border-radius: 6px;
            margin: 2px 8px;
        }

        QListWidget#navigationList::item:selected {
            background: #2e7d72;
            color: #ffffff;
        }

        QListWidget#navigationList::item:hover {
            background: #253549;
            color: #ffffff;
        }

        QWidget#screenSurface {
            background: #ffffff;
            border: 1px solid #d8e0e8;
            border-radius: 8px;
        }

        QPushButton {
            background: #2e7d72;
            color: #ffffff;
            border: none;
            border-radius: 6px;
            padding: 8px 14px;
            font-weight: 600;
        }

        QPushButton:hover {
            background: #256b62;
        }

        QPushButton:pressed {
            background: #1f5a53;
        }

        QPushButton:disabled {
            background: #c7d0d9;
            color: #6b7785;
        }

        QPushButton#dangerButton {
            background: #b85c5c;
        }

        QPushButton#dangerButton:hover {
            background: #9f4c4c;
        }

        QFrame#characterCard {
            background: #f9fbfc;
            border: 1px solid #d8e0e8;
            border-radius: 8px;
        }

        QLabel#characterCardTitle {
            color: #17202a;
            font-size: 14pt;
            font-weight: 700;
        }

        QLabel#characterCardMetaLabel {
            color: #52616f;
            font-weight: 600;
        }

        QLabel#characterCardMetaValue {
            color: #1f2933;
        }

        QLabel#emptyStateLabel {
            color: #6b7785;
            padding: 36px;
            font-size: 12pt;
        }

        QPlainTextEdit {
            background: #ffffff;
            border: 1px solid #d8e0e8;
            border-radius: 6px;
            padding: 10px;
            font-family: Consolas, "Courier New", monospace;
        }
        """
    )
