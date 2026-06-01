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
            background: #f2f5f7;
        }

        QWidget {
            color: #1f2933;
            font-size: 10pt;
        }

        QStatusBar {
            background: #ffffff;
            border-top: 1px solid #d8e0e8;
            color: #52616f;
        }

        QLabel#statusChip {
            background: #eef3f6;
            border: 1px solid #d8e0e8;
            border-radius: 4px;
            padding: 3px 8px;
            color: #253549;
            font-weight: 600;
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

        QGroupBox {
            border: 1px solid #d8e0e8;
            border-radius: 6px;
            margin-top: 10px;
            padding: 12px 10px 10px 10px;
            font-weight: 700;
            color: #253549;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
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

        QLineEdit,
        QComboBox,
        QSpinBox,
        QTextEdit,
        QListWidget,
        QTableWidget {
            background: #ffffff;
            border: 1px solid #cbd5df;
            border-radius: 5px;
            padding: 5px;
            selection-background-color: #2e7d72;
            selection-color: #ffffff;
        }

        QLineEdit:focus,
        QComboBox:focus,
        QSpinBox:focus,
        QTextEdit:focus,
        QListWidget:focus,
        QTableWidget:focus {
            border: 1px solid #2e7d72;
        }

        QTabWidget::pane {
            border: 1px solid #d8e0e8;
            border-radius: 6px;
            background: #ffffff;
        }

        QTabBar::tab {
            background: #eef3f6;
            border: 1px solid #d8e0e8;
            padding: 8px 12px;
            margin-right: 2px;
            border-top-left-radius: 5px;
            border-top-right-radius: 5px;
        }

        QTabBar::tab:selected {
            background: #ffffff;
            color: #17202a;
            font-weight: 700;
        }

        QHeaderView::section {
            background: #eef3f6;
            color: #253549;
            border: none;
            border-bottom: 1px solid #d8e0e8;
            padding: 7px;
            font-weight: 700;
        }

        QTableWidget {
            gridline-color: #e1e7ee;
            alternate-background-color: #f8fafc;
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

        QLabel#inlineStatus {
            color: #52616f;
            padding: 6px 0;
        }

        QLabel#warningStatus {
            color: #9f4c4c;
            font-weight: 600;
            padding: 6px 0;
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
