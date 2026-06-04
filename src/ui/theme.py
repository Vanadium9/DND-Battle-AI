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
            color: #2f3f4f;
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
            font-size: 18pt;
            font-weight: 700;
        }

        QLabel#screenSubtitle {
            color: #33475b;
            font-size: 10pt;
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
            border: 1px solid #c6d2dc;
            border-radius: 6px;
            margin-top: 8px;
            padding: 10px 8px 8px 8px;
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
            padding: 6px 12px;
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
            padding: 4px;
            selection-background-color: #2e7d72;
            selection-color: #ffffff;
            min-height: 24px;
        }

        QSpinBox {
            padding-right: 18px;
        }

        QSpinBox::up-button,
        QSpinBox::down-button {
            width: 18px;
            border-left: 1px solid #cbd5df;
            background: #eef3f6;
        }

        QSpinBox::up-button:hover,
        QSpinBox::down-button:hover {
            background: #dce7ed;
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
            color: #33475b;
            font-weight: 600;
        }

        QLabel#characterCardMetaValue {
            color: #1f2933;
        }

        QLabel#characterSheetSummary {
            background: #f7faf9;
            border: 1px solid #cbded9;
            border-radius: 8px;
            padding: 10px;
            color: #1f2933;
            font-size: 10pt;
            font-weight: 600;
            line-height: 150%;
        }

        QLabel#sheetMetricValue {
            color: #17202a;
            font-size: 10.5pt;
            font-weight: 700;
            padding: 1px 4px;
        }

        QLabel#pointBuyStatus {
            background: #eef6f4;
            border: 1px solid #cbded9;
            border-radius: 5px;
            color: #17202a;
            font-weight: 700;
            padding: 5px 7px;
        }

        QLabel#pointBuyStatus[overBudget="true"] {
            background: #fff1f1;
            border: 1px solid #d89b9b;
            color: #8a2f2f;
        }

        QLabel#compactHeader {
            color: #33475b;
            font-size: 9pt;
            font-weight: 700;
        }

        QLabel#emptyStateLabel {
            color: #3f5367;
            padding: 36px;
            font-size: 12pt;
        }

        QLabel#inlineStatus {
            color: #33475b;
            padding: 6px 0;
        }

        QFrame#battleResourcesStrip {
            background: #f7fafc;
            border: 1px solid #d8e0e8;
            border-radius: 6px;
        }

        QLabel#battleResourcesLabel {
            color: #253549;
            padding: 2px;
        }

        QPushButton#compactActionButton {
            background: #eef3f6;
            border: 1px solid #b8c6d2;
            color: #253549;
            padding: 0;
            border-radius: 4px;
            font-size: 11pt;
            font-weight: 800;
        }

        QPushButton#compactActionButton:hover {
            background: #dce7ed;
            border: 1px solid #2e7d72;
        }

        QPushButton#compactActionButton:checked {
            background: #2e7d72;
            border: 2px solid #1f5a53;
            color: #ffffff;
        }

        QPushButton#compactActionButton:disabled {
            background: #f4f7f9;
            border: 1px solid #d8e0e8;
            color: #a7b2bd;
        }

        QGroupBox#abilityColumn {
            background: #f8fafc;
            border: 1px solid #d8e0e8;
            border-radius: 5px;
            margin-top: 8px;
            padding: 7px 4px 4px 4px;
            color: #33475b;
            font-size: 8.5pt;
            font-weight: 700;
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
