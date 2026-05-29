"""PySide6 application bootstrap."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow
from ui.theme import apply_theme


def create_app(argv: Sequence[str] | None = None) -> QApplication:
    """Create or reuse the QApplication instance."""

    app = QApplication.instance()
    if app is None:
        app = QApplication(list(argv) if argv is not None else sys.argv)
    apply_theme(app)
    return app


def run_app(argv: Sequence[str] | None = None) -> int:
    """Run the desktop UI event loop."""

    app = create_app(argv)
    window = MainWindow()
    window.show()
    return app.exec()
