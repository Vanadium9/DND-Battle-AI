import os

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from ui.app import create_app
from ui.main_window import MainWindow
from ui.navigation import NAVIGATION_ITEMS


def test_main_window_registers_navigation_screens() -> None:
    app = create_app(["test_gui_smoke"])
    window = MainWindow()

    assert window.windowTitle() == "D&D Battle AI"
    assert set(window._screen_indexes) == {item.key for item in NAVIGATION_ITEMS}

    window.set_screen("replays")
    assert window._stack.currentIndex() == window._screen_indexes["replays"]

    window.close()
    app.processEvents()
