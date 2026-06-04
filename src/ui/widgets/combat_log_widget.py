"""Combat log panel for battle events."""

from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtWidgets import QGroupBox, QPlainTextEdit, QVBoxLayout

from ui.text import translate_battle_log


class CombatLogWidget(QGroupBox):
    """Read-only combat log with automatic scroll to latest event."""

    def __init__(self, parent=None) -> None:
        super().__init__("Боевой лог", parent)
        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self.setMaximumHeight(190)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 14, 10, 10)
        layout.addWidget(self._text)

    def set_entries(self, entries: Iterable[str]) -> None:
        self._text.setPlainText("\n".join(translate_battle_log(entry) for entry in entries))
        self.scroll_to_bottom()

    def append_entry(self, entry: str) -> None:
        entry = translate_battle_log(entry)
        if self._text.toPlainText():
            self._text.appendPlainText(entry)
        else:
            self._text.setPlainText(entry)
        self.scroll_to_bottom()

    def scroll_to_bottom(self) -> None:
        scrollbar = self._text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def text(self) -> str:
        return self._text.toPlainText()
