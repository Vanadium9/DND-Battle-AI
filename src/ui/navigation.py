"""Navigation widgets for the desktop UI."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget


@dataclass(frozen=True)
class NavigationItem:
    """One screen entry in the main navigation."""

    key: str
    title: str


NAVIGATION_ITEMS: tuple[NavigationItem, ...] = (
    NavigationItem("home", "⌂ Главная"),
    NavigationItem("characters", "👥 Персонажи"),
    NavigationItem("character_create", "+ Создать персонажа"),
    NavigationItem("random_battle", "⚔ Случайный бой"),
    NavigationItem("custom_battle", "⚔ Кастомный бой"),
    NavigationItem("settings", "⚙ Настройки"),
)


class NavigationPanel(QWidget):
    """Side navigation that emits a screen key when selection changes."""

    screen_selected = Signal(str)

    def __init__(
        self,
        items: tuple[NavigationItem, ...] = NAVIGATION_ITEMS,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("navigationPanel")
        self._items = items
        self._list = QListWidget()
        self._list.setObjectName("navigationList")
        self._build_layout()
        self._populate_items()
        self._list.currentRowChanged.connect(self._emit_current_screen)

    def select(self, key: str) -> None:
        """Select a screen by key."""

        for row, item in enumerate(self._items):
            if item.key == key:
                self._list.setCurrentRow(row)
                return

    def _build_layout(self) -> None:
        title = QLabel("D&D Battle AI")
        title.setObjectName("navigationTitle")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 14, 0, 14)
        layout.setSpacing(8)
        layout.addWidget(title)
        layout.addWidget(self._list, stretch=1)

    def _populate_items(self) -> None:
        for item in self._items:
            list_item = QListWidgetItem(item.title)
            list_item.setData(256, item.key)
            self._list.addItem(list_item)

    def _emit_current_screen(self, row: int) -> None:
        if row < 0 or row >= len(self._items):
            return
        self.screen_selected.emit(self._items[row].key)
