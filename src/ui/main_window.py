"""Main window for the PySide6 desktop UI."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QMainWindow, QStackedWidget, QWidget

from ui.navigation import NAVIGATION_ITEMS, NavigationPanel
from ui.screens import HomeScreen, PlaceholderScreen


SCREEN_DEFINITIONS: dict[str, tuple[str, str]] = {
    "characters": (
        "Персонажи",
        "Список созданных и импортированных персонажей.",
    ),
    "character_create": (
        "Создать персонажа",
        "Пошаговая сборка персонажа по поддержанному ruleset.",
    ),
    "random_battle": (
        "Случайный бой",
        "Быстрый запуск encounter через существующий EncounterGenerator.",
    ),
    "custom_battle": (
        "Кастомный бой",
        "Настройка состава сторон, карты и стартовых условий боя.",
    ),
    "replays": (
        "Реплеи",
        "Просмотр сохранённых BattleReplay JSON и истории шагов боя.",
    ),
    "settings": (
        "Настройки",
        "Параметры интерфейса, путей к чекпоинтам и правил проекта.",
    ),
}


class MainWindow(QMainWindow):
    """Top-level application window with navigation and screen stack."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("D&D Battle AI")
        self.resize(1180, 760)
        self.setMinimumSize(900, 560)

        self._navigation = NavigationPanel(NAVIGATION_ITEMS)
        self._navigation.setFixedWidth(230)
        self._stack = QStackedWidget()
        self._screen_indexes: dict[str, int] = {}

        self._build_layout()
        self._build_screens()
        self._navigation.screen_selected.connect(self.set_screen)
        self._navigation.select("home")

    def set_screen(self, key: str) -> None:
        """Switch to a registered screen by key."""

        index = self._screen_indexes.get(key)
        if index is None:
            return
        self._stack.setCurrentIndex(index)

    def _build_layout(self) -> None:
        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._navigation)
        layout.addWidget(self._stack, stretch=1)
        self.setCentralWidget(central)

    def _build_screens(self) -> None:
        self._add_screen("home", HomeScreen())
        for item in NAVIGATION_ITEMS:
            if item.key == "home":
                continue
            title, description = SCREEN_DEFINITIONS[item.key]
            self._add_screen(item.key, PlaceholderScreen(title, description))
        self._stack.setCurrentIndex(self._screen_indexes["home"])
        self._stack.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

    def _add_screen(self, key: str, widget: QWidget) -> None:
        self._screen_indexes[key] = self._stack.addWidget(widget)
