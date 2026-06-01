"""Main window for the PySide6 desktop UI."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QHBoxLayout, QMainWindow, QStackedWidget, QWidget

from character import CharacterRepository
from ui.navigation import NAVIGATION_ITEMS, NavigationPanel
from ui.screens import (
    BattleScreen,
    CharacterBuilderScreen,
    CharacterListScreen,
    CustomBattleScreen,
    HomeScreen,
    PlaceholderScreen,
    RandomBattleScreen,
    ReplayListScreen,
    ReplayViewerScreen,
    SettingsScreen,
)
from ui.services import BattleSetupResult, BattleSetupService, ModelService


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

    def __init__(
        self,
        character_repository: CharacterRepository | None = None,
        model_service: ModelService | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("D&D Battle AI")
        self.resize(1180, 760)
        self.setMinimumSize(900, 560)

        self._model_service = model_service or ModelService()
        self._character_repository = character_repository or CharacterRepository(
            self._model_service.settings.character_dir
        )
        self._battle_setup_service = BattleSetupService(
            self._character_repository,
            map_dir=self._model_service.settings.map_dir,
        )
        self._navigation = NavigationPanel(NAVIGATION_ITEMS)
        self._navigation.setFixedWidth(230)
        self._stack = QStackedWidget()
        self._screen_indexes: dict[str, int] = {}
        self._character_builder_screen: CharacterBuilderScreen | None = None
        self._battle_screen: BattleScreen | None = None
        self._battle_screen_index: int | None = None
        self._replay_list_screen: ReplayListScreen | None = None
        self._replay_viewer_screen: ReplayViewerScreen | None = None
        self._replay_viewer_screen_index: int | None = None
        self._selected_battle_status = "Бой: не выбран"
        self._model_status_label = QLabel()
        self._fallback_status_label = QLabel()
        self._battle_status_label = QLabel()

        self._build_layout()
        self._build_status_bar()
        self._build_screens()
        self._navigation.screen_selected.connect(self.set_screen)
        self._navigation.select("home")

    def set_screen(self, key: str) -> None:
        """Switch to a registered screen by key."""

        index = self._screen_indexes.get(key)
        if index is None:
            return
        self._stack.setCurrentIndex(index)
        current_widget = self._stack.widget(index)
        if key in {"random_battle", "custom_battle"}:
            self._battle_setup_service.map_dir = Path(self._model_service.settings.map_dir)
        if key == "replays" and self._replay_list_screen is not None:
            self._replay_list_screen.replay_dir = Path(self._model_service.settings.replay_dir)
        refresh = getattr(current_widget, "refresh", None)
        if callable(refresh):
            refresh()
        self._update_status_bar()

    def _build_layout(self) -> None:
        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._navigation)
        layout.addWidget(self._stack, stretch=1)
        self.setCentralWidget(central)

    def _build_status_bar(self) -> None:
        for label in (
            self._model_status_label,
            self._fallback_status_label,
            self._battle_status_label,
        ):
            label.setObjectName("statusChip")
            self.statusBar().addPermanentWidget(label)
        self._update_status_bar()

    def _build_screens(self) -> None:
        self._add_screen("home", HomeScreen())
        characters_screen = CharacterListScreen(self._character_repository)
        characters_screen.create_requested.connect(self._open_character_create)
        characters_screen.edit_requested.connect(self._open_character_editor)
        self._add_screen("characters", characters_screen)
        self._character_builder_screen = CharacterBuilderScreen(
            self._character_repository
        )
        self._character_builder_screen.saved.connect(self._character_saved)
        self._character_builder_screen.cancelled.connect(
            lambda: self._navigation.select("characters")
        )
        self._add_screen("character_create", self._character_builder_screen)
        random_battle_screen = RandomBattleScreen(
            self._battle_setup_service,
            self._model_service,
        )
        random_battle_screen.battle_started.connect(self._open_battle_screen)
        self._add_screen("random_battle", random_battle_screen)
        self._add_screen("custom_battle", CustomBattleScreen(self._battle_setup_service))
        self._battle_screen = BattleScreen(self._model_service)
        self._battle_screen_index = self._stack.addWidget(self._battle_screen)
        self._replay_list_screen = ReplayListScreen(self._model_service.settings.replay_dir)
        self._replay_list_screen.open_requested.connect(self._open_replay_viewer)
        self._add_screen("replays", self._replay_list_screen)
        self._replay_viewer_screen = ReplayViewerScreen()
        self._replay_viewer_screen.back_requested.connect(
            lambda: self._navigation.select("replays")
        )
        self._replay_viewer_screen_index = self._stack.addWidget(self._replay_viewer_screen)
        settings_screen = SettingsScreen(self._model_service)
        settings_screen.settings_changed.connect(self._update_status_bar)
        self._add_screen("settings", settings_screen)
        for item in NAVIGATION_ITEMS:
            if item.key in {
                "home",
                "characters",
                "character_create",
                "custom_battle",
                "random_battle",
                "replays",
                "settings",
            }:
                continue
            title, description = SCREEN_DEFINITIONS[item.key]
            self._add_screen(item.key, PlaceholderScreen(title, description))
        self._stack.setCurrentIndex(self._screen_indexes["home"])
        self._stack.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

    def _add_screen(self, key: str, widget: QWidget) -> None:
        self._screen_indexes[key] = self._stack.addWidget(widget)

    def _open_character_create(self) -> None:
        if self._character_builder_screen is not None:
            self._character_builder_screen.new_character()
        self._navigation.select("character_create")

    def _open_character_editor(self, character_id: str) -> None:
        if self._character_builder_screen is not None:
            self._character_builder_screen.load_character(character_id)
        self._navigation.select("character_create")

    def _character_saved(self) -> None:
        self._navigation.select("characters")

    def _open_battle_screen(self, setup_result: BattleSetupResult) -> None:
        if self._battle_screen is None or self._battle_screen_index is None:
            return
        self._battle_screen.set_battle(setup_result)
        self._selected_battle_status = (
            f"Бой: {setup_result.map_name} | {setup_result.difficulty}"
        )
        self._update_status_bar()
        self._stack.setCurrentIndex(self._battle_screen_index)

    def _open_replay_viewer(self, path: object) -> None:
        if self._replay_viewer_screen is None or self._replay_viewer_screen_index is None:
            return
        self._replay_viewer_screen.load_replay(Path(str(path)))
        self._stack.setCurrentIndex(self._replay_viewer_screen_index)
        self._update_status_bar()

    def _update_status_bar(self) -> None:
        settings = self._model_service.settings
        model_text = (
            self._model_service.get_policy_name()
            if self._model_service.is_model_loaded()
            else f"{settings.model_type.upper()}: checkpoint не загружен"
        )
        fallback_text = self._model_service.fallback_agent_label(settings.fallback_agent)
        self._model_status_label.setText(f"Модель: {model_text}")
        self._fallback_status_label.setText(f"Fallback: {fallback_text}")
        self._battle_status_label.setText(self._selected_battle_status)
