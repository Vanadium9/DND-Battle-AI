"""Random test battle setup screen."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ui.services import BattleSetupRequest, BattleSetupResult, BattleSetupService, ModelService
from ui.widgets.screen import ScreenFrame


class RandomBattleScreen(QWidget):
    """Configure and launch a random CombatEnvironment from GUI options."""

    battle_started = Signal(object)

    def __init__(
        self,
        battle_setup_service: BattleSetupService,
        model_service: ModelService,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._battle_setup_service = battle_setup_service
        self._model_service = model_service
        self._party_preset_combo = QComboBox()
        self._character_list = QListWidget()
        self._difficulty_combo = QComboBox()
        self._map_combo = QComboBox()
        self._enemy_group_combo = QComboBox()
        self._controller_combo = QComboBox()
        self._seed_enabled = QCheckBox("Использовать seed")
        self._seed_spin = QSpinBox()
        self._summary_text = QTextEdit()
        self._build_layout()
        self.refresh()

    def refresh(self) -> None:
        self._populate_characters()
        self._update_summary()

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(0)

        frame = ScreenFrame(
            "Случайный бой",
            "Быстрый запуск тестового encounter через существующий combat engine.",
        )

        content = QHBoxLayout()
        content.addLayout(self._left_column(), stretch=1)
        content.addLayout(self._right_column(), stretch=1)
        frame.content_layout.addLayout(content)

        button_row = QHBoxLayout()
        refresh_button = QPushButton("Обновить описание")
        start_button = QPushButton("Начать бой")
        refresh_button.clicked.connect(self._update_summary)
        start_button.clicked.connect(self._start_battle)
        button_row.addWidget(refresh_button)
        button_row.addWidget(start_button)
        button_row.addStretch(1)
        frame.content_layout.addLayout(button_row)
        layout.addWidget(frame)

    def _left_column(self) -> QVBoxLayout:
        layout = QVBoxLayout()
        party_group = QGroupBox("Party")
        party_layout = QVBoxLayout(party_group)
        self._fill_combo(self._party_preset_combo, self._battle_setup_service.party_presets())
        self._party_preset_combo.currentIndexChanged.connect(self._update_summary)
        party_layout.addWidget(QLabel("Готовый preset party"))
        party_layout.addWidget(self._party_preset_combo)
        party_layout.addWidget(QLabel("Или созданные персонажи"))
        self._character_list.itemChanged.connect(self._update_summary)
        party_layout.addWidget(self._character_list)
        layout.addWidget(party_group)
        return layout

    def _right_column(self) -> QVBoxLayout:
        layout = QVBoxLayout()
        form_group = QGroupBox("Параметры боя")
        form = QFormLayout(form_group)

        self._fill_combo(self._difficulty_combo, self._battle_setup_service.difficulties())
        self._fill_combo(self._map_combo, self._battle_setup_service.maps())
        self._fill_combo(self._enemy_group_combo, self._battle_setup_service.enemy_groups())
        self._fill_combo(self._controller_combo, self._battle_setup_service.controller_modes())
        self._set_combo_data(self._difficulty_combo, "medium")
        self._set_combo_data(self._controller_combo, "manual_players_ai_enemies")

        for combo in (
            self._difficulty_combo,
            self._map_combo,
            self._enemy_group_combo,
            self._controller_combo,
        ):
            combo.currentIndexChanged.connect(self._update_summary)

        self._seed_spin.setRange(0, 2_147_483_647)
        self._seed_spin.setValue(0)
        self._seed_spin.setEnabled(False)
        self._seed_enabled.stateChanged.connect(self._toggle_seed)
        self._seed_spin.valueChanged.connect(self._update_summary)

        seed_row = QHBoxLayout()
        seed_row.addWidget(self._seed_enabled)
        seed_row.addWidget(self._seed_spin)

        form.addRow("Сложность", self._difficulty_combo)
        form.addRow("Карта", self._map_combo)
        form.addRow("Враги", self._enemy_group_combo)
        form.addRow("Управление", self._controller_combo)
        form.addRow("Seed", seed_row)

        layout.addWidget(form_group)
        self._summary_text.setReadOnly(True)
        self._summary_text.setMinimumHeight(180)
        layout.addWidget(QLabel("Краткое описание боя"))
        layout.addWidget(self._summary_text)
        return layout

    def _populate_characters(self) -> None:
        selected_ids = set(self._selected_character_ids())
        self._character_list.blockSignals(True)
        self._character_list.clear()
        for character in self._battle_setup_service.list_saved_characters():
            item = QListWidgetItem(
                f"{character.name} | {character.race_name} {character.class_name} {character.level}"
            )
            item.setData(Qt.ItemDataRole.UserRole, character.id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if character.id in selected_ids
                else Qt.CheckState.Unchecked
            )
            self._character_list.addItem(item)
        self._character_list.blockSignals(False)

    def _start_battle(self) -> None:
        request = self._build_request()
        try:
            result = self._battle_setup_service.create_random_battle(request)
        except ValueError as error:
            QMessageBox.warning(self, "Нельзя начать бой", str(error))
            self._summary_text.setPlainText(str(error))
            return
        if not self._model_service.is_model_loaded():
            QMessageBox.warning(
                self,
                "Модель не загружена",
                "Checkpoint не загружен. Для AI будет использован fallback agent.",
            )
        self.battle_started.emit(result)

    def _update_summary(self) -> None:
        request = self._build_request()
        try:
            summary = self._battle_setup_service.preview_battle(request)
        except ValueError as error:
            summary = str(error)
        self._summary_text.setPlainText(summary)

    def _build_request(self) -> BattleSetupRequest:
        return BattleSetupRequest(
            saved_character_ids=self._selected_character_ids(),
            party_preset=str(self._party_preset_combo.currentData() or "none"),
            difficulty=str(self._difficulty_combo.currentData() or "medium"),
            map_name=str(self._map_combo.currentData() or "open_field"),
            enemy_group=str(self._enemy_group_combo.currentData() or "auto"),
            controller_mode=str(
                self._controller_combo.currentData() or "manual_players_ai_enemies"
            ),
            seed=self._seed_spin.value() if self._seed_enabled.isChecked() else None,
        )

    def _selected_character_ids(self) -> tuple[str, ...]:
        selected = []
        for row in range(self._character_list.count()):
            item = self._character_list.item(row)
            if item.checkState() == Qt.CheckState.Checked:
                selected.append(str(item.data(Qt.ItemDataRole.UserRole)))
        return tuple(selected)

    def _toggle_seed(self) -> None:
        self._seed_spin.setEnabled(self._seed_enabled.isChecked())
        self._update_summary()

    @staticmethod
    def _fill_combo(combo: QComboBox, values: dict[str, str]) -> None:
        combo.clear()
        for key, label in values.items():
            combo.addItem(label, key)

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: str) -> None:
        for index in range(combo.count()):
            if combo.itemData(index) == value:
                combo.setCurrentIndex(index)
                return
