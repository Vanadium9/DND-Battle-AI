"""Random test battle setup screen."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
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

from combat.map_config import MapConfigValidationError
from ui.services import BattleSetupRequest, BattleSetupResult, BattleSetupService, ModelService
from ui.widgets import MapPreviewWidget
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
        self._map_preview = MapPreviewWidget()
        self._summary_text = QTextEdit()
        self._start_button = QPushButton("Начать бой")
        self._build_layout()
        self._apply_settings_seed()
        self.refresh()

    def refresh(self) -> None:
        self._populate_characters()
        self._populate_map_options()
        self._update_map_preview()
        self._update_summary()

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(0)

        frame = ScreenFrame(
            "Случайный бой",
            "Быстрый запуск тестового encounter через существующий combat engine.",
        )

        content = QGridLayout()
        content.setHorizontalSpacing(12)
        content.setVerticalSpacing(10)
        content.addWidget(self._party_group(), 0, 0)
        content.addWidget(self._parameters_group(), 0, 1)
        content.addWidget(self._map_group(), 1, 0)
        content.addWidget(self._summary_group(), 1, 1)
        content.setColumnStretch(0, 1)
        content.setColumnStretch(1, 1)
        content.setRowStretch(0, 0)
        content.setRowStretch(1, 1)
        frame.content_layout.addLayout(content, stretch=1)

        button_row = QHBoxLayout()
        refresh_button = QPushButton("Обновить описание")
        refresh_button.clicked.connect(self._update_summary)
        self._start_button.clicked.connect(self._start_battle)
        button_row.addWidget(refresh_button)
        button_row.addWidget(self._start_button)
        button_row.addStretch(1)
        frame.content_layout.addLayout(button_row)
        layout.addWidget(frame)

    def _party_group(self) -> QGroupBox:
        party_group = QGroupBox("Отряд")
        party_layout = QVBoxLayout(party_group)
        party_layout.setSpacing(6)
        self._fill_combo(self._party_preset_combo, self._battle_setup_service.party_presets())
        self._party_preset_combo.currentIndexChanged.connect(self._update_summary)
        party_layout.addWidget(QLabel("Готовый шаблон отряда"))
        party_layout.addWidget(self._party_preset_combo)
        party_layout.addWidget(QLabel("Или созданные персонажи"))
        self._character_list.itemChanged.connect(self._update_summary)
        self._character_list.setMaximumHeight(170)
        party_layout.addWidget(self._character_list)
        return party_group

    def _parameters_group(self) -> QGroupBox:
        form_group = QGroupBox("Параметры боя")
        form = QFormLayout(form_group)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(6)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._fill_combo(self._difficulty_combo, self._battle_setup_service.difficulties())
        self._populate_map_options()
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
        return form_group

    def _map_group(self) -> QGroupBox:
        group = QGroupBox("Предпросмотр карты")
        layout = QVBoxLayout(group)
        self._map_preview.setMinimumSize(240, 170)
        layout.addWidget(self._map_preview, stretch=1)
        return group

    def _summary_group(self) -> QGroupBox:
        group = QGroupBox("Краткое описание боя")
        layout = QVBoxLayout(group)
        self._summary_text.setReadOnly(True)
        self._summary_text.setMinimumHeight(120)
        layout.addWidget(self._summary_text)
        return group

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
        self._set_start_busy(True)
        try:
            result = self._battle_setup_service.create_random_battle(request)
        except MapConfigValidationError as error:
            QMessageBox.warning(self, "Невалидная карта", str(error))
            self._summary_text.setPlainText(str(error))
            self._set_start_busy(False)
            return
        except ValueError as error:
            QMessageBox.warning(self, "Нельзя начать бой", str(error))
            self._summary_text.setPlainText(str(error))
            self._set_start_busy(False)
            return
        self._set_start_busy(False)
        if not self._model_service.is_model_loaded():
            QMessageBox.warning(
                self,
                "Модель не загружена",
                "Файл модели не загружен. Для AI будет использован запасной агент.",
            )
        self.battle_started.emit(result)

    def _update_summary(self) -> None:
        if not self._battle_setup_service.maps():
            self._summary_text.setPlainText("Нет карт. Проверьте папку maps/ в настройках.")
            self._map_preview.set_error("Нет карт. Проверьте папку maps/ в настройках.")
            return
        request = self._build_request()
        self._update_map_preview(request)
        try:
            summary = self._battle_setup_service.preview_battle(request)
        except ValueError as error:
            summary = str(error)
        self._summary_text.setPlainText(summary)

    def _update_map_preview(self, request: BattleSetupRequest | None = None) -> None:
        if not self._battle_setup_service.maps():
            self._map_preview.set_error("Нет карт. Проверьте папку maps/ в настройках.")
            return
        request = request or self._build_request()
        try:
            map_name = self._battle_setup_service.resolve_map_name(
                request.map_name,
                request.seed,
            )
            self._map_preview.set_map_config(
                self._battle_setup_service.get_map_config(map_name)
            )
        except (ValueError, MapConfigValidationError) as error:
            self._map_preview.set_error(str(error))

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

    def _apply_settings_seed(self) -> None:
        seed = self._model_service.settings.random_battle_seed
        enabled = seed is not None
        checkbox_blocked = self._seed_enabled.blockSignals(True)
        spin_blocked = self._seed_spin.blockSignals(True)
        self._seed_enabled.setChecked(enabled)
        self._seed_spin.setEnabled(enabled)
        if seed is not None:
            self._seed_spin.setValue(seed)
        self._seed_enabled.blockSignals(checkbox_blocked)
        self._seed_spin.blockSignals(spin_blocked)

    def _populate_map_options(self) -> None:
        current = self._map_combo.currentData()
        maps = self._battle_setup_service.maps()
        self._map_combo.blockSignals(True)
        self._map_combo.clear()
        if not maps:
            self._map_combo.addItem("Нет карт", "")
            self._map_combo.setEnabled(False)
            self._start_button.setEnabled(False)
            self._map_preview.set_error("Нет карт. Проверьте папку maps/ в настройках.")
        else:
            self._map_combo.setEnabled(True)
            self._start_button.setEnabled(True)
            self._fill_combo(self._map_combo, maps)
            if current is not None:
                self._set_combo_data(self._map_combo, str(current))
        self._map_combo.blockSignals(False)

    def _set_start_busy(self, busy: bool) -> None:
        self._start_button.setEnabled(not busy)
        self._start_button.setText("Создание боя..." if busy else "Начать бой")
        QApplication.processEvents()

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
