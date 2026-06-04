"""Custom battle setup and simple map builder."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from combat.map_config import (
    MapConfig,
    MapConfigValidationError,
    generate_random_map_config,
    normalize_map_key,
    save_map_config,
)
from ui.services import BattleSetupService
from ui.settings import resolve_project_path
from ui.widgets import MapPreviewWidget
from ui.widgets.screen import ScreenFrame


MAP_TEMPLATES: dict[str, str] = {
    "balanced": "Сбалансированная",
    "open": "Открытое поле",
    "cover": "Много укрытий",
    "terrain": "Труднопроходимая местность",
    "obstacles": "Препятствия и коридоры",
}


class CustomBattleScreen(QWidget):
    """Custom battle shell with map selection and map generation."""

    def __init__(
        self,
        battle_setup_service: BattleSetupService,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._battle_setup_service = battle_setup_service
        self._map_combo = QComboBox()
        self._map_preview = MapPreviewWidget()
        self._status_label = QLabel()
        self._name_edit = QLineEdit()
        self._width_spin = QSpinBox()
        self._height_spin = QSpinBox()
        self._template_combo = QComboBox()
        self._seed_spin = QSpinBox()
        self._generated_config: MapConfig | None = None
        self._build_layout()
        self.refresh()

    def refresh(self) -> None:
        current = self._map_combo.currentData()
        self._map_combo.blockSignals(True)
        self._map_combo.clear()
        maps = self._battle_setup_service.maps()
        if not maps:
            self._map_combo.addItem("Нет карт", "")
            self._map_combo.setEnabled(False)
        else:
            self._map_combo.setEnabled(True)
        for key, label in maps.items():
            if key != "random":
                self._map_combo.addItem(label, key)
        self._map_combo.blockSignals(False)
        if current is not None:
            self._set_combo_data(str(current))
        self._update_map_preview()

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(0)

        frame = ScreenFrame(
            "Кастомный бой",
            "Выбор карты и простой генератор карт. Настройку состава сторон подключим следующим этапом.",
        )
        content = QHBoxLayout()
        content.addWidget(self._selection_group(), stretch=1)
        content.addWidget(self._builder_group(), stretch=1)
        frame.content_layout.addLayout(content)
        frame.content_layout.addWidget(QLabel("Предпросмотр карты"))
        frame.content_layout.addWidget(self._map_preview, stretch=1)
        self._status_label.setWordWrap(True)
        frame.content_layout.addWidget(self._status_label)
        layout.addWidget(frame)

    def _selection_group(self) -> QGroupBox:
        group = QGroupBox("Готовые карты")
        form = QFormLayout(group)
        self._map_combo.currentIndexChanged.connect(self._update_map_preview)
        form.addRow("Карта", self._map_combo)
        refresh_button = QPushButton("Обновить список")
        refresh_button.clicked.connect(self.refresh)
        form.addRow("", refresh_button)
        return group

    def _builder_group(self) -> QGroupBox:
        group = QGroupBox("Создать карту")
        form = QFormLayout(group)
        self._name_edit.setPlaceholderText("Например: Лесная засада")
        self._name_edit.setText("Случайная карта")
        self._width_spin.setRange(4, 16)
        self._width_spin.setValue(8)
        self._height_spin.setRange(4, 12)
        self._height_spin.setValue(6)
        for key, label in MAP_TEMPLATES.items():
            self._template_combo.addItem(label, key)
        self._seed_spin.setRange(0, 2_147_483_647)
        self._seed_spin.setValue(0)

        generate_button = QPushButton("Сгенерировать")
        generate_button.clicked.connect(self._generate_map)
        save_button = QPushButton("Сохранить карту")
        save_button.clicked.connect(self._save_generated_map)
        button_row = QHBoxLayout()
        button_row.addWidget(generate_button)
        button_row.addWidget(save_button)

        form.addRow("Название", self._name_edit)
        form.addRow("Ширина", self._width_spin)
        form.addRow("Высота", self._height_spin)
        form.addRow("Тип", self._template_combo)
        form.addRow("Seed", self._seed_spin)
        form.addRow("", button_row)
        return group

    def _update_map_preview(self) -> None:
        if self._generated_config is not None:
            self._map_preview.set_map_config(self._generated_config)
            self._status_label.setObjectName("inlineStatus")
            self._status_label.setText(
                f"Сгенерирована карта: {self._generated_config.name}, "
                f"{self._generated_config.width}x{self._generated_config.height}. "
                "Нажмите «Сохранить карту», чтобы добавить её в список."
            )
            return
        if not self._battle_setup_service.maps():
            message = "Нет карт. Создайте карту через генератор справа."
            self._map_preview.set_error(message)
            self._status_label.setObjectName("warningStatus")
            self._status_label.setText(message)
            return
        map_name = str(self._map_combo.currentData() or "open_field")
        try:
            config = self._battle_setup_service.get_map_config(map_name)
        except (ValueError, MapConfigValidationError) as error:
            self._map_preview.set_error(str(error))
            self._status_label.setObjectName("warningStatus")
            self._status_label.setText(f"Ошибка карты: {error}")
            return
        self._map_preview.set_map_config(config)
        self._status_label.setObjectName("inlineStatus")
        self._status_label.setText(
            f"Карта валидна: {config.name}, {config.width}x{config.height}."
        )

    def _generate_map(self) -> None:
        name = self._name_edit.text().strip() or "Случайная карта"
        try:
            self._generated_config = generate_random_map_config(
                name=name,
                width=self._width_spin.value(),
                height=self._height_spin.value(),
                seed=self._seed_spin.value(),
                template=str(self._template_combo.currentData() or "balanced"),
            )
        except MapConfigValidationError as error:
            QMessageBox.warning(self, "Невалидная карта", str(error))
            return
        self._update_map_preview()

    def _save_generated_map(self) -> None:
        if self._generated_config is None:
            self._generate_map()
        if self._generated_config is None:
            return
        map_dir = resolve_project_path(self._battle_setup_service.map_dir)
        file_name = f"{normalize_map_key(self._generated_config.name)}.json"
        target_path = _unique_map_path(map_dir / file_name)
        try:
            save_map_config(self._generated_config, target_path)
        except OSError as error:
            QMessageBox.warning(self, "Не удалось сохранить карту", str(error))
            return
        QMessageBox.information(
            self,
            "Карта сохранена",
            f"Карта сохранена: {target_path.name}",
        )
        self._generated_config = None
        self.refresh()

    def _set_combo_data(self, value: str) -> None:
        for index in range(self._map_combo.count()):
            if self._map_combo.itemData(index) == value:
                self._map_combo.setCurrentIndex(index)
                return


def _unique_map_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(2, 1000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    return path.with_name(f"{stem}_new{suffix}")
