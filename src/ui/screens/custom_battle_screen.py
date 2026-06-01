"""Custom battle setup shell with map config preview."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from combat.map_config import MapConfigValidationError
from ui.services import BattleSetupService
from ui.widgets import MapPreviewWidget
from ui.widgets.screen import ScreenFrame


class CustomBattleScreen(QWidget):
    """Start of custom battle configuration focused on map selection."""

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
            "Здесь выбирается карта из JSON-конфигов. Состав сторон будет подключаться к существующему combat engine отдельно.",
        )
        form = QFormLayout()
        self._map_combo.currentIndexChanged.connect(self._update_map_preview)
        form.addRow("Карта", self._map_combo)
        frame.content_layout.addLayout(form)
        frame.content_layout.addWidget(QLabel("Preview карты"))
        frame.content_layout.addWidget(self._map_preview, stretch=1)
        self._status_label.setWordWrap(True)
        frame.content_layout.addWidget(self._status_label)
        layout.addWidget(frame)

    def _update_map_preview(self) -> None:
        if not self._battle_setup_service.maps():
            message = "Нет карт. Проверьте папку maps/ в настройках."
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

    def _set_combo_data(self, value: str) -> None:
        for index in range(self._map_combo.count()):
            if self._map_combo.itemData(index) == value:
                self._map_combo.setCurrentIndex(index)
                return
