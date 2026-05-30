"""Settings screen for GUI inference configuration."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from inference import CheckpointLoadError
from ui.services import ModelService
from ui.widgets.screen import ScreenFrame


class SettingsScreen(QWidget):
    """Configure inference-only model settings for the desktop UI."""

    def __init__(
        self,
        model_service: ModelService,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._model_service = model_service
        self._checkpoint_input = QLineEdit()
        self._model_type_combo = QComboBox()
        self._fallback_combo = QComboBox()
        self._status_label = QLabel()
        self._build_layout()
        self.refresh()

    def refresh(self) -> None:
        settings = self._model_service.settings
        self._checkpoint_input.setText(settings.checkpoint_path)
        self._set_combo_value(self._model_type_combo, settings.model_type)
        self._set_combo_value(self._fallback_combo, settings.fallback_agent)
        self._update_status()

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(0)

        frame = ScreenFrame(
            "Настройки",
            "Загрузка обученного checkpoint для GUI-инференса. Обучение запускается только через CLI-скрипты.",
        )
        form = QFormLayout()

        checkpoint_row = QHBoxLayout()
        browse_button = QPushButton("Обзор")
        browse_button.clicked.connect(self._browse_checkpoint)
        self._checkpoint_input.editingFinished.connect(self._save_settings)
        checkpoint_row.addWidget(self._checkpoint_input, stretch=1)
        checkpoint_row.addWidget(browse_button)
        form.addRow("Checkpoint", checkpoint_row)

        self._model_type_combo.addItems(self._model_service.available_model_types())
        self._model_type_combo.currentTextChanged.connect(self._save_settings)
        form.addRow("model_type", self._model_type_combo)

        self._fallback_combo.addItems(self._model_service.available_fallback_agents())
        self._fallback_combo.currentTextChanged.connect(self._save_settings)
        form.addRow("fallback_agent", self._fallback_combo)

        frame.content_layout.addLayout(form)

        button_row = QHBoxLayout()
        load_button = QPushButton("Загрузить checkpoint")
        fallback_button = QPushButton("Использовать fallback")
        load_button.clicked.connect(self._load_checkpoint)
        fallback_button.clicked.connect(self._use_fallback)
        button_row.addWidget(load_button)
        button_row.addWidget(fallback_button)
        button_row.addStretch(1)
        frame.content_layout.addLayout(button_row)

        self._status_label.setWordWrap(True)
        frame.content_layout.addWidget(self._status_label)
        frame.content_layout.addStretch(1)
        layout.addWidget(frame)

    def _browse_checkpoint(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Выбрать checkpoint",
            str(Path(self._checkpoint_input.text() or "checkpoints").parent),
            "PyTorch checkpoint (*.pt *.pth);;All files (*.*)",
        )
        if selected:
            self._checkpoint_input.setText(selected)
            self._save_settings()

    def _save_settings(self) -> None:
        self._model_service.set_settings(
            checkpoint_path=self._checkpoint_input.text().strip(),
            model_type=self._model_type_combo.currentText(),
            fallback_agent=self._fallback_combo.currentText(),
        )
        self._update_status()

    def _load_checkpoint(self) -> None:
        self._save_settings()
        checkpoint_path = self._checkpoint_input.text().strip()
        if not checkpoint_path:
            self._show_error("Checkpoint не выбран. Будет использоваться fallback agent.")
            return
        try:
            self._model_service.load_checkpoint(
                checkpoint_path,
                self._model_type_combo.currentText(),
            )
        except CheckpointLoadError as error:
            self._show_error(str(error))
            return
        self._update_status()

    def _use_fallback(self) -> None:
        self._model_service.battle_ai.unload_checkpoint()
        self._update_status()

    def _update_status(self) -> None:
        policy_name = self._model_service.get_policy_name()
        if self._model_service.is_model_loaded():
            self._status_label.setText(f"Активная политика: {policy_name}")
            return
        self._status_label.setText(
            f"Checkpoint не загружен. Активная политика: {policy_name}"
        )

    def _show_error(self, message: str) -> None:
        self._status_label.setText(f"Ошибка: {message}")
        QMessageBox.warning(self, "Ошибка загрузки модели", message)

    @staticmethod
    def _set_combo_value(combo: QComboBox, value: str) -> None:
        was_blocked = combo.blockSignals(True)
        index = combo.findText(value)
        if index >= 0:
            combo.setCurrentIndex(index)
        combo.blockSignals(was_blocked)
