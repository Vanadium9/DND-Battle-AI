"""Settings screen for the desktop GUI."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFormLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from inference import CheckpointLoadError, PolicyCompatibilityError
from ui.animations import MAX_ANIMATION_SPEED_MS, MIN_ANIMATION_SPEED_MS
from ui.services import ModelService
from ui.settings import DEFAULT_CHECKPOINT_PATH, FIXED_FALLBACK_AGENT, FIXED_MODEL_TYPE
from ui.widgets.screen import ScreenFrame


class SettingsScreen(QWidget):
    """Configure safe GUI-only settings and show read-only model status."""

    settings_changed = Signal()

    def __init__(
        self,
        model_service: ModelService,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._model_service = model_service
        self._animations_checkbox = QCheckBox("Включены")
        self._animation_speed_spin = QSpinBox()
        self._autobattle_delay_spin = QSpinBox()
        self._model_label = QLabel()
        self._checkpoint_label = QLabel()
        self._fallback_label = QLabel()
        self._data_dirs_label = QLabel()
        self._status_label = QLabel()
        self._check_button = QPushButton("Проверить модель")
        self._build_layout()
        self.refresh()

    def refresh(self) -> None:
        settings = self._model_service.settings
        self._set_checked(self._animations_checkbox, settings.animations_enabled)
        self._set_spin_value(self._animation_speed_spin, settings.animation_speed)
        self._set_spin_value(self._autobattle_delay_spin, settings.autobattle_delay)
        self._model_label.setText("PPO Actor-Critic + GNN encoder")
        self._checkpoint_label.setText(settings.checkpoint_path or DEFAULT_CHECKPOINT_PATH)
        self._fallback_label.setText(
            self._model_service.fallback_agent_label(settings.fallback_agent)
        )
        self._data_dirs_label.setText(
            f"characters: {settings.character_dir}; replays: {settings.replay_dir}; maps: {settings.map_dir}"
        )
        self._update_status()

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(0)

        frame = ScreenFrame(
            "Настройки",
            "Безопасные параметры интерфейса. Архитектура модели и служебные пути зафиксированы проектом.",
        )
        form = QFormLayout()

        self._model_label.setWordWrap(True)
        self._checkpoint_label.setWordWrap(True)
        self._fallback_label.setWordWrap(True)
        self._data_dirs_label.setWordWrap(True)

        form.addRow("Модель", self._model_label)
        form.addRow("Checkpoint", self._checkpoint_label)
        form.addRow("Fallback", self._fallback_label)
        form.addRow("Папки данных", self._data_dirs_label)

        self._animations_checkbox.stateChanged.connect(lambda _state: self._save_settings())
        form.addRow("Анимации", self._animations_checkbox)

        self._configure_delay_spin(self._animation_speed_spin)
        self._animation_speed_spin.valueChanged.connect(lambda _value: self._save_settings())
        form.addRow("Скорость анимаций", self._animation_speed_spin)

        self._configure_delay_spin(self._autobattle_delay_spin)
        self._autobattle_delay_spin.valueChanged.connect(lambda _value: self._save_settings())
        form.addRow("Задержка автобоя", self._autobattle_delay_spin)

        frame.content_layout.addLayout(form)

        save_button = QPushButton("Сохранить")
        save_button.clicked.connect(self._save_settings)
        self._check_button.clicked.connect(self._check_model)
        frame.content_layout.addWidget(save_button)
        frame.content_layout.addWidget(self._check_button)

        self._status_label.setWordWrap(True)
        frame.content_layout.addWidget(self._status_label)
        frame.content_layout.addStretch(1)
        layout.addWidget(frame)

    def _save_settings(self) -> None:
        self._model_service.set_settings(
            model_type=FIXED_MODEL_TYPE,
            fallback_agent=FIXED_FALLBACK_AGENT,
            animations_enabled=self._animations_checkbox.isChecked(),
            animation_speed=self._animation_speed_spin.value(),
            autobattle_delay=self._autobattle_delay_spin.value(),
        )
        self.refresh()
        self._update_status(saved=True)
        self.settings_changed.emit()

    def _check_model(self) -> None:
        self._status_label.setText("Загрузка checkpoint...")
        self._set_check_busy(True)
        QApplication.processEvents()
        try:
            message = self._model_service.check_model()
        except (PolicyCompatibilityError, CheckpointLoadError) as error:
            self._show_error(_friendly_model_error(error))
            self._set_check_busy(False)
            return
        self._set_check_busy(False)
        self._status_label.setText(message)
        self.settings_changed.emit()
        QMessageBox.information(self, "Проверка модели", message)

    def _set_check_busy(self, busy: bool) -> None:
        self._check_button.setEnabled(not busy)
        self._check_button.setText("Загрузка..." if busy else "Проверить модель")

    def _update_status(self, *, saved: bool = False) -> None:
        prefix = "Настройки сохранены. " if saved else ""
        if self._model_service.last_error:
            message = _friendly_model_error(CheckpointLoadError(self._model_service.last_error))
            self._status_label.setText(f"{prefix}Ошибка: {message}")
            return
        policy_name = self._model_service.get_policy_name()
        if self._model_service.is_model_loaded():
            self._status_label.setText(f"{prefix}Активная политика: {policy_name}")
            return
        self._status_label.setText(
            f"{prefix}Checkpoint не загружен. Активная политика: {policy_name}"
        )

    def _show_error(self, message: str) -> None:
        self._status_label.setText(f"Ошибка: {message}")
        QMessageBox.warning(self, "Ошибка модели", message)

    @staticmethod
    def _configure_delay_spin(spin: QSpinBox) -> None:
        spin.setRange(MIN_ANIMATION_SPEED_MS, MAX_ANIMATION_SPEED_MS)
        spin.setSingleStep(100)
        spin.setSuffix(" мс")

    @staticmethod
    def _set_checked(checkbox: QCheckBox, checked: bool) -> None:
        was_blocked = checkbox.blockSignals(True)
        checkbox.setChecked(checked)
        checkbox.blockSignals(was_blocked)

    @staticmethod
    def _set_spin_value(spin: QSpinBox, value: int) -> None:
        was_blocked = spin.blockSignals(True)
        spin.setValue(value)
        spin.blockSignals(was_blocked)


def _friendly_model_error(error: CheckpointLoadError) -> str:
    text = str(error)
    lowered = text.lower()
    if "checkpoint not found" in lowered or "checkpoint не найден" in lowered:
        return text.replace("Checkpoint not found", "Checkpoint не найден")
    if "does not contain gnn" in lowered or "model_type" in lowered or "gnn" in lowered:
        return (
            "Checkpoint не соответствует фиксированной архитектуре "
            f"PPO Actor-Critic + GNN encoder: {text}"
        )
    if "incompatible" in lowered or "size mismatch" in lowered or "cannot infer" in lowered:
        return f"Неподдерживаемый action space или архитектура checkpoint: {text}"
    if "unsupported checkpoint format" in lowered or "no tensor" in lowered:
        return f"Checkpoint повреждён или имеет неподдерживаемый формат: {text}"
    if "failed to load" in lowered:
        return f"Checkpoint повреждён или не читается: {text}"
    return text
