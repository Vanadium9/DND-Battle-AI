"""Home screen for the desktop UI."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ui.widgets.screen import ScreenFrame


class HomeScreen(QWidget):
    """Start screen with a compact project overview."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(0)

        frame = ScreenFrame(
            "D&D Battle AI",
            "Desktop-интерфейс для просмотра и настройки тактических боёв.",
        )
        frame.add_body_text(
            "Проект использует обученную нейросеть для выбора действий в "
            "пошаговых D&D-like боях. Вся боевая логика остаётся в существующем "
            "combat engine: GUI только управляет сценариями, персонажами и "
            "визуализацией."
        )
        frame.add_body_text(
            "Обучение PPO/GNN не запускается из этого приложения. Для обучения "
            "используются отдельные консольные скрипты из папки scripts, например "
            "scripts/train_ppo.py."
        )

        status = QLabel(
            "Текущая основа GUI: навигация, экраны-заготовки и единая тема. "
            "Следующие итерации могут подключить редактор персонажей, запуск "
            "encounter и просмотр BattleReplay."
        )
        status.setWordWrap(True)
        frame.content_layout.addWidget(status)
        frame.content_layout.addStretch(1)
        layout.addWidget(frame)
