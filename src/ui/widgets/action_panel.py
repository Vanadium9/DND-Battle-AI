"""Manual combat action panel."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ui.services.manual_action_builder import (
    ManualActionOption,
    ManualActionPlan,
    ManualTargetMode,
)


class ActionPanel(QWidget):
    """Render grouped legal manual actions for the active creature."""

    option_selected = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = QLabel("Ручное управление")
        self._title.setObjectName("panelTitle")
        self._hint = QLabel("AI управляет текущим существом.")
        self._hint.setWordWrap(True)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(8)
        self._scroll.setWidget(self._content)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self._title)
        layout.addWidget(self._hint)
        layout.addWidget(self._scroll, stretch=1)

    def set_viewer_mode(self, message: str = "AI управляет текущим существом.") -> None:
        self._hint.setText(message)
        self._clear_options()
        self._scroll.setEnabled(False)

    def set_plan(
        self,
        plan: ManualActionPlan,
        pending_option: ManualActionOption | None = None,
    ) -> None:
        pending_text = ""
        if pending_option is not None:
            pending_text = f" Выбрано: {pending_option.label}. Кликните по карте."
        self._hint.setText(f"Активный участник: {plan.actor_name}.{pending_text}")
        self._scroll.setEnabled(True)
        self._clear_options()
        for group_name in (
            "Movement",
            "Main Action",
            "Bonus Action",
            "Reaction",
            "End Turn",
        ):
            options = plan.groups.get(group_name, ())
            if not options:
                continue
            group = QGroupBox(group_name)
            group_layout = QVBoxLayout(group)
            group_layout.setContentsMargins(8, 8, 8, 8)
            group_layout.setSpacing(6)
            for option in options:
                button = QPushButton(_option_label(option))
                button.setCheckable(True)
                button.setChecked(pending_option is not None and option.id == pending_option.id)
                button.clicked.connect(lambda _checked=False, item=option: self.option_selected.emit(item))
                group_layout.addWidget(button)
            self._content_layout.addWidget(group)
        self._content_layout.addStretch(1)

    def _clear_options(self) -> None:
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()


def _option_label(option: ManualActionOption) -> str:
    suffix = ""
    if option.target_mode is ManualTargetMode.CREATURE:
        suffix = " -> цель"
    elif option.target_mode is ManualTargetMode.CELL:
        suffix = " -> клетка"
    return f"{option.label}{suffix}"
