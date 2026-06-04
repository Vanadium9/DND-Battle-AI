"""Manual combat action panel."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.services.manual_action_builder import (
    ManualActionOption,
    ManualActionPlan,
    ManualTargetMode,
)
from ui.text import ru_label, ru_sentence


class ActionPanel(QWidget):
    """Render grouped manual actions as stable compact ability slots."""

    option_selected = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("actionPanel")
        self.setMinimumWidth(220)
        self._title = QLabel("Ручное управление")
        self._title.setObjectName("panelTitle")
        self._hint = QLabel("AI управляет текущим существом.")
        self._hint.setWordWrap(True)
        self._content = QWidget()
        self._content_layout = QHBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(6)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._title)
        layout.addWidget(self._hint)
        layout.addWidget(self._content, stretch=1)

    def set_viewer_mode(self, message: str = "AI управляет текущим существом.") -> None:
        self._hint.setText(message)
        self._clear_options()
        self._content.setEnabled(False)

    def set_plan(
        self,
        plan: ManualActionPlan,
        pending_option: ManualActionOption | None = None,
    ) -> None:
        pending_text = ""
        if pending_option is not None:
            pending_text = f" Выбрано: {ru_sentence(pending_option.label)}. Кликните по карте."
        self._hint.setText(f"Активный участник: {ru_sentence(plan.actor_name)}.{pending_text}")
        self._content.setEnabled(True)
        self._clear_options()

        grouped = {
            group_name: {
                _slot_key(option.label): option
                for option in plan.groups.get(group_name, ())
            }
            for group_name, _labels in STABLE_SLOTS
        }

        for group_name, stable_labels in STABLE_SLOTS:
            legal_options = list(plan.groups.get(group_name, ()))
            used_option_ids: set[str] = set()
            group = QGroupBox(ru_label(group_name))
            group.setObjectName("abilityColumn")
            group.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
            group_layout = QGridLayout(group)
            group_layout.setContentsMargins(5, 8, 5, 5)
            group_layout.setSpacing(3)

            row = 0
            column = 0
            for label in stable_labels:
                option = grouped.get(group_name, {}).get(_slot_key(label))
                if option is not None:
                    used_option_ids.add(option.id)
                self._add_slot_button(
                    group_layout,
                    row,
                    column,
                    label,
                    option,
                    pending_option,
                )
                row, column = _next_grid_position(row, column)

            for option in legal_options:
                if option.id in used_option_ids:
                    continue
                self._add_slot_button(
                    group_layout,
                    row,
                    column,
                    option.label,
                    option,
                    pending_option,
                )
                row, column = _next_grid_position(row, column)

            self._content_layout.addWidget(group, stretch=0)
        self._content_layout.addStretch(1)

    def _add_slot_button(
        self,
        layout: QGridLayout,
        row: int,
        column: int,
        label: str,
        option: ManualActionOption | None,
        pending_option: ManualActionOption | None,
    ) -> None:
        source_label = option.label if option is not None else label
        button = QPushButton(_option_short_label(source_label, option.target_mode if option else ManualTargetMode.NONE))
        button.setToolTip(_option_label(option) if option is not None else f"{ru_sentence(label)} недоступно")
        button.setCheckable(option is not None)
        button.setEnabled(option is not None)
        button.setObjectName("compactActionButton")
        button.setFixedSize(32, 32)
        button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        if option is not None:
            button.setChecked(pending_option is not None and option.id == pending_option.id)
            button.clicked.connect(lambda _checked=False, item=option: self.option_selected.emit(item))
        layout.addWidget(button, row, column, alignment=Qt.AlignmentFlag.AlignCenter)

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
    quantity = ""
    if option.item is not None:
        quantity = f" | количество: {getattr(option.item, 'quantity', 0)}"
    return f"{ru_sentence(option.label)}{suffix}{quantity}"


def _option_short_label(label: str, target_mode: ManualTargetMode) -> str:
    translated = ru_sentence(label)
    key = _slot_key(label)
    target_marker = ""
    if target_mode is ManualTargetMode.CREATURE:
        target_marker = "◎"
    elif target_mode is ManualTargetMode.CELL:
        target_marker = "◇"
    icon = _ACTION_ICONS.get(key)
    if icon is not None:
        return f"{icon}{target_marker}"
    letters = "".join(character for character in translated if character.isalnum())
    return f"{letters[:2].upper() or '?'}{target_marker}"


def _slot_key(label: str) -> str:
    normalized = str(label).casefold()
    normalized = normalized.removeprefix("class feature:").strip()
    if ":" in normalized:
        normalized = normalized.split(":", 1)[0].strip()
    if normalized.startswith("shove"):
        return "shove"
    if normalized.startswith("search"):
        return "search"
    if normalized.startswith("move"):
        return "move"
    return normalized


def _next_grid_position(row: int, column: int) -> tuple[int, int]:
    column += 1
    if column >= SLOT_COLUMNS:
        return row + 1, 0
    return row, column


SLOT_COLUMNS = 4

STABLE_SLOTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Movement", ("Move", "Dash", "Disengage", "Dodge", "Hide")),
    (
        "Attack Abilities",
        (
            "Attack",
            "Help",
            "Grapple",
            "Shove",
            "Stabilize",
            "Search",
        ),
    ),
    (
        "Spells",
        (
            "Cast Spell",
            "Action Surge",
            "Second Wind",
            "Preserve Life",
            "Healing Word",
        ),
    ),
    ("Inventory", ("Potion of Healing", "Bomb", "Alchemist Fire", "HealerKit")),
    ("End Turn", ("End Turn",)),
)

_ACTION_ICONS: dict[str, str] = {
    "move": "↦",
    "attack": "⚔",
    "cast spell": "✦",
    "dash": "⇥",
    "disengage": "↩",
    "dodge": "◈",
    "help": "✚",
    "hide": "◌",
    "search": "⌕",
    "use object": "▣",
    "ready": "⏱",
    "grapple": "⌁",
    "shove": "⇢",
    "stabilize": "✚",
    "end turn": "⏭",
    "second wind": "✚",
    "action surge": "★",
    "preserve life": "✚",
    "healing word": "✚",
    "potion of healing": "✚",
    "shield": "⬡",
    "opportunity attack": "⚔",
}
