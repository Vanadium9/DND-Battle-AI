"""Supported spell selector widget."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QLabel, QVBoxLayout, QWidget

from combat.spellcasting import SpellDefinition, get_supported_spell_definitions
from ui.text import ru_label


class SpellSelector(QWidget):
    """Checkbox list of prepared spells allowed for the selected class/level."""

    selection_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._definitions: tuple[SpellDefinition, ...] = ()
        self._checkboxes: dict[str, QCheckBox] = {}
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(8)
        self.set_options(None, 1)

    def set_options(
        self,
        class_name: str | None,
        level: int,
        selected: tuple[str, ...] = (),
    ) -> None:
        """Refresh spell options for a class level."""

        self._clear()
        self._definitions = get_supported_spell_definitions(class_name, level)
        selected_keys = {_key(name) for name in selected}
        if not self._definitions:
            label = QLabel("Для выбранного класса и уровня заклинания недоступны.")
            label.setWordWrap(True)
            self._layout.addWidget(label)
            self._layout.addStretch(1)
            return

        for definition in self._definitions:
            checkbox = QCheckBox(_spell_label(definition))
            checkbox.setChecked(_key(definition.name) in selected_keys)
            checkbox.stateChanged.connect(self.selection_changed.emit)
            self._checkboxes[definition.name] = checkbox
            self._layout.addWidget(checkbox)
        self._layout.addStretch(1)

    def selected_spell_names(self) -> tuple[str, ...]:
        """Return checked spell names."""

        return tuple(
            name
            for name, checkbox in self._checkboxes.items()
            if checkbox.isChecked()
        )

    def selected_spell_payloads(self) -> tuple[dict[str, object], ...]:
        """Return checked spells with levels for InternalCharacter storage."""

        selected = {_key(name) for name in self.selected_spell_names()}
        return tuple(
            {
                "name": definition.name,
                "level": definition.spell_level,
                "action_cost": definition.action_cost,
            }
            for definition in self._definitions
            if _key(definition.name) in selected
        )

    def _clear(self) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._checkboxes = {}


def _spell_label(definition: SpellDefinition) -> str:
    level = "заговор" if definition.spell_level == 0 else f"{definition.spell_level} уровень"
    return f"{ru_label(definition.name)} ({level})"


def _key(value: object) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())
