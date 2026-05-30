"""Ability score editor widget."""

from __future__ import annotations

import random

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QGridLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


ABILITY_LABELS: dict[str, str] = {
    "str": "STR",
    "dex": "DEX",
    "con": "CON",
    "int": "INT",
    "wis": "WIS",
    "cha": "CHA",
}


class StatEditor(QWidget):
    """Editable ability scores with a simple generator."""

    stats_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._spin_boxes: dict[str, QSpinBox] = {}
        self._build_layout()
        self.set_stats({ability: 10 for ability in ABILITY_LABELS})

    def stats(self) -> dict[str, int]:
        """Return current ability scores."""

        return {
            ability: spin_box.value()
            for ability, spin_box in self._spin_boxes.items()
        }

    def set_stats(self, stats: dict[str, int]) -> None:
        """Replace visible ability scores."""

        for ability, spin_box in self._spin_boxes.items():
            spin_box.setValue(int(stats.get(ability, 10)))

    def apply_bonuses(self, bonuses: dict[str, int]) -> None:
        """Apply racial/ASI bonuses to the visible scores."""

        for ability, bonus in bonuses.items():
            spin_box = self._spin_boxes.get(ability)
            if spin_box is None:
                continue
            spin_box.setValue(spin_box.value() + int(bonus))

    def _build_layout(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        for column, (ability, label) in enumerate(ABILITY_LABELS.items()):
            label_widget = QLabel(label)
            spin_box = QSpinBox()
            spin_box.setRange(1, 30)
            spin_box.valueChanged.connect(self.stats_changed.emit)
            self._spin_boxes[ability] = spin_box
            grid.addWidget(label_widget, 0, column)
            grid.addWidget(spin_box, 1, column)
        root.addLayout(grid)

        generate_button = QPushButton("Сгенерировать характеристики")
        generate_button.clicked.connect(self._generate_stats)
        root.addWidget(generate_button)

    def _generate_stats(self) -> None:
        generated = {
            ability: _roll_4d6_drop_lowest()
            for ability in ABILITY_LABELS
        }
        self.set_stats(generated)


def _roll_4d6_drop_lowest() -> int:
    rolls = sorted(random.randint(1, 6) for _ in range(4))
    return sum(rolls[1:])
