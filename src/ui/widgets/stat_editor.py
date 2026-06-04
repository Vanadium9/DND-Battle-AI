"""Ability score editor widget."""

from __future__ import annotations

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
    "str": "Сила",
    "dex": "Ловкость",
    "con": "Телосложение",
    "int": "Интеллект",
    "wis": "Мудрость",
    "cha": "Харизма",
}

POINT_BUY_BUDGET = 27
STAT_MIN = 8
STAT_MAX = 20
POINT_BUY_COSTS: dict[int, int] = {
    8: 0,
    9: 1,
    10: 2,
    11: 3,
    12: 4,
    13: 5,
    14: 7,
    15: 9,
}


class StatEditor(QWidget):
    """D&D 5e point-buy ability score editor."""

    stats_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._spin_boxes: dict[str, QSpinBox] = {}
        self._bonus_labels: dict[str, QLabel] = {}
        self._final_labels: dict[str, QLabel] = {}
        self._modifier_labels: dict[str, QLabel] = {}
        self._save_labels: dict[str, QLabel] = {}
        self._racial_bonuses: dict[str, int] = {}
        self._proficiency_bonus = 2
        self._saving_throw_proficiencies: set[str] = set()
        self._build_layout()
        self.set_stats({ability: 8 for ability in ABILITY_LABELS})

    def stats(self) -> dict[str, int]:
        """Return final ability scores after racial bonuses."""

        return {
            ability: spin_box.value() + int(self._racial_bonuses.get(ability, 0))
            for ability, spin_box in self._spin_boxes.items()
        }

    def points_remaining(self) -> int:
        """Return remaining point-buy budget before racial bonuses."""

        spent = sum(
            _point_buy_cost(spin_box.value())
            for spin_box in self._spin_boxes.values()
        )
        return POINT_BUY_BUDGET - spent

    def set_stats(self, stats: dict[str, int]) -> None:
        """Replace visible final ability scores, preserving current racial bonuses."""

        changed = False
        for ability, spin_box in self._spin_boxes.items():
            bonus = int(self._racial_bonuses.get(ability, 0))
            value = max(STAT_MIN, min(STAT_MAX, int(stats.get(ability, STAT_MIN)) - bonus))
            if spin_box.value() == value:
                continue
            was_blocked = spin_box.blockSignals(True)
            spin_box.setValue(value)
            spin_box.blockSignals(was_blocked)
            changed = True
        self._refresh_summary()
        if changed:
            self.stats_changed.emit()

    def apply_bonuses(self, bonuses: dict[str, int]) -> None:
        """Compatibility wrapper for old callers."""

        self.set_racial_bonuses(bonuses)

    def set_racial_bonuses(self, bonuses: dict[str, int]) -> None:
        """Replace racial bonuses and update final values immediately."""

        self._racial_bonuses = {ability: int(value) for ability, value in bonuses.items()}
        self._refresh_summary()
        self.stats_changed.emit()

    def set_proficiency_context(
        self,
        *,
        proficiency_bonus: int,
        saving_throw_proficiencies: tuple[str, ...] | list[str] | set[str],
    ) -> None:
        """Update class proficiency context used for saving throw display."""

        self._proficiency_bonus = int(proficiency_bonus)
        self._saving_throw_proficiencies = {
            str(ability).casefold()
            for ability in saving_throw_proficiencies
        }
        self._refresh_summary()

    def _build_layout(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        self._points_label = QLabel()
        self._points_label.setObjectName("pointBuyStatus")
        root.addWidget(self._points_label)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        headers = ("Характеристика", "База", "Раса", "Итог", "Мод.", "Спас.")
        for column, header in enumerate(headers):
            header_label = QLabel(header)
            header_label.setObjectName("compactHeader")
            grid.addWidget(header_label, 0, column)

        for row, (ability, label) in enumerate(ABILITY_LABELS.items(), start=1):
            label_widget = QLabel(label)
            spin_box = QSpinBox()
            spin_box.setRange(STAT_MIN, STAT_MAX)
            spin_box.valueChanged.connect(self._base_stat_changed)
            bonus_label = QLabel("+0")
            final_label = QLabel(str(STAT_MIN))
            final_label.setObjectName("sheetMetricValue")
            modifier_label = QLabel("-1")
            save_label = QLabel("-1")
            self._spin_boxes[ability] = spin_box
            self._bonus_labels[ability] = bonus_label
            self._final_labels[ability] = final_label
            self._modifier_labels[ability] = modifier_label
            self._save_labels[ability] = save_label
            grid.addWidget(label_widget, row, 0)
            grid.addWidget(spin_box, row, 1)
            grid.addWidget(bonus_label, row, 2)
            grid.addWidget(final_label, row, 3)
            grid.addWidget(modifier_label, row, 4)
            grid.addWidget(save_label, row, 5)
        root.addLayout(grid)

        reset_button = QPushButton("Сбросить до 8")
        reset_button.clicked.connect(self._reset_stats)
        root.addWidget(reset_button)

    def _base_stat_changed(self) -> None:
        self._refresh_summary()
        self.stats_changed.emit()

    def _reset_stats(self) -> None:
        self.set_stats({
            ability: 8 + int(self._racial_bonuses.get(ability, 0))
            for ability in ABILITY_LABELS
        })

    def _refresh_summary(self) -> None:
        spent = 0
        for ability, spin_box in self._spin_boxes.items():
            base_value = spin_box.value()
            bonus = int(self._racial_bonuses.get(ability, 0))
            final_value = base_value + bonus
            modifier = _ability_modifier(final_value)
            save_bonus = None
            if ability in self._saving_throw_proficiencies:
                save_bonus = modifier + self._proficiency_bonus
            spent += _point_buy_cost(base_value)
            self._bonus_labels[ability].setText(f"+{bonus}" if bonus >= 0 else str(bonus))
            self._final_labels[ability].setText(str(final_value))
            self._modifier_labels[ability].setText(_signed(modifier))
            self._save_labels[ability].setText(_signed(save_bonus) if save_bonus is not None else "-")

        remaining = POINT_BUY_BUDGET - spent
        self._points_label.setText(
            f"Доступные очки характеристик: {remaining} из {POINT_BUY_BUDGET}"
        )
        self._points_label.setProperty("overBudget", remaining < 0)
        self._points_label.style().unpolish(self._points_label)
        self._points_label.style().polish(self._points_label)


def _ability_modifier(score: int) -> int:
    return (int(score) - 10) // 2


def _point_buy_cost(value: int) -> int:
    normalized = max(STAT_MIN, int(value))
    if normalized in POINT_BUY_COSTS:
        return POINT_BUY_COSTS[normalized]
    return POINT_BUY_COSTS[15] + (normalized - 15) * 2


def _signed(value: int) -> str:
    return f"+{value}" if value >= 0 else str(value)
