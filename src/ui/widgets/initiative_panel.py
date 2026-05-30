"""Initiative order panel for the battle screen."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QGroupBox, QListWidget, QListWidgetItem, QVBoxLayout

from combat import CombatEnvironment


class InitiativePanel(QGroupBox):
    """Show round number, active actor and initiative order."""

    creature_selected = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__("Инициатива", parent)
        self._environment: CombatEnvironment | None = None
        self._selected_creature_id: int | None = None
        self._list = QListWidget()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 14, 10, 10)
        layout.addWidget(self._list)
        self._list.itemClicked.connect(self._item_clicked)

    def set_environment(
        self,
        environment: CombatEnvironment | None,
        selected_creature_id: int | None = None,
    ) -> None:
        self._environment = environment
        self._selected_creature_id = selected_creature_id
        self.refresh()

    def set_selected_creature_id(self, creature_id: int | None) -> None:
        self._selected_creature_id = creature_id
        self.refresh()

    def refresh(self) -> None:
        self._list.clear()
        if self._environment is None:
            return
        state = self._environment.combat_state
        self.setTitle(f"Инициатива | Раунд {state.round_number}")
        order = state.initiative_order or list(range(len(state.characters)))
        active_actor_id = state.active_actor_id
        for position, creature_id in enumerate(order, start=1):
            creature = state.character_at(creature_id)
            if creature is None:
                continue
            active_marker = "▶ " if creature_id == active_actor_id else "  "
            dead_marker = " ✕" if creature.is_dead else ""
            item = QListWidgetItem(
                f"{active_marker}{position}. {creature.name} ({creature.hp}/{creature.max_hp}){dead_marker}"
            )
            item.setData(Qt.ItemDataRole.UserRole, creature_id)
            if creature_id == active_actor_id:
                item.setForeground(QColor("#1f5a53"))
            elif creature.is_dead:
                item.setForeground(QColor("#6b7280"))
            if creature_id == self._selected_creature_id:
                item.setBackground(QColor("#e6f2f0"))
            self._list.addItem(item)

    def _item_clicked(self, item: QListWidgetItem) -> None:
        creature_id = int(item.data(Qt.ItemDataRole.UserRole))
        self._selected_creature_id = creature_id
        self.creature_selected.emit(creature_id)
        self.refresh()
