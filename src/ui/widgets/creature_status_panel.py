"""Selected creature status panel."""

from __future__ import annotations

from PySide6.QtWidgets import QGroupBox, QPlainTextEdit, QVBoxLayout

from combat import CombatEnvironment
from ui.text import ru_label, ru_sentence


class CreatureStatusPanel(QGroupBox):
    """Show compact details for the selected creature."""

    def __init__(self, parent=None) -> None:
        super().__init__("Существо", parent)
        self._environment: CombatEnvironment | None = None
        self._creature_id: int | None = None
        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 14, 10, 10)
        layout.addWidget(self._text)

    def set_environment(
        self,
        environment: CombatEnvironment | None,
        creature_id: int | None = None,
    ) -> None:
        self._environment = environment
        self._creature_id = creature_id
        self.refresh()

    def set_creature_id(self, creature_id: int | None) -> None:
        self._creature_id = creature_id
        self.refresh()

    def refresh(self) -> None:
        if self._environment is None:
            self._text.setPlainText("Бой не запущен.")
            return
        state = self._environment.combat_state
        creature_id = self._creature_id
        if creature_id is None:
            creature_id = state.active_actor_id
        creature = state.character_at(creature_id) if creature_id is not None else None
        if creature is None:
            self._text.setPlainText("Существо не выбрано.")
            return
        self.setTitle(f"Существо | {creature.name}")
        self._text.setPlainText(_format_creature_status(creature_id, creature))


def _format_creature_status(creature_id: int, creature: object) -> str:
    conditions = [
        condition.name
        for condition in getattr(creature, "conditions", ())
    ]
    flags = []
    for label, attribute in (
        ("prone", "prone"),
        ("grappled", "grappled"),
        ("hidden", "hidden"),
        ("dodging", "dodging_until_start_of_next_turn"),
        ("disengaged", "disengaged_until_end_of_turn"),
        ("stable", "stable"),
    ):
        if bool(getattr(creature, attribute, False)):
            flags.append(label)

    resources = getattr(creature, "resources", {})
    resource_lines = [
        f"- {ru_sentence(name)}: {resource.uses_remaining}/{resource.max_uses}"
        for name, resource in resources.items()
    ]
    slots = getattr(creature, "spell_slots_remaining", {})
    slot_lines = [
        f"- уровень {level}: {slots.get(level, 0)}/{max_count}"
        for level, max_count in getattr(creature, "spell_slots", {}).items()
    ]
    weapons = [ru_sentence(weapon.name) for weapon in getattr(creature, "weapons", ())]
    inventory = [
        f"{ru_sentence(getattr(item, 'name', item))} x{getattr(item, 'quantity', 0)}"
        for item in getattr(creature, "inventory", ())
    ]
    return "\n".join(
        (
            f"ID: {creature_id}",
            f"Сторона: {ru_label(creature.team.name)}",
            f"HP: {creature.hp}/{creature.max_hp}",
            f"КД: {creature.ac}",
            f"Позиция: {creature.position.x}, {creature.position.y}",
            f"Класс: {ru_label(creature.class_name) if creature.class_name else '-'}",
            f"Подкласс: {ru_label(creature.subclass_name) if creature.subclass_name else '-'}",
            f"Раса: {ru_label(creature.race_name) if creature.race_name else '-'}",
            f"Роль: {ru_label(getattr(creature, 'role', '-'))}",
            f"Состояния: {', '.join(ru_label(item) for item in [*conditions, *flags]) or '-'}",
            "Экономика действий:",
            f"- действие: {creature.action_economy.action_available}",
            f"- бонусное действие: {creature.action_economy.bonus_action_available}",
            f"- реакция: {creature.action_economy.reaction_available}",
            f"- перемещение: {creature.action_economy.movement_remaining}",
            f"Оружие: {', '.join(weapons) or '-'}",
            "Ресурсы:",
            *(resource_lines or ["-"]),
            "Ячейки заклинаний:",
            *(slot_lines or ["-"]),
            f"Инвентарь: {', '.join(inventory) or '-'}",
        )
    )
