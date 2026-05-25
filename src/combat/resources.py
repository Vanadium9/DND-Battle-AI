"""Runtime resource helpers for spell slots and similar combat resources."""

from __future__ import annotations

from typing import Any


def reset_spell_slots(character: Any, slots: dict[int, int] | None = None) -> dict[int, int]:
    """Reset a character's remaining spell slots to the provided slot table."""

    slot_table = _normalize_slots(slots if slots is not None else getattr(character, "spell_slots", {}))
    character.spell_slots = dict(slot_table)
    character.spell_slots_remaining = dict(slot_table)
    return dict(slot_table)


def has_spell_slot(character: Any, spell_level: int) -> bool:
    """Return True if the character has an available slot for a levelled spell."""

    normalized_level = int(spell_level)
    if normalized_level <= 0:
        return True
    remaining = getattr(character, "spell_slots_remaining", None)
    if isinstance(remaining, dict):
        return int(remaining.get(normalized_level, 0)) > 0
    slots = getattr(character, "spell_slots", None)
    if isinstance(slots, dict):
        return int(slots.get(normalized_level, 0)) > 0
    if isinstance(slots, int):
        return slots > 0
    return False


def spend_spell_slot(character: Any, spell_level: int) -> bool:
    """Spend one spell slot for a levelled spell."""

    normalized_level = int(spell_level)
    if normalized_level <= 0:
        return True
    if not has_spell_slot(character, normalized_level):
        return False
    remaining = getattr(character, "spell_slots_remaining", None)
    if not isinstance(remaining, dict):
        remaining = dict(getattr(character, "spell_slots", {}))
        character.spell_slots_remaining = remaining
    remaining[normalized_level] = max(0, int(remaining.get(normalized_level, 0)) - 1)
    return True


def max_available_spell_slot_level(character: Any) -> int:
    """Return the highest spell slot level with at least one remaining slot."""

    remaining = getattr(character, "spell_slots_remaining", None)
    if not isinstance(remaining, dict):
        return 0
    available = [level for level, count in remaining.items() if int(count) > 0]
    return max(available, default=0)


def _normalize_slots(slots: dict[int, int] | None) -> dict[int, int]:
    if slots is None:
        return {}
    return {
        int(level): max(0, int(count))
        for level, count in slots.items()
        if int(level) > 0 and int(count) > 0
    }
