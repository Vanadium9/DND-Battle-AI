"""Spellcasting progression tables for supported levels 1-5."""

from __future__ import annotations


FULL_CASTER = "full_caster"

FULL_CASTER_SPELL_SLOTS: dict[int, dict[int, int]] = {
    1: {1: 2},
    2: {1: 3},
    3: {1: 4, 2: 2},
    4: {1: 4, 2: 3},
    5: {1: 4, 2: 3, 3: 2},
}

SPELLCASTING_TYPES: dict[str, dict[int, dict[int, int]]] = {
    FULL_CASTER: FULL_CASTER_SPELL_SLOTS,
}

CLASS_SPELLCASTING_TYPES: dict[str, str] = {
    "cleric": FULL_CASTER,
    "wizard": FULL_CASTER,
}

CLASS_SPELLCASTING_ABILITIES: dict[str, str] = {
    "cleric": "wis",
    "wizard": "int",
}


def get_spell_slots_for_progression(
    spellcasting_type: str | None,
    class_level: int,
) -> dict[int, int]:
    """Return spell slots for a class level and spellcasting progression type."""

    if spellcasting_type is None:
        return {}
    progression = SPELLCASTING_TYPES.get(spellcasting_type)
    if progression is None:
        progression = SPELLCASTING_TYPES.get(_lookup_key(spellcasting_type))
    if progression is None:
        return {}
    eligible_levels = [
        level
        for level in progression
        if level <= max(1, min(5, int(class_level)))
    ]
    if not eligible_levels:
        return {}
    return dict(progression[max(eligible_levels)])


def get_max_spell_level_for_progression(
    spellcasting_type: str | None,
    class_level: int,
) -> int:
    """Return the highest slot level available to this progression."""

    slots = get_spell_slots_for_progression(spellcasting_type, class_level)
    return max(slots, default=0)


def get_spellcasting_type_for_class(class_name: str | None) -> str | None:
    """Return the spellcasting progression type for a class."""

    if class_name is None:
        return None
    return CLASS_SPELLCASTING_TYPES.get(_lookup_key(class_name))


def get_spellcasting_ability_for_class(class_name: str | None) -> str | None:
    """Return the spellcasting ability score name for a class."""

    if class_name is None:
        return None
    return CLASS_SPELLCASTING_ABILITIES.get(_lookup_key(class_name))


def class_uses_spellcasting_progression(class_name: str | None) -> bool:
    """Return True if a class has a supported spellcasting progression."""

    return get_spellcasting_type_for_class(class_name) is not None


def _lookup_key(value: object) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())
