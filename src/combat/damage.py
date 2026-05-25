"""Damage type and damage modifier rules."""

from __future__ import annotations

from enum import Enum
from typing import Any, Iterable


class DamageType(str, Enum):
    """Supported D&D-like damage types."""

    SLASHING = "slashing"
    PIERCING = "piercing"
    BLUDGEONING = "bludgeoning"
    FIRE = "fire"
    COLD = "cold"
    LIGHTNING = "lightning"
    ACID = "acid"
    POISON = "poison"
    NECROTIC = "necrotic"
    RADIANT = "radiant"
    FORCE = "force"
    PSYCHIC = "psychic"
    THUNDER = "thunder"


DAMAGE_TYPES: tuple[DamageType, ...] = tuple(DamageType)


def coerce_damage_type(value: object) -> DamageType | None:
    """Return a DamageType from enum/string-like input."""

    if value is None:
        return None
    if isinstance(value, DamageType):
        return value
    normalized = _lookup_key(value)
    for damage_type in DamageType:
        if normalized in {_lookup_key(damage_type.name), _lookup_key(damage_type.value)}:
            return damage_type
    return None


def normalize_damage_type_set(values: Iterable[object] | None) -> set[DamageType]:
    """Normalize mixed enum/string damage type values to a set."""

    if values is None:
        return set()
    result: set[DamageType] = set()
    for value in values:
        damage_type = coerce_damage_type(value)
        if damage_type is not None:
            result.add(damage_type)
    return result


def normalize_character_damage_profile(character: Any) -> None:
    """Normalize damage profile sets stored on a character."""

    character.resistances = normalize_damage_type_set(getattr(character, "resistances", ()))
    character.immunities = normalize_damage_type_set(getattr(character, "immunities", ()))
    character.vulnerabilities = normalize_damage_type_set(
        getattr(character, "vulnerabilities", ())
    )


def character_resistances(character: Any) -> set[DamageType]:
    """Return all damage resistances active on a character."""

    resistances = normalize_damage_type_set(getattr(character, "resistances", ()))
    traits = getattr(character, "race_traits", None)
    if traits is not None:
        resistances.update(
            normalize_damage_type_set(getattr(traits, "damage_resistances", ()))
        )
    return resistances


def character_immunities(character: Any) -> set[DamageType]:
    """Return all damage immunities active on a character."""

    return normalize_damage_type_set(getattr(character, "immunities", ()))


def character_vulnerabilities(character: Any) -> set[DamageType]:
    """Return all damage vulnerabilities active on a character."""

    return normalize_damage_type_set(getattr(character, "vulnerabilities", ()))


def has_damage_resistance(character: Any, damage_type: object) -> bool:
    """Return True if a character resists a damage type."""

    normalized_type = coerce_damage_type(damage_type)
    return normalized_type is not None and normalized_type in character_resistances(character)


def has_damage_immunity(character: Any, damage_type: object) -> bool:
    """Return True if a character is immune to a damage type."""

    normalized_type = coerce_damage_type(damage_type)
    return normalized_type is not None and normalized_type in character_immunities(character)


def has_damage_vulnerability(character: Any, damage_type: object) -> bool:
    """Return True if a character is vulnerable to a damage type."""

    normalized_type = coerce_damage_type(damage_type)
    return (
        normalized_type is not None
        and normalized_type in character_vulnerabilities(character)
    )


def apply_damage_modifiers(
    character: Any,
    damage: int,
    damage_type: object,
) -> int:
    """Apply immunity, resistance and vulnerability after roll modifiers."""

    normalized_damage = max(0, int(damage))
    normalized_type = coerce_damage_type(damage_type)
    if normalized_damage <= 0 or normalized_type is None:
        return normalized_damage
    if normalized_type in character_immunities(character):
        return 0
    if normalized_type in character_resistances(character):
        normalized_damage //= 2
    if normalized_type in character_vulnerabilities(character):
        normalized_damage *= 2
    return max(0, normalized_damage)


def _lookup_key(value: object) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())
