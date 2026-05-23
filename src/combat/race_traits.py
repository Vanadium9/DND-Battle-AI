"""Apply racial traits to combat characters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from combat.action_economy import reset_turn_resources
from rules.races import RaceDefinition, get_race_definition

if TYPE_CHECKING:
    from combat.models import Character


@dataclass
class RaceTraits:
    """Runtime copy of racial traits stored on a character."""

    name: str
    ability_score_bonuses: dict[str, int]
    speed: int
    size: str
    darkvision_range: int | None = None
    skill_proficiencies: tuple[str, ...] = ()
    weapon_proficiencies: tuple[str, ...] = ()
    saving_throw_advantages: tuple[str, ...] = ()
    damage_resistances: tuple[str, ...] = ()
    special_traits: tuple[str, ...] = ()
    halfling_lucky_enabled: bool = False


def traits_from_definition(definition: RaceDefinition) -> RaceTraits:
    """Create mutable runtime traits from a static race definition."""

    return RaceTraits(
        name=definition.name,
        ability_score_bonuses=dict(definition.ability_score_bonuses),
        speed=definition.speed,
        size=definition.size,
        darkvision_range=definition.darkvision_range,
        skill_proficiencies=tuple(definition.skill_proficiencies),
        weapon_proficiencies=tuple(definition.weapon_proficiencies),
        saving_throw_advantages=tuple(definition.saving_throw_advantages),
        damage_resistances=tuple(definition.damage_resistances),
        special_traits=tuple(definition.special_traits),
        halfling_lucky_enabled="Lucky" in definition.special_traits,
    )


def apply_race_traits(
    character: "Character",
    race_name: str | None = None,
    *,
    allow_custom_fallback: bool = False,
    override_speed: int | None = None,
) -> "Character":
    """Apply racial bonuses, speed, proficiencies and resistances to a character."""

    definition = get_race_definition(
        race_name or character.race_name,
        allow_custom_fallback=allow_custom_fallback,
    )
    character.race_name = definition.name
    character.race_traits = traits_from_definition(definition)
    character.size = character.race_traits.size
    _apply_ability_score_bonuses(character, character.race_traits.ability_score_bonuses)
    character.speed = override_speed if override_speed is not None else character.race_traits.speed
    _apply_weapon_proficiencies(character)
    reset_turn_resources(character)
    return character


def weapon_is_racially_proficient(character: "Character", weapon_name: str) -> bool:
    traits = getattr(character, "race_traits", None)
    if traits is None:
        return False
    weapon_key = _normalization_key(weapon_name)
    return any(
        _normalization_key(proficiency) == weapon_key
        for proficiency in traits.weapon_proficiencies
    )


def has_damage_resistance(character: "Character", damage_type: str | None) -> bool:
    if not damage_type:
        return False
    traits = getattr(character, "race_traits", None)
    if traits is None:
        return False
    damage_key = _normalization_key(damage_type)
    return any(
        _normalization_key(resistance) == damage_key
        for resistance in traits.damage_resistances
    )


def apply_damage_resistance(
    character: "Character",
    damage: int,
    damage_type: str | None,
) -> int:
    """Reduce damage when a race grants resistance to its damage type."""

    normalized_damage = max(0, int(damage))
    if has_damage_resistance(character, damage_type):
        return normalized_damage // 2
    return normalized_damage


def use_halfling_lucky(character: "Character", roll: int, reroll: int) -> int:
    """Use the simplified Halfling Lucky feature flag for one natural 1 reroll."""

    traits = getattr(character, "race_traits", None)
    if traits is None or not traits.halfling_lucky_enabled or roll != 1:
        return roll
    return reroll


def _apply_ability_score_bonuses(
    character: "Character",
    bonuses: dict[str, int],
) -> None:
    for ability_name, bonus in bonuses.items():
        normalized_ability = ability_name.lower()
        if not hasattr(character.stats, normalized_ability):
            continue
        current_value = getattr(character.stats, normalized_ability)
        setattr(character.stats, normalized_ability, current_value + int(bonus))


def _apply_weapon_proficiencies(character: "Character") -> None:
    for weapon in character.weapons:
        if not weapon.proficient and weapon_is_racially_proficient(character, weapon.name):
            weapon.proficient = True


def _normalization_key(value: str) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())
