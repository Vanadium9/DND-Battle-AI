"""Spellcasting runtime helpers and MVP supported spell definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from combat.aoe import AoEShape, coerce_aoe_shape
from combat.abilities import SpellAbility, ability_modifier
from combat.damage import DamageType
from combat.resources import has_spell_slot, reset_spell_slots, spend_spell_slot
from rules.spellcasting_progression import (
    get_max_spell_level_for_progression,
    get_spell_slots_for_progression,
    get_spellcasting_ability_for_class,
    get_spellcasting_type_for_class,
)


@dataclass(frozen=True)
class SpellDefinition:
    """Data-driven definition for a supported spell."""

    name: str
    spell_level: int
    classes: tuple[str, ...]
    range: int
    action_cost: str = "action"
    target_type: str = "enemy"
    damage: int | str | None = None
    healing: int | str | None = None
    damage_type: DamageType | str | None = None
    save_ability: str | None = None
    save_half_damage: bool = False
    concentration: bool = False
    upcast_damage_per_level: int | str | None = None
    upcast_healing_per_level: int | str | None = None
    school: str | None = None
    area_shape: AoEShape | str | None = None
    area_size: int = 0
    ac_bonus: int = 0
    duration: str | None = None
    implemented: bool = True

    def to_ability(self, casting_level: int | None = None) -> SpellAbility:
        """Convert this definition into a combat spell ability."""

        return SpellAbility(
            name=self.name,
            description=f"Supported {self.name} spell.",
            range=self.range,
            spell_level=self.spell_level,
            casting_level=casting_level,
            action_cost=self.action_cost,
            target_type=self.target_type,
            damage=self.damage,
            healing=self.healing,
            damage_type=self.damage_type,
            save_ability=self.save_ability,
            save_half_damage=self.save_half_damage,
            concentration=self.concentration,
            upcast_damage_per_level=self.upcast_damage_per_level,
            upcast_healing_per_level=self.upcast_healing_per_level,
            school=self.school,
            area_shape=self.area_shape,
            area_size=self.area_size,
            ac_bonus=self.ac_bonus,
            duration=self.duration,
        )


from combat.spells.cleric_spells import (  # noqa: E402
    CLERIC_DEFAULT_CANTRIPS,
    CLERIC_DEFAULT_PREPARED_SPELLS,
    CLERIC_SPELLS,
)
from combat.spells.wizard_spells import (  # noqa: E402
    WIZARD_DEFAULT_CANTRIPS,
    WIZARD_DEFAULT_PREPARED_SPELLS,
    WIZARD_SPELLS,
)

SUPPORTED_SPELLS: tuple[SpellDefinition, ...] = (*WIZARD_SPELLS, *CLERIC_SPELLS)

DEFAULT_CANTRIPS: dict[str, tuple[str, ...]] = {
    "cleric": CLERIC_DEFAULT_CANTRIPS,
    "wizard": WIZARD_DEFAULT_CANTRIPS,
}

DEFAULT_PREPARED_SPELLS: dict[str, tuple[str, ...]] = {
    "cleric": CLERIC_DEFAULT_PREPARED_SPELLS,
    "wizard": WIZARD_DEFAULT_PREPARED_SPELLS,
}


def configure_spellcasting(character: Any) -> Any:
    """Populate spellcasting fields derived from class, level and stats."""

    class_name = getattr(character, "class_name", None)
    spellcasting_type = get_spellcasting_type_for_class(class_name)
    spellcasting_ability = get_spellcasting_ability_for_class(class_name)
    if spellcasting_type is None or spellcasting_ability is None:
        character.spellcasting = False
        return character

    slots = get_spell_slots_for_progression(spellcasting_type, getattr(character, "level", 1))
    character.spellcasting = True
    character.spellcasting_ability = spellcasting_ability
    reset_spell_slots(character, slots)
    ability_bonus = ability_modifier(character.stats, spellcasting_ability)
    character.spell_save_dc = 8 + character.proficiency_bonus + ability_bonus
    character.spell_attack_bonus = character.proficiency_bonus + ability_bonus

    if not getattr(character, "cantrips", None):
        character.cantrips = default_cantrips_for_class(class_name, character.level)
    if not getattr(character, "prepared_spells", None):
        character.prepared_spells = default_prepared_spells_for_class(
            class_name,
            character.level,
        )
    if not getattr(character, "known_spells", None):
        character.known_spells = _unique_spells(
            [*character.cantrips, *character.prepared_spells]
        )
    sync_spell_abilities(character)
    return character


def sync_spell_abilities(character: Any) -> None:
    """Ensure cantrips and prepared spells are visible in legacy abilities."""

    for spell in _unique_spells([*getattr(character, "cantrips", ()), *getattr(character, "prepared_spells", ())]):
        if spell not in character.abilities:
            character.abilities.append(spell)


def get_supported_spell_definitions(
    class_name: str | None,
    level: int,
) -> tuple[SpellDefinition, ...]:
    """Return spells a builder may show for a class at a supported level."""

    spellcasting_type = get_spellcasting_type_for_class(class_name)
    max_spell_level = get_max_spell_level_for_progression(spellcasting_type, level)
    class_key = _lookup_key(class_name)
    return tuple(
        definition
        for definition in SUPPORTED_SPELLS
        if definition.implemented
        and definition.spell_level <= max_spell_level
        and (
            definition.spell_level == 0
            or max_spell_level >= definition.spell_level
        )
        and any(_lookup_key(spell_class) == class_key for spell_class in definition.classes)
    )


def get_spell_definition(name: str | SpellAbility | SpellDefinition) -> SpellDefinition | None:
    """Return a supported spell definition by name or ability object."""

    if isinstance(name, SpellDefinition):
        return name
    spell_name = getattr(name, "name", name)
    spell_key = _lookup_key(spell_name)
    for definition in SUPPORTED_SPELLS:
        if _lookup_key(definition.name) == spell_key:
            return definition
    return None


def validate_spell_selection(
    class_name: str | None,
    level: int,
    *,
    known_spells: tuple[str, ...] = (),
    prepared_spells: tuple[str, ...] = (),
    cantrips: tuple[str, ...] = (),
) -> None:
    """Validate builder spell selections against supported spells and level."""

    if not any((known_spells, prepared_spells, cantrips)):
        return
    allowed = {
        _lookup_key(definition.name): definition
        for definition in get_supported_spell_definitions(class_name, level)
    }
    if not allowed:
        raise ValueError(f"{class_name} does not support spellcasting at level {level}.")

    for spell_name in (*known_spells, *prepared_spells, *cantrips):
        if _lookup_key(spell_name) not in allowed:
            raise ValueError(f"Spell '{spell_name}' is not supported for {class_name} level {level}.")

    for spell_name in cantrips:
        if allowed[_lookup_key(spell_name)].spell_level != 0:
            raise ValueError(f"Spell '{spell_name}' is not a cantrip.")

    explicit_known = {_lookup_key(name) for name in known_spells}
    if explicit_known:
        for spell_name in prepared_spells:
            if _lookup_key(spell_name) not in explicit_known:
                raise ValueError(f"Prepared spell '{spell_name}' is not in known_spells.")


def resolve_spell_list(spell_names: tuple[str, ...]) -> list[SpellAbility]:
    """Convert supported spell names into SpellAbility instances."""

    spells: list[SpellAbility] = []
    for spell_name in spell_names:
        definition = get_spell_definition(spell_name)
        if definition is not None:
            spells.append(definition.to_ability())
    return spells


def default_cantrips_for_class(class_name: str | None, level: int) -> list[SpellAbility]:
    """Return fixed MVP cantrips for a class."""

    return _default_spells(class_name, level, DEFAULT_CANTRIPS)


def default_prepared_spells_for_class(class_name: str | None, level: int) -> list[SpellAbility]:
    """Return fixed MVP prepared spells for a class."""

    return _default_spells(class_name, level, DEFAULT_PREPARED_SPELLS)


def available_castable_spells(character: Any) -> list[SpellAbility]:
    """Return cantrips and prepared spells that can currently be cast."""

    if spell_system_available(character):
        spells = _unique_spells([
            *getattr(character, "cantrips", ()),
            *getattr(character, "prepared_spells", ()),
        ])
    else:
        spells = [
            ability
            for ability in getattr(character, "available_abilities", ())
            if isinstance(ability, SpellAbility)
        ]
    return [spell for spell in spells if spell.available and can_cast_spell(character, spell)]


def can_cast_spell(
    character: Any,
    spell: SpellAbility,
    cast_level: int | None = None,
) -> bool:
    """Return True if the character can spend resources for a spell."""

    if not spell.available:
        return False
    casting_level = spell_cast_level(spell, cast_level)
    if spell.spell_level <= 0:
        return True
    if not spell_system_available(character):
        return False
    return has_spell_slot(character, casting_level)


def spend_spell_resources(
    character: Any,
    spell: SpellAbility,
    cast_level: int | None = None,
) -> bool:
    """Spend the slot required to cast a spell."""

    casting_level = spell_cast_level(spell, cast_level)
    if spell.spell_level <= 0:
        return True
    return spend_spell_slot(character, casting_level)


def spell_cast_level(spell: SpellAbility, cast_level: int | None = None) -> int:
    """Return the level used for this cast, including upcast requests."""

    requested_level = cast_level if cast_level is not None else spell.casting_level
    if requested_level is None:
        requested_level = spell.spell_level
    return max(int(spell.spell_level), int(requested_level))


def spell_system_available(character: Any) -> bool:
    """Return True when full spellcasting state is active for a character."""

    return bool(getattr(character, "spellcasting", False))


def spell_aoe_shape(spell: SpellAbility) -> AoEShape | None:
    """Return the AoE shape for a spell, if any."""

    return coerce_aoe_shape(getattr(spell, "area_shape", None))


def spell_has_aoe(spell: SpellAbility) -> bool:
    """Return True if a spell uses a supported AoE template."""

    return spell_aoe_shape(spell) is not None and int(getattr(spell, "area_size", 0)) > 0


def spell_requires_target_cell(spell: SpellAbility) -> bool:
    """Return True if spell placement is selected by grid cell."""

    return spell_aoe_shape(spell) is AoEShape.RADIUS


def spell_requires_direction(spell: SpellAbility) -> bool:
    """Return True if spell placement is selected by direction."""

    return spell_aoe_shape(spell) in {AoEShape.CONE, AoEShape.LINE}


def spell_requires_concentration(spell: SpellAbility) -> bool:
    """Return True if a spell starts concentration when cast."""

    return bool(getattr(spell, "concentration", False))


def begin_spell_concentration(character: Any, spell: SpellAbility) -> bool:
    """Start concentration for a spell that requires it."""

    if not spell_requires_concentration(spell):
        return False
    from combat.conditions import start_concentration

    start_concentration(character, spell)
    return True


def can_target_spell(
    actor: Any,
    target: Any,
    spell: SpellAbility,
    *,
    distance: int,
    has_line_of_sight: bool,
    has_full_cover: bool,
) -> bool:
    """Return True if a spell may target a creature under map constraints."""

    if target is None:
        return False
    if not getattr(target, "is_alive", False):
        return False
    if distance > spell.range or not has_line_of_sight or has_full_cover:
        return False
    if spell.healing is not None:
        return target.team == actor.team and target.hp < target.max_hp
    if spell.target_type in {"ally", "self_or_ally"}:
        if target.team != actor.team:
            return False
        if spell.damage is None and spell.healing is None:
            return True
    if spell.target_type == "self" and target is not actor:
        return False
    if spell.target_type == "enemy" and target is actor:
        return False
    if spell.damage is not None:
        return target.team != actor.team
    return True


def _default_spells(
    class_name: str | None,
    level: int,
    defaults: dict[str, tuple[str, ...]],
) -> list[SpellAbility]:
    class_key = _lookup_key(class_name)
    names = defaults.get(class_key, ())
    allowed = {
        _lookup_key(definition.name): definition
        for definition in get_supported_spell_definitions(class_name, level)
    }
    return [
        allowed[_lookup_key(name)].to_ability()
        for name in names
        if _lookup_key(name) in allowed
    ]


def _unique_spells(spells: list[SpellAbility]) -> list[SpellAbility]:
    seen: set[str] = set()
    unique: list[SpellAbility] = []
    for spell in spells:
        key = _lookup_key(spell.name)
        if key in seen:
            continue
        seen.add(key)
        unique.append(spell)
    return unique


def _lookup_key(value: object) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())
