"""Character builder helpers backed by the ruleset registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from combat.abilities import SpellAbility, WeaponAttack
from combat.items import ItemDefinition
from combat.models import Character, Position, Stats, Team

if TYPE_CHECKING:
    from rules.classes import ClassDefinition
    from rules.ruleset import Ruleset
    from rules.subclasses import SubclassDefinition


@dataclass(frozen=True)
class CharacterBuildRequest:
    """Input DTO for creating a combat character from supported rules."""

    name: str
    class_name: str
    level: int = 1
    subclass_name: str | None = None
    stats: Stats = field(default_factory=Stats)
    team: Team = Team.PLAYERS
    hp: int = 10
    max_hp: int = 10
    ac: int = 12
    position: Position = field(default_factory=Position)
    speed: int = 3
    experience: int = 0
    weapons: tuple[WeaponAttack, ...] = ()
    fighting_style: str | None = None
    wearing_armor: bool = False
    known_spells: tuple[str, ...] = ()
    prepared_spells: tuple[str, ...] = ()
    cantrips: tuple[str, ...] = ()
    inventory: tuple[str | ItemDefinition, ...] = ()


def supported_class_options(
    ruleset: Ruleset | None = None,
) -> tuple[ClassDefinition, ...]:
    """Return classes that a builder UI/CLI may show."""

    from rules.classes import get_supported_class_definitions
    from rules.registry import get_active_ruleset

    return get_supported_class_definitions(ruleset or get_active_ruleset())


def supported_subclass_options(
    class_name: str,
    *,
    level: int | None = None,
    ruleset: Ruleset | None = None,
) -> tuple[SubclassDefinition, ...]:
    """Return subclasses that may be selected for the class and level."""

    from rules.classes import get_supported_subclass_definitions
    from rules.registry import get_active_ruleset

    return get_supported_subclass_definitions(
        class_name,
        ruleset or get_active_ruleset(),
        level,
    )


def supported_spell_options(
    class_name: str,
    *,
    level: int,
) -> tuple[object, ...]:
    """Return supported spells that a builder UI/CLI may show."""

    from combat.spellcasting import get_supported_spell_definitions

    return get_supported_spell_definitions(class_name, level)


def supported_item_options() -> tuple[ItemDefinition, ...]:
    """Return implemented items that a builder UI/CLI may show."""

    from combat.inventory import get_supported_item_definitions

    return get_supported_item_definitions()


def validate_class_selection(
    class_name: str,
    subclass_name: str | None = None,
    *,
    level: int = 1,
    ruleset: Ruleset | None = None,
) -> None:
    """Validate a class/subclass selection against ruleset and subclass level."""

    from rules.classes import get_class_definition
    from rules.registry import get_active_ruleset

    active_ruleset = ruleset or get_active_ruleset()
    class_definition = get_class_definition(class_name)
    if class_definition is None or not active_ruleset.is_supported_content(
        "class",
        class_name,
    ):
        raise ValueError(active_ruleset.get_unsupported_reason("class", class_name))

    if subclass_name is None:
        return

    subclass_level = class_definition.subclass_level
    if subclass_level is not None and int(level) < subclass_level:
        raise ValueError(
            f"{class_definition.name} chooses a subclass at level {subclass_level}."
        )

    supported_names = {
        _lookup_key(definition.name)
        for definition in supported_subclass_options(
            class_definition.name,
            level=level,
            ruleset=active_ruleset,
        )
    }
    if _lookup_key(subclass_name) not in supported_names:
        reason_name = f"{class_definition.name}: {subclass_name}"
        raise ValueError(active_ruleset.get_unsupported_reason("subclass", reason_name))


def validate_spell_selection(
    class_name: str,
    *,
    level: int,
    known_spells: tuple[str, ...] = (),
    prepared_spells: tuple[str, ...] = (),
    cantrips: tuple[str, ...] = (),
) -> None:
    """Validate a spell selection for character creation."""

    from combat.spellcasting import validate_spell_selection as validate_spells

    validate_spells(
        class_name,
        level,
        known_spells=known_spells,
        prepared_spells=prepared_spells,
        cantrips=cantrips,
    )


def validate_item_selection(
    items: tuple[str | ItemDefinition, ...] = (),
) -> None:
    """Validate that selected inventory items are implemented."""

    from combat.inventory import validate_item_selection as validate_items

    validate_items(items)


def build_character(
    request: CharacterBuildRequest | None = None,
    **kwargs: object,
) -> Character:
    """Create a Character using data-driven class progression."""

    if request is None:
        request = CharacterBuildRequest(**kwargs)
    elif kwargs:
        raise ValueError("Pass either CharacterBuildRequest or keyword options, not both.")

    from combat.spellcasting import resolve_spell_list
    from rules.progression import get_proficiency_bonus, sync_character_progression

    validate_class_selection(
        request.class_name,
        request.subclass_name,
        level=request.level,
    )
    validate_spell_selection(
        request.class_name,
        level=request.level,
        known_spells=request.known_spells,
        prepared_spells=request.prepared_spells,
        cantrips=request.cantrips,
    )
    validate_item_selection(request.inventory)
    cantrips: list[SpellAbility] = resolve_spell_list(request.cantrips)
    prepared_spells: list[SpellAbility] = resolve_spell_list(request.prepared_spells)
    known_spells: list[SpellAbility] = resolve_spell_list(request.known_spells)
    from combat.inventory import resolve_inventory_items

    inventory = resolve_inventory_items(request.inventory)
    character = Character(
        name=request.name,
        hp=request.hp,
        max_hp=request.max_hp,
        ac=request.ac,
        position=request.position,
        speed=request.speed,
        stats=request.stats,
        team=request.team,
        class_name=request.class_name,
        subclass_name=request.subclass_name,
        level=request.level,
        experience=request.experience,
        proficiency_bonus=get_proficiency_bonus(request.level),
        fighting_style=request.fighting_style,
        wearing_armor=request.wearing_armor,
        weapons=list(request.weapons),
        known_spells=known_spells,
        prepared_spells=prepared_spells,
        cantrips=cantrips,
        inventory=inventory,
    )
    sync_character_progression(character)
    return character


def _lookup_key(value: object) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())
