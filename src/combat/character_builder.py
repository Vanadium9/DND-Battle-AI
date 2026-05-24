"""Character builder helpers backed by the ruleset registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from combat.abilities import WeaponAttack
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


def build_character(
    request: CharacterBuildRequest | None = None,
    **kwargs: object,
) -> Character:
    """Create a Character using data-driven class progression."""

    if request is None:
        request = CharacterBuildRequest(**kwargs)
    elif kwargs:
        raise ValueError("Pass either CharacterBuildRequest or keyword options, not both.")

    from rules.progression import get_proficiency_bonus, sync_character_progression

    validate_class_selection(
        request.class_name,
        request.subclass_name,
        level=request.level,
    )
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
        weapons=list(request.weapons),
    )
    sync_character_progression(character)
    return character


def _lookup_key(value: object) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())
