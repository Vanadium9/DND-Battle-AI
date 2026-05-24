"""Feat and ability score improvement definitions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rules.registry import get_active_ruleset
from rules.ruleset import Ruleset

if TYPE_CHECKING:
    from combat.models import Character


COMBAT_HOOK_NAMES: tuple[str, ...] = (
    "on_attack_roll",
    "on_damage_roll",
    "on_saving_throw",
    "on_ability_check",
    "on_turn_start",
    "on_turn_end",
)

ABILITY_SCORE_IMPROVEMENT_NAME = "Ability Score Improvement"
GRAPPLER_NAME = "Grappler"

ABILITY_ALIASES: dict[str, str] = {
    "str": "str",
    "strength": "str",
    "dex": "dex",
    "dexterity": "dex",
    "con": "con",
    "constitution": "con",
    "int": "int",
    "intelligence": "int",
    "wis": "wis",
    "wisdom": "wis",
    "cha": "cha",
    "charisma": "cha",
}


@dataclass(frozen=True)
class FeatDefinition:
    """Data-driven definition for a character feat or ASI choice."""

    name: str
    prerequisites: dict[str, Any] = field(default_factory=dict)
    stat_bonuses: dict[str, int] = field(default_factory=dict)
    passive_effects: tuple[str, ...] = ()
    active_effects: tuple[str, ...] = ()
    combat_hooks: dict[str, tuple[str, ...]] = field(default_factory=dict)
    implemented: bool = False

    def has_hook(self, hook_name: str) -> bool:
        """Return True when this feat declares a named combat hook."""

        return bool(self.combat_hooks.get(hook_name))


def _lookup_key(value: object) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


FEAT_DEFINITIONS: dict[str, FeatDefinition] = {
    _lookup_key(ABILITY_SCORE_IMPROVEMENT_NAME): FeatDefinition(
        name=ABILITY_SCORE_IMPROVEMENT_NAME,
        prerequisites={"min_level": 4},
        passive_effects=("stat_bonus_choice",),
        implemented=True,
    ),
    _lookup_key(GRAPPLER_NAME): FeatDefinition(
        name=GRAPPLER_NAME,
        prerequisites={"min_level": 4, "str": 13},
        passive_effects=("advantage_against_grappled_target",),
        combat_hooks={"on_attack_roll": ("grapple_advantage",)},
        implemented=True,
    ),
}


def get_feat_definition(name: str | FeatDefinition) -> FeatDefinition | None:
    """Return a known feat definition by display name."""

    if isinstance(name, FeatDefinition):
        return name
    return FEAT_DEFINITIONS.get(_lookup_key(name))


def is_feat_supported(
    name: str | FeatDefinition,
    ruleset: Ruleset | None = None,
    *,
    require_implemented: bool = True,
) -> bool:
    """Return True when a feat is both known and allowed by the active ruleset."""

    definition = get_feat_definition(name)
    if definition is None:
        return False
    if require_implemented and not definition.implemented:
        return False
    active_ruleset = ruleset or get_active_ruleset()
    return active_ruleset.is_supported_content("feat", definition.name)


def get_unsupported_feat_reason(
    name: str | FeatDefinition,
    ruleset: Ruleset | None = None,
) -> str:
    """Explain why a feat cannot be selected for a character."""

    definition = get_feat_definition(name)
    if definition is None:
        return f"Feat '{name}' is unknown to the feat registry."
    if not definition.implemented:
        return f"Feat '{definition.name}' is not implemented for combat."
    active_ruleset = ruleset or get_active_ruleset()
    return active_ruleset.get_unsupported_reason("feat", definition.name)


def get_supported_feat_definitions(
    character: "Character | None" = None,
    ruleset: Ruleset | None = None,
) -> tuple[FeatDefinition, ...]:
    """Return feats that a builder UI/CLI may show."""

    active_ruleset = ruleset or get_active_ruleset()
    supported: list[FeatDefinition] = []
    for definition in FEAT_DEFINITIONS.values():
        if not is_feat_supported(definition, active_ruleset):
            continue
        if character is not None and not validate_feat_prerequisites(character, definition):
            continue
        supported.append(definition)
    return tuple(supported)


def validate_feat_prerequisites(
    character: "Character",
    feat: str | FeatDefinition,
) -> bool:
    """Return True when a character satisfies a feat's prerequisites."""

    return not get_feat_prerequisite_failures(character, feat)


def get_feat_prerequisite_failures(
    character: "Character",
    feat: str | FeatDefinition,
) -> tuple[str, ...]:
    """Return human-readable prerequisite failures for a feat."""

    definition = get_feat_definition(feat)
    if definition is None:
        return (f"Unknown feat: {feat}",)

    failures: list[str] = []
    for key, expected in definition.prerequisites.items():
        normalized_key = str(key).strip().casefold()
        ability_name = _ability_from_prerequisite_key(normalized_key)
        if ability_name is not None:
            actual_score = int(getattr(character.stats, ability_name))
            required_score = int(expected)
            if actual_score < required_score:
                failures.append(
                    f"{definition.name} requires {ability_name.upper()} {required_score}."
                )
            continue

        if normalized_key in {"level", "min_level", "required_level"}:
            actual_level = int(getattr(character, "level", 1))
            required_level = int(expected)
            if actual_level < required_level:
                failures.append(f"{definition.name} requires level {required_level}.")
            continue

        if normalized_key in {"class", "class_name"}:
            if not _matches_string_choice(getattr(character, "class_name", None), expected):
                failures.append(f"{definition.name} requires class {expected}.")
            continue

        if normalized_key in {"subclass", "subclass_name"}:
            if not _matches_string_choice(getattr(character, "subclass_name", None), expected):
                failures.append(f"{definition.name} requires subclass {expected}.")
            continue

        if normalized_key in {"race", "race_name"}:
            if not _matches_string_choice(getattr(character, "race_name", None), expected):
                failures.append(f"{definition.name} requires race {expected}.")
            continue

        failures.append(f"{definition.name} has unsupported prerequisite '{key}'.")

    return tuple(failures)


def normalize_ability_name(ability_name: str) -> str:
    """Normalize a long or short ability name to the stored stat field."""

    key = str(ability_name).strip().casefold()
    try:
        return ABILITY_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Unknown ability score: {ability_name}") from exc


def _ability_from_prerequisite_key(prerequisite_key: str) -> str | None:
    if prerequisite_key.startswith("min_"):
        prerequisite_key = prerequisite_key[4:]
    return ABILITY_ALIASES.get(prerequisite_key)


def _matches_string_choice(actual: str | None, expected: Any) -> bool:
    if actual is None:
        return False
    actual_key = _lookup_key(actual)
    if isinstance(expected, str):
        return actual_key == _lookup_key(expected)
    if isinstance(expected, Iterable):
        return any(actual_key == _lookup_key(str(candidate)) for candidate in expected)
    return False

