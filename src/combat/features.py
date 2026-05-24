"""Runtime helpers for ASI choices, feats and combat feature hooks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rules.feats import (
    ABILITY_SCORE_IMPROVEMENT_NAME,
    COMBAT_HOOK_NAMES,
    FeatDefinition,
    get_feat_definition,
    get_feat_prerequisite_failures,
    get_supported_feat_definitions,
    is_feat_supported,
    normalize_ability_name,
    validate_feat_prerequisites,
)
from rules.ruleset import Ruleset

if TYPE_CHECKING:
    from combat.models import Character, CombatState


DEFAULT_ABILITY_SCORE_CAP = 20


@dataclass(frozen=True)
class AbilityScoreImprovement:
    """A stored ASI selection applied to a character."""

    bonuses: dict[str, int]
    source: str = ABILITY_SCORE_IMPROVEMENT_NAME

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "bonuses",
            _normalize_asi_bonuses(self.bonuses),
        )


def apply_ability_score_improvement(
    character: "Character",
    bonuses: Mapping[str, int],
    *,
    stat_cap: int | None = DEFAULT_ABILITY_SCORE_CAP,
) -> "Character":
    """Apply a legal +2 or +1/+1 ASI choice to a character."""

    normalized_bonuses = validate_ability_score_improvement(
        character,
        bonuses,
        stat_cap=stat_cap,
    )
    for ability_name, increase in normalized_bonuses.items():
        setattr(character.stats, ability_name, getattr(character.stats, ability_name) + increase)

    if not hasattr(character, "ability_score_improvements"):
        character.ability_score_improvements = []
    character.ability_score_improvements.append(
        AbilityScoreImprovement(dict(normalized_bonuses))
    )
    return character


def validate_ability_score_improvement(
    character: "Character",
    bonuses: Mapping[str, int],
    *,
    stat_cap: int | None = DEFAULT_ABILITY_SCORE_CAP,
) -> dict[str, int]:
    """Validate and normalize an ASI bonus map without mutating the character."""

    normalized_bonuses = _normalize_asi_bonuses(bonuses)
    if stat_cap is not None:
        normalized_cap = int(stat_cap)
        for ability_name, increase in normalized_bonuses.items():
            current_score = int(getattr(character.stats, ability_name))
            if current_score + increase > normalized_cap:
                raise ValueError(
                    f"ASI would raise {ability_name.upper()} above cap {normalized_cap}."
                )
    return normalized_bonuses


def add_feat(
    character: "Character",
    feat_name: str,
    *,
    ruleset: Ruleset | None = None,
) -> "Character":
    """Add a supported implemented feat after ruleset and prerequisite checks."""

    definition = get_feat_definition(feat_name)
    if definition is None:
        raise ValueError(f"Feat '{feat_name}' is unknown.")
    if definition.name == ABILITY_SCORE_IMPROVEMENT_NAME:
        raise ValueError(
            "Ability Score Improvement must be applied with "
            "apply_ability_score_improvement()."
        )
    if not is_feat_supported(definition, ruleset):
        raise ValueError(f"Feat '{definition.name}' is not supported by the active ruleset.")
    failures = get_feat_prerequisite_failures(character, definition)
    if failures:
        raise ValueError("; ".join(failures))
    if character_has_feat(character, definition.name):
        raise ValueError(f"{character.name} already has feat '{definition.name}'.")

    if not hasattr(character, "feats"):
        character.feats = []
    character.feats.append(definition)
    return character


def get_supported_feats_for_builder(
    character: "Character",
    *,
    ruleset: Ruleset | None = None,
) -> tuple[FeatDefinition, ...]:
    """Return feat choices that a UI/CLI builder may display."""

    return get_supported_feat_definitions(character, ruleset)


def can_choose_asi_or_feat(character: "Character") -> bool:
    """Return True when a level-4 ASI/feat choice is currently available."""

    if int(getattr(character, "level", 1)) < 4:
        return False
    has_asi_feature = any(
        getattr(feature, "name", "") == ABILITY_SCORE_IMPROVEMENT_NAME
        for feature in getattr(character, "class_features", ())
    )
    class_name = getattr(character, "class_name", None)
    if not has_asi_feature and isinstance(class_name, str):
        has_asi_feature = class_name.strip().casefold() in {"fighter", "cleric", "wizard"}
    if not has_asi_feature:
        return False
    return not getattr(character, "ability_score_improvements", ()) and not getattr(
        character,
        "feats",
        (),
    )


def apply_level_four_choice(
    character: "Character",
    *,
    asi_bonuses: Mapping[str, int] | None = None,
    feat_name: str | None = None,
    stat_cap: int | None = DEFAULT_ABILITY_SCORE_CAP,
    ruleset: Ruleset | None = None,
) -> "Character":
    """Apply the level-4 progression choice as either ASI or a feat."""

    if not can_choose_asi_or_feat(character):
        raise ValueError(f"{character.name} has no available level-4 ASI/feat choice.")
    if (asi_bonuses is None) == (feat_name is None):
        raise ValueError("Choose exactly one of asi_bonuses or feat_name.")
    if asi_bonuses is not None:
        return apply_ability_score_improvement(character, asi_bonuses, stat_cap=stat_cap)
    return add_feat(character, str(feat_name), ruleset=ruleset)


def character_has_feat(character: "Character", feat_name: str) -> bool:
    """Return True when a character stores the named feat."""

    target = _lookup_key(feat_name)
    return any(_lookup_key(_feat_display_name(feat)) == target for feat in _raw_feats(character))


def get_active_feat_definitions(
    character: "Character",
    *,
    ruleset: Ruleset | None = None,
) -> tuple[FeatDefinition, ...]:
    """Return stored feats that may currently affect combat."""

    definitions: list[FeatDefinition] = []
    for feat in _raw_feats(character):
        definition = get_feat_definition(feat) if not isinstance(feat, FeatDefinition) else feat
        if definition is None or not definition.implemented:
            continue
        if not is_feat_supported(definition, ruleset):
            continue
        definitions.append(definition)
    return tuple(definitions)


def get_active_combat_hooks(
    character: "Character",
    *,
    ruleset: Ruleset | None = None,
) -> dict[str, tuple[str, ...]]:
    """Return active combat hooks exposed by implemented supported feats."""

    hooks: dict[str, list[str]] = {hook_name: [] for hook_name in COMBAT_HOOK_NAMES}
    for definition in get_active_feat_definitions(character, ruleset=ruleset):
        for hook_name in COMBAT_HOOK_NAMES:
            if definition.has_hook(hook_name):
                hooks[hook_name].append(definition.name)
    return {
        hook_name: tuple(feat_names)
        for hook_name, feat_names in hooks.items()
        if feat_names
    }


def attack_roll_advantage_state(
    character: "Character",
    target: "Character | None" = None,
    combat_state: "CombatState | None" = None,
    *,
    has_advantage: bool = False,
    has_disadvantage: bool = False,
) -> tuple[bool, bool]:
    """Allow active feats to modify attack advantage before dice are rolled."""

    for definition in get_active_feat_definitions(character):
        if not definition.has_hook("on_attack_roll"):
            continue
        for effect_name in definition.combat_hooks["on_attack_roll"]:
            if effect_name == "grapple_advantage" and _is_grappling_target(
                character,
                target,
                combat_state,
            ):
                has_advantage = True
    return has_advantage, has_disadvantage


def apply_combat_hook(
    character: "Character",
    hook_name: str,
    value: Any = None,
    **context: Any,
) -> Any:
    """Apply active implemented feat effects for a hook and return the value."""

    if hook_name not in COMBAT_HOOK_NAMES:
        raise ValueError(f"Unknown combat hook: {hook_name}")
    result = value
    for definition in get_active_feat_definitions(character):
        for effect_name in definition.combat_hooks.get(hook_name, ()):
            result = _apply_feat_effect(
                definition,
                effect_name,
                hook_name,
                result,
                character,
                context,
            )
    return result


def on_attack_roll(character: "Character", roll: int, **context: Any) -> int:
    """Hook for attack roll totals or kept d20 values."""

    return int(apply_combat_hook(character, "on_attack_roll", int(roll), **context))


def on_damage_roll(character: "Character", damage: int, **context: Any) -> int:
    """Hook for damage before target resistances are applied."""

    return max(0, int(apply_combat_hook(character, "on_damage_roll", int(damage), **context)))


def on_saving_throw(character: "Character", result: Any, **context: Any) -> Any:
    """Hook for saving throw results."""

    return apply_combat_hook(character, "on_saving_throw", result, **context)


def on_ability_check(character: "Character", result: Any, **context: Any) -> Any:
    """Hook for ability check results."""

    return apply_combat_hook(character, "on_ability_check", result, **context)


def on_turn_start(character: "Character", **context: Any) -> "Character":
    """Hook called at the start of a character's turn."""

    return apply_combat_hook(character, "on_turn_start", character, **context)


def on_turn_end(character: "Character", **context: Any) -> "Character":
    """Hook called at the end of a character's turn."""

    return apply_combat_hook(character, "on_turn_end", character, **context)


def _apply_feat_effect(
    definition: FeatDefinition,
    effect_name: str,
    hook_name: str,
    value: Any,
    character: "Character",
    context: Mapping[str, Any],
) -> Any:
    if definition.name == "Grappler" and effect_name == "grapple_advantage":
        return value
    return value


def _normalize_asi_bonuses(bonuses: Mapping[str, int]) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for ability_name, increase in bonuses.items():
        normalized_ability = normalize_ability_name(str(ability_name))
        normalized[normalized_ability] = normalized.get(normalized_ability, 0) + int(increase)

    values = tuple(normalized.values())
    is_plus_two = len(normalized) == 1 and values == (2,)
    is_two_plus_ones = len(normalized) == 2 and all(value == 1 for value in values)
    if not (is_plus_two or is_two_plus_ones):
        raise ValueError("ASI must be +2 to one ability or +1 to two abilities.")
    return normalized


def _is_grappling_target(
    character: "Character",
    target: "Character | None",
    combat_state: "CombatState | None",
) -> bool:
    if target is None:
        return False

    target_id = None
    actor_id = None
    if combat_state is not None:
        try:
            target_id = combat_state.characters.index(target)
        except ValueError:
            target_id = None
        try:
            actor_id = combat_state.characters.index(character)
        except ValueError:
            actor_id = None

    if target_id is not None and getattr(character, "grappling_target_id", None) == target_id:
        return True
    if actor_id is not None and getattr(target, "grappled_by", None) == actor_id:
        return True
    return False


def _raw_feats(character: "Character") -> tuple[Any, ...]:
    feats = getattr(character, "feats", ())
    if feats is None:
        return ()
    return tuple(feats)


def _feat_display_name(feat: Any) -> str:
    if isinstance(feat, FeatDefinition):
        return feat.name
    return str(getattr(feat, "name", feat))


def _lookup_key(value: object) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())
