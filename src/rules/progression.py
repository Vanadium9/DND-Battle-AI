"""Character progression rules for levels 1 through 5."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rules.spellcasting_progression import FULL_CASTER, FULL_CASTER_SPELL_SLOTS

if TYPE_CHECKING:
    from combat.class_features import ClassFeature, Resource
    from combat.models import Character


MIN_SUPPORTED_LEVEL = 1
MAX_SUPPORTED_LEVEL = 5

XP_THRESHOLDS: dict[int, int] = {
    1: 0,
    2: 300,
    3: 900,
    4: 2700,
    5: 6500,
}

PROFICIENCY_BONUS_BY_LEVEL: dict[int, int] = {
    1: 2,
    2: 2,
    3: 2,
    4: 2,
    5: 3,
}

BASE_COMMON_ACTIONS: tuple[str, ...] = (
    "move",
    "attack",
    "cast_spell",
    "dash",
    "disengage",
    "dodge",
    "help",
    "hide",
    "search",
    "use_object",
    "ready",
    "grapple",
    "shove",
    "stabilize",
    "improvised_action",
    "opportunity_attack",
    "end_turn",
)

CLASS_RESOURCE_MAX_USES: dict[str, int] = {
    "action_surge": 1,
    "arcane_recovery": 1,
    "channel_divinity": 1,
    "second_wind": 1,
}


def get_level_for_xp(xp: int) -> int:
    """Return the supported character level for an XP total."""

    normalized_xp = max(0, int(xp))
    level = MIN_SUPPORTED_LEVEL
    for candidate_level, threshold in sorted(XP_THRESHOLDS.items()):
        if normalized_xp >= threshold:
            level = candidate_level
    return min(level, MAX_SUPPORTED_LEVEL)


def get_proficiency_bonus(level: int) -> int:
    """Return proficiency bonus for a supported level."""

    normalized_level = _clamp_level(level)
    return PROFICIENCY_BONUS_BY_LEVEL[normalized_level]


def can_level_up(character: "Character") -> bool:
    """Return True if stored XP supports a higher level after combat."""

    current_level = _clamp_level(getattr(character, "level", MIN_SUPPORTED_LEVEL))
    target_level = get_level_for_xp(getattr(character, "experience", 0))
    return current_level < target_level


def apply_level_up(character: "Character") -> "Character":
    """Apply all pending level-ups and recalculate derived progression data."""

    target_level = get_level_for_xp(getattr(character, "experience", 0))
    character.level = min(target_level, MAX_SUPPORTED_LEVEL)
    sync_character_progression(character)
    return character


def sync_character_progression(character: "Character") -> "Character":
    """Recalculate proficiency, class features, resources, actions and spell slots."""

    character.level = _clamp_level(getattr(character, "level", MIN_SUPPORTED_LEVEL))
    character.experience = max(0, int(getattr(character, "experience", 0)))
    character.proficiency_bonus = get_proficiency_bonus(character.level)
    character.common_actions = _merge_unique(
        BASE_COMMON_ACTIONS,
        getattr(character, "common_actions", ()),
    )

    class_name = _normalized_class_name(getattr(character, "class_name", None))
    if class_name is not None:
        character.class_features = build_class_features(
            class_name,
            character.level,
            getattr(character, "subclass_name", None),
        )
        character.resources = build_class_resources(
            character.class_features,
            getattr(character, "resources", {}),
        )

    from combat.class_features import apply_defense_fighting_style

    apply_defense_fighting_style(character)

    if is_spellcaster(character):
        from combat.spellcasting import configure_spellcasting

        configure_spellcasting(character)

    return character


def build_class_features(
    class_name: str | None,
    level: int,
    subclass_name: str | None = None,
) -> list[ClassFeature]:
    """Build class features available at a supported level."""

    from rules.classes import build_class_features as build_features_from_rules

    return build_features_from_rules(
        _normalized_class_name(class_name),
        _clamp_level(level),
        subclass_name,
    )


def build_class_resources(
    features: list[ClassFeature],
    existing_resources: dict[str, Resource] | None = None,
) -> dict[str, Resource]:
    """Build resources required by the active class features."""

    from combat.class_features import Resource, feature_resource_name

    existing_resources = existing_resources or {}
    resources: dict[str, Resource] = {}
    for feature in features:
        resource_name = feature_resource_name(feature)
        if resource_name is None:
            continue
        max_uses = CLASS_RESOURCE_MAX_USES.get(resource_name, 1)
        existing = existing_resources.get(resource_name)
        if existing is not None and existing.max_uses == max_uses:
            resources[resource_name] = existing
            existing.reset()
        else:
            resources[resource_name] = Resource(
                name=resource_name,
                max_uses=max_uses,
            )
    return resources


def is_spellcaster(character: "Character") -> bool:
    """Return True if progression should maintain spell slots for this character."""

    class_name = _normalized_class_name(getattr(character, "class_name", None))
    from rules.classes import class_uses_spellcasting

    if class_uses_spellcasting(class_name):
        return True
    return bool(getattr(character, "spellcasting", False))


def spell_slots_for_level(level: int) -> dict[int, int]:
    """Return simplified full-caster spell slots for levels 1-5."""

    from rules.spellcasting_progression import get_spell_slots_for_progression

    return get_spell_slots_for_progression(FULL_CASTER, _clamp_level(level))


def _normalized_class_name(class_name: str | None) -> str | None:
    if class_name is None:
        return None
    stripped = class_name.strip()
    return stripped or None


def _clamp_level(level: int) -> int:
    return max(MIN_SUPPORTED_LEVEL, min(MAX_SUPPORTED_LEVEL, int(level)))


def _merge_unique(
    primary: tuple[str, ...],
    secondary: tuple[str, ...] | list[str],
) -> list[str]:
    result: list[str] = []
    for action_name in (*primary, *tuple(secondary)):
        if action_name not in result:
            result.append(action_name)
    return result
