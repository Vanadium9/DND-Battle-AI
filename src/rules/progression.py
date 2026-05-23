"""Character progression rules for levels 1 through 5."""

from __future__ import annotations

from typing import TYPE_CHECKING

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

FULL_CASTER_SPELL_SLOTS: dict[int, dict[int, int]] = {
    1: {1: 2},
    2: {1: 3},
    3: {1: 4, 2: 2},
    4: {1: 4, 2: 3},
    5: {1: 4, 2: 3, 3: 2},
}

SPELLCASTER_CLASSES = {"cleric", "wizard"}

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

    if is_spellcaster(character):
        spell_slots = spell_slots_for_level(character.level)
        character.spellcasting = True
        character.spell_slots = dict(spell_slots)
        character.spell_slots_remaining = dict(spell_slots)

    return character


def build_class_features(
    class_name: str | None,
    level: int,
    subclass_name: str | None = None,
) -> list[ClassFeature]:
    """Build class features available at a supported level."""

    normalized_class = _normalized_class_name(class_name)
    if normalized_class is None:
        return []

    normalized_level = _clamp_level(level)
    definitions = _feature_definitions(normalized_class, subclass_name)
    return [
        feature
        for feature in definitions
        if feature.required_level <= normalized_level
    ]


def build_class_resources(
    features: list[ClassFeature],
    existing_resources: dict[str, Resource] | None = None,
) -> dict[str, Resource]:
    """Build resources required by the active class features."""

    from combat.class_features import Resource

    existing_resources = existing_resources or {}
    resources: dict[str, Resource] = {}
    for feature in features:
        if feature.resource_name is None:
            continue
        max_uses = CLASS_RESOURCE_MAX_USES.get(feature.resource_name, 1)
        existing = existing_resources.get(feature.resource_name)
        if existing is not None and existing.max_uses == max_uses:
            resources[feature.resource_name] = existing
            existing.reset()
        else:
            resources[feature.resource_name] = Resource(
                name=feature.resource_name,
                max_uses=max_uses,
            )
    return resources


def is_spellcaster(character: "Character") -> bool:
    """Return True if progression should maintain spell slots for this character."""

    class_name = _normalized_class_name(getattr(character, "class_name", None))
    if class_name is not None and class_name.casefold() in SPELLCASTER_CLASSES:
        return True
    return bool(getattr(character, "spellcasting", False))


def spell_slots_for_level(level: int) -> dict[int, int]:
    """Return simplified full-caster spell slots for levels 1-5."""

    return dict(FULL_CASTER_SPELL_SLOTS[_clamp_level(level)])


def _feature_definitions(
    class_name: str,
    subclass_name: str | None,
) -> list[ClassFeature]:
    normalized_class = class_name.casefold()
    normalized_subclass = (subclass_name or "").casefold()

    if normalized_class == "fighter":
        features = [
            _class_feature(
                name="Second Wind",
                description="Recover hit points once per combat.",
                resource_name="second_wind",
                required_level=1,
            ),
            _class_feature(
                name="Action Surge",
                description="Class resource for taking an extra action.",
                resource_name="action_surge",
                required_level=2,
            ),
            _class_feature(
                name="Ability Score Improvement",
                description="Level 4 fighter progression note.",
                required_level=4,
            ),
            _class_feature(
                name="Extra Attack",
                description="Level 5 fighter progression note.",
                required_level=5,
            ),
        ]
        if normalized_subclass == "champion":
            features.append(
                _class_feature(
                    name="Improved Critical",
                    description="Champion feature saved for future attack logic.",
                    required_level=3,
                )
            )
        return features

    if normalized_class == "cleric":
        features = [
            _class_feature(
                name="Spellcasting",
                description="Cleric spellcasting progression.",
                required_level=1,
            ),
            _class_feature(
                name="Channel Divinity",
                description="Cleric class resource.",
                resource_name="channel_divinity",
                required_level=2,
            ),
            _class_feature(
                name="2nd-level Spells",
                description="Cleric spell slot progression note.",
                required_level=3,
            ),
            _class_feature(
                name="Ability Score Improvement",
                description="Level 4 cleric progression note.",
                required_level=4,
            ),
            _class_feature(
                name="3rd-level Spells",
                description="Cleric spell slot progression note.",
                required_level=5,
            ),
        ]
        if normalized_subclass == "life domain":
            features.append(
                _class_feature(
                    name="Disciple of Life",
                    description="Life Domain feature note.",
                    required_level=1,
                )
            )
        return features

    if normalized_class == "wizard":
        features = [
            _class_feature(
                name="Spellcasting",
                description="Wizard spellcasting progression.",
                required_level=1,
            ),
            _class_feature(
                name="Arcane Recovery",
                description="Wizard class resource.",
                resource_name="arcane_recovery",
                required_level=1,
            ),
            _class_feature(
                name="2nd-level Spells",
                description="Wizard spell slot progression note.",
                required_level=3,
            ),
            _class_feature(
                name="Ability Score Improvement",
                description="Level 4 wizard progression note.",
                required_level=4,
            ),
            _class_feature(
                name="3rd-level Spells",
                description="Wizard spell slot progression note.",
                required_level=5,
            ),
        ]
        if normalized_subclass == "school of evocation":
            features.extend(
                [
                    _class_feature(
                        name="Evocation Savant",
                        description="School of Evocation feature note.",
                        required_level=2,
                    ),
                    _class_feature(
                        name="Sculpt Spells",
                        description="School of Evocation feature note.",
                        required_level=2,
                    ),
                ]
            )
        return features

    return []


def _class_feature(
    name: str,
    description: str = "",
    resource_name: str | None = None,
    required_level: int = 1,
) -> "ClassFeature":
    from combat.class_features import ClassFeature

    return ClassFeature(
        name=name,
        description=description,
        resource_name=resource_name,
        required_level=required_level,
    )


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
