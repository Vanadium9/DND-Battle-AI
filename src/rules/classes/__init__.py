"""Data-driven class definitions and 1-5 progression tables."""

from __future__ import annotations

from dataclasses import dataclass, field

from combat.class_features import ClassFeature, FeatureDefinition
from rules.registry import get_active_ruleset
from rules.ruleset import Ruleset
from rules.spellcasting_progression import get_spell_slots_for_progression
from rules.subclasses import (
    SubclassDefinition,
    get_subclass_definition,
    get_subclasses_for_class,
)


@dataclass(frozen=True)
class ClassDefinition:
    """Progression and proficiency data for one supported class."""

    name: str
    hit_die: int
    primary_abilities: tuple[str, ...]
    saving_throw_proficiencies: tuple[str, ...]
    armor_proficiencies: tuple[str, ...]
    weapon_proficiencies: tuple[str, ...]
    skill_choices: tuple[str, ...]
    level_features: dict[int, tuple[FeatureDefinition, ...]] = field(
        default_factory=dict
    )
    spellcasting_progression: dict[int, dict[int, int]] | None = None
    spellcasting_type: str | None = None
    spellcasting_ability: str | None = None
    subclass_level: int | None = None

    def features_for_level(self, level: int) -> tuple[FeatureDefinition, ...]:
        """Return class features available up to the requested level."""

        normalized_level = int(level)
        return tuple(
            feature
            for feature_level, features in sorted(self.level_features.items())
            if feature_level <= normalized_level
            for feature in features
        )

    def spell_slots_for_level(self, level: int) -> dict[int, int]:
        """Return spell slots granted at a class level."""

        if self.spellcasting_type is not None:
            return get_spell_slots_for_progression(self.spellcasting_type, level)
        if self.spellcasting_progression is None:
            return {}
        eligible_levels = [
            feature_level
            for feature_level in self.spellcasting_progression
            if feature_level <= int(level)
        ]
        if not eligible_levels:
            return {}
        return dict(self.spellcasting_progression[max(eligible_levels)])


def _lookup_key(value: object) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


from rules.classes.fighter import FIGHTER_DEFINITION  # noqa: E402
from rules.classes.cleric import CLERIC_DEFINITION  # noqa: E402
from rules.classes.wizard import WIZARD_DEFINITION  # noqa: E402


CLASS_DEFINITIONS: dict[str, ClassDefinition] = {
    _lookup_key(FIGHTER_DEFINITION.name): FIGHTER_DEFINITION,
    _lookup_key(CLERIC_DEFINITION.name): CLERIC_DEFINITION,
    _lookup_key(WIZARD_DEFINITION.name): WIZARD_DEFINITION,
}


def get_class_definition(class_name: str | None) -> ClassDefinition | None:
    """Return a class definition by name."""

    if class_name is None:
        return None
    return CLASS_DEFINITIONS.get(_lookup_key(class_name))


def get_supported_class_definitions(
    ruleset: Ruleset | None = None,
) -> tuple[ClassDefinition, ...]:
    """Return class definitions allowed by the active ruleset."""

    active_ruleset = ruleset or get_active_ruleset()
    return tuple(
        definition
        for definition in CLASS_DEFINITIONS.values()
        if active_ruleset.is_supported_content("class", definition.name)
    )


def get_supported_subclass_definitions(
    class_name: str | None,
    ruleset: Ruleset | None = None,
    level: int | None = None,
) -> tuple[SubclassDefinition, ...]:
    """Return subclass definitions allowed for a class by the active ruleset."""

    class_definition = get_class_definition(class_name)
    if class_definition is None:
        return ()
    active_ruleset = ruleset or get_active_ruleset()
    if level is not None and class_definition.subclass_level is not None:
        if int(level) < class_definition.subclass_level:
            return ()

    supported: list[SubclassDefinition] = []
    for subclass_definition in get_subclasses_for_class(class_definition.name):
        ruleset_name = f"{class_definition.name}: {subclass_definition.name}"
        if active_ruleset.is_supported_content("subclass", ruleset_name):
            supported.append(subclass_definition)
    return tuple(supported)


def build_class_features(
    class_name: str | None,
    level: int,
    subclass_name: str | None = None,
) -> list[ClassFeature]:
    """Build class and subclass features available at a supported level."""

    class_definition = get_class_definition(class_name)
    if class_definition is None:
        return []

    normalized_level = int(level)
    features = list(class_definition.features_for_level(normalized_level))

    if (
        subclass_name is not None
        and class_definition.subclass_level is not None
        and normalized_level >= class_definition.subclass_level
    ):
        subclass_definition = get_subclass_definition(
            class_definition.name,
            subclass_name,
        )
        if subclass_definition is not None:
            features.extend(subclass_definition.features_for_level(normalized_level))

    return [feature for feature in features if isinstance(feature, ClassFeature)]


def class_uses_spellcasting(class_name: str | None) -> bool:
    """Return True if a class has a spellcasting progression table."""

    class_definition = get_class_definition(class_name)
    return (
        class_definition is not None
        and (
            class_definition.spellcasting_type is not None
            or class_definition.spellcasting_progression is not None
        )
    )


def spellcasting_type_for_class(class_name: str | None) -> str | None:
    """Return the spellcasting progression type for a class definition."""

    class_definition = get_class_definition(class_name)
    if class_definition is None:
        return None
    return class_definition.spellcasting_type


def spellcasting_ability_for_class(class_name: str | None) -> str | None:
    """Return the spellcasting ability for a class definition."""

    class_definition = get_class_definition(class_name)
    if class_definition is None:
        return None
    return class_definition.spellcasting_ability


def spell_slots_for_class_level(class_name: str | None, level: int) -> dict[int, int]:
    """Return spell slots for a class level."""

    class_definition = get_class_definition(class_name)
    if class_definition is None:
        return {}
    return class_definition.spell_slots_for_level(level)
