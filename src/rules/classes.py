"""Data-driven class definitions and 1-5 progression tables."""

from __future__ import annotations

from dataclasses import dataclass, field

from combat.class_features import ClassFeature, FeatureDefinition
from rules.registry import get_active_ruleset
from rules.ruleset import Ruleset
from rules.subclasses import (
    SubclassDefinition,
    get_subclass_definition,
    get_subclasses_for_class,
)


FULL_CASTER_SPELL_SLOTS: dict[int, dict[int, int]] = {
    1: {1: 2},
    2: {1: 3},
    3: {1: 4, 2: 2},
    4: {1: 4, 2: 3},
    5: {1: 4, 2: 3, 3: 2},
}


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


CLASS_DEFINITIONS: dict[str, ClassDefinition] = {
    _lookup_key("Fighter"): ClassDefinition(
        name="Fighter",
        hit_die=10,
        primary_abilities=("str", "dex", "con"),
        saving_throw_proficiencies=("str", "con"),
        armor_proficiencies=("light", "medium", "heavy", "shield"),
        weapon_proficiencies=("simple", "martial"),
        skill_choices=(
            "acrobatics",
            "animal_handling",
            "athletics",
            "history",
            "insight",
            "intimidation",
            "perception",
            "survival",
        ),
        subclass_level=3,
        level_features={
            1: (
                ClassFeature(
                    name="Second Wind",
                    level=1,
                    action_cost="bonus_action",
                    resource_cost="second_wind",
                    active_action="second_wind",
                    description="Recover hit points once per combat.",
                    implemented=False,
                ),
            ),
            2: (
                ClassFeature(
                    name="Action Surge",
                    level=2,
                    resource_cost="action_surge",
                    active_action="action_surge",
                    description="Class resource for taking an extra action.",
                    implemented=False,
                ),
            ),
            4: (
                ClassFeature(
                    name="Ability Score Improvement",
                    level=4,
                    passive_hooks=("on_level_up",),
                    description="Level 4 fighter progression choice.",
                    implemented=True,
                ),
            ),
            5: (
                ClassFeature(
                    name="Extra Attack",
                    level=5,
                    passive_hooks=("on_attack_action",),
                    description="Level 5 fighter progression note.",
                    implemented=False,
                ),
            ),
        },
    ),
    _lookup_key("Cleric"): ClassDefinition(
        name="Cleric",
        hit_die=8,
        primary_abilities=("wis", "con", "str"),
        saving_throw_proficiencies=("wis", "cha"),
        armor_proficiencies=("light", "medium", "shield"),
        weapon_proficiencies=("simple",),
        skill_choices=("history", "insight", "medicine", "persuasion", "religion"),
        spellcasting_progression=FULL_CASTER_SPELL_SLOTS,
        subclass_level=1,
        level_features={
            1: (
                ClassFeature(
                    name="Spellcasting",
                    level=1,
                    passive_hooks=("on_spellcasting",),
                    description="Cleric spellcasting progression.",
                    implemented=True,
                ),
            ),
            2: (
                ClassFeature(
                    name="Channel Divinity",
                    level=2,
                    resource_cost="channel_divinity",
                    active_action="channel_divinity",
                    description="Cleric class resource.",
                    implemented=False,
                ),
            ),
            3: (
                ClassFeature(
                    name="2nd-level Spells",
                    level=3,
                    passive_hooks=("on_spell_slots",),
                    description="Cleric spell slot progression note.",
                    implemented=True,
                ),
            ),
            4: (
                ClassFeature(
                    name="Ability Score Improvement",
                    level=4,
                    passive_hooks=("on_level_up",),
                    description="Level 4 cleric progression choice.",
                    implemented=True,
                ),
            ),
            5: (
                ClassFeature(
                    name="3rd-level Spells",
                    level=5,
                    passive_hooks=("on_spell_slots",),
                    description="Cleric spell slot progression note.",
                    implemented=True,
                ),
            ),
        },
    ),
    _lookup_key("Wizard"): ClassDefinition(
        name="Wizard",
        hit_die=6,
        primary_abilities=("int", "con", "dex"),
        saving_throw_proficiencies=("int", "wis"),
        armor_proficiencies=(),
        weapon_proficiencies=("dagger", "dart", "sling", "quarterstaff", "light_crossbow"),
        skill_choices=("arcana", "history", "insight", "investigation", "medicine", "religion"),
        spellcasting_progression=FULL_CASTER_SPELL_SLOTS,
        subclass_level=2,
        level_features={
            1: (
                ClassFeature(
                    name="Spellcasting",
                    level=1,
                    passive_hooks=("on_spellcasting",),
                    description="Wizard spellcasting progression.",
                    implemented=True,
                ),
                ClassFeature(
                    name="Arcane Recovery",
                    level=1,
                    resource_cost="arcane_recovery",
                    active_action="arcane_recovery",
                    description="Wizard class resource.",
                    implemented=False,
                ),
            ),
            3: (
                ClassFeature(
                    name="2nd-level Spells",
                    level=3,
                    passive_hooks=("on_spell_slots",),
                    description="Wizard spell slot progression note.",
                    implemented=True,
                ),
            ),
            4: (
                ClassFeature(
                    name="Ability Score Improvement",
                    level=4,
                    passive_hooks=("on_level_up",),
                    description="Level 4 wizard progression choice.",
                    implemented=True,
                ),
            ),
            5: (
                ClassFeature(
                    name="3rd-level Spells",
                    level=5,
                    passive_hooks=("on_spell_slots",),
                    description="Wizard spell slot progression note.",
                    implemented=True,
                ),
            ),
        },
    ),
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
        and class_definition.spellcasting_progression is not None
    )


def spell_slots_for_class_level(class_name: str | None, level: int) -> dict[int, int]:
    """Return spell slots for a class level."""

    class_definition = get_class_definition(class_name)
    if class_definition is None:
        return {}
    return class_definition.spell_slots_for_level(level)
