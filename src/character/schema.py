"""Serializable character schema focused on progression metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from rules.progression import get_level_for_xp, get_proficiency_bonus
from rules.feats import FeatDefinition, get_feat_definition
from rules.races import get_race_definition


@dataclass(frozen=True)
class CharacterProgressionSchema:
    """Level, XP and class metadata for importing or exporting a character."""

    level: int = 1
    experience: int = 0
    proficiency_bonus: int = 2
    class_name: str | None = None
    subclass_name: str | None = None

    @classmethod
    def from_xp(
        cls,
        experience: int,
        class_name: str | None = None,
        subclass_name: str | None = None,
    ) -> "CharacterProgressionSchema":
        level = get_level_for_xp(experience)
        return cls(
            level=level,
            experience=max(0, int(experience)),
            proficiency_bonus=get_proficiency_bonus(level),
            class_name=class_name,
            subclass_name=subclass_name,
        )


@dataclass(frozen=True)
class CharacterRaceSchema:
    """Race metadata for importer/exporter code."""

    race_name: str | None = None
    size: str = "Medium"
    speed: int | None = None
    darkvision_range: int | None = None
    skill_proficiencies: tuple[str, ...] = ()
    weapon_proficiencies: tuple[str, ...] = ()
    saving_throw_advantages: tuple[str, ...] = ()
    damage_resistances: tuple[str, ...] = ()
    special_traits: tuple[str, ...] = ()

    @classmethod
    def from_imported_name(cls, race_name: str | None) -> "CharacterRaceSchema":
        definition = get_race_definition(race_name, allow_custom_fallback=True)
        return cls(
            race_name=definition.name,
            size=definition.size,
            speed=definition.speed,
            darkvision_range=definition.darkvision_range,
            skill_proficiencies=tuple(definition.skill_proficiencies),
            weapon_proficiencies=tuple(definition.weapon_proficiencies),
            saving_throw_advantages=tuple(definition.saving_throw_advantages),
            damage_resistances=tuple(definition.damage_resistances),
            special_traits=tuple(definition.special_traits),
        )


@dataclass(frozen=True)
class CharacterFeatSchema:
    """Feat metadata exposed to importer/exporter and builder code."""

    name: str
    prerequisites: dict[str, Any]
    stat_bonuses: dict[str, int]
    passive_effects: tuple[str, ...]
    active_effects: tuple[str, ...]
    combat_hooks: dict[str, tuple[str, ...]]
    implemented: bool = False

    @classmethod
    def from_value(cls, feat: object) -> "CharacterFeatSchema":
        definition = _coerce_feat_definition(feat)
        if definition is not None:
            return cls(
                name=definition.name,
                prerequisites=dict(definition.prerequisites),
                stat_bonuses=dict(definition.stat_bonuses),
                passive_effects=tuple(definition.passive_effects),
                active_effects=tuple(definition.active_effects),
                combat_hooks={
                    hook_name: tuple(effect_names)
                    for hook_name, effect_names in definition.combat_hooks.items()
                },
                implemented=definition.implemented,
            )
        return cls(
            name=str(getattr(feat, "name", feat)),
            prerequisites={},
            stat_bonuses={},
            passive_effects=(),
            active_effects=(),
            combat_hooks={},
            implemented=False,
        )


@dataclass(frozen=True)
class AbilityScoreImprovementSchema:
    """Serialized ASI selection."""

    bonuses: dict[str, int]
    source: str = "Ability Score Improvement"

    @classmethod
    def from_value(cls, asi: object) -> "AbilityScoreImprovementSchema":
        bonuses = getattr(asi, "bonuses", asi)
        if not isinstance(bonuses, Mapping):
            bonuses = {}
        return cls(
            bonuses={str(ability): int(value) for ability, value in bonuses.items()},
            source=str(getattr(asi, "source", "Ability Score Improvement")),
        )


@dataclass(frozen=True)
class CharacterSchema:
    """Minimal character schema used by importer-facing code."""

    name: str
    progression: CharacterProgressionSchema = CharacterProgressionSchema()
    race: CharacterRaceSchema = CharacterRaceSchema()
    feats: tuple[CharacterFeatSchema, ...] = ()
    ability_score_improvements: tuple[AbilityScoreImprovementSchema, ...] = ()

    @classmethod
    def from_character(cls, character: object) -> "CharacterSchema":
        level = int(getattr(character, "level", 1))
        race_traits = getattr(character, "race_traits", None)
        return cls(
            name=str(getattr(character, "name", "")),
            progression=CharacterProgressionSchema(
                level=level,
                experience=int(getattr(character, "experience", 0)),
                proficiency_bonus=int(
                    getattr(character, "proficiency_bonus", get_proficiency_bonus(level))
                ),
                class_name=getattr(character, "class_name", None),
                subclass_name=getattr(character, "subclass_name", None),
            ),
            race=CharacterRaceSchema(
                race_name=getattr(character, "race_name", None),
                size=str(getattr(character, "size", "Medium")),
                speed=getattr(race_traits, "speed", None),
                darkvision_range=getattr(race_traits, "darkvision_range", None),
                skill_proficiencies=tuple(
                    getattr(race_traits, "skill_proficiencies", ())
                ),
                weapon_proficiencies=tuple(
                    getattr(race_traits, "weapon_proficiencies", ())
                ),
                saving_throw_advantages=tuple(
                    getattr(race_traits, "saving_throw_advantages", ())
                ),
                damage_resistances=tuple(
                    getattr(race_traits, "damage_resistances", ())
                ),
                special_traits=tuple(getattr(race_traits, "special_traits", ())),
            ),
            feats=tuple(
                CharacterFeatSchema.from_value(feat)
                for feat in getattr(character, "feats", ())
            ),
            ability_score_improvements=tuple(
                AbilityScoreImprovementSchema.from_value(asi)
                for asi in getattr(character, "ability_score_improvements", ())
            ),
        )


def _coerce_feat_definition(feat: object) -> FeatDefinition | None:
    if isinstance(feat, FeatDefinition):
        return feat
    if isinstance(feat, str):
        return get_feat_definition(feat)
    name = getattr(feat, "name", None)
    if isinstance(name, str):
        return get_feat_definition(name)
    return None
