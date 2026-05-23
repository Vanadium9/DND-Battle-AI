"""Serializable character schema focused on progression metadata."""

from __future__ import annotations

from dataclasses import dataclass

from rules.progression import get_level_for_xp, get_proficiency_bonus
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
class CharacterSchema:
    """Minimal character schema used by importer-facing code."""

    name: str
    progression: CharacterProgressionSchema = CharacterProgressionSchema()
    race: CharacterRaceSchema = CharacterRaceSchema()

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
        )
