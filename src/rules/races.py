"""Data-driven race definitions for the supported minimal ruleset."""

from __future__ import annotations

from dataclasses import dataclass, field
import warnings


@dataclass(frozen=True)
class RaceDefinition:
    """Static racial traits that can be applied to a character."""

    name: str
    ability_score_bonuses: dict[str, int] = field(default_factory=dict)
    speed: int = 3
    size: str = "Medium"
    darkvision_range: int | None = None
    skill_proficiencies: tuple[str, ...] = ()
    weapon_proficiencies: tuple[str, ...] = ()
    saving_throw_advantages: tuple[str, ...] = ()
    damage_resistances: tuple[str, ...] = ()
    special_traits: tuple[str, ...] = ()


SUPPORTED_RACES: dict[str, RaceDefinition] = {
    "human": RaceDefinition(
        name="Human",
        ability_score_bonuses={
            "str": 1,
            "dex": 1,
            "con": 1,
            "int": 1,
            "wis": 1,
            "cha": 1,
        },
        speed=3,
    ),
    "dwarf": RaceDefinition(
        name="Dwarf",
        ability_score_bonuses={"con": 2},
        speed=2,
        darkvision_range=6,
        weapon_proficiencies=("battleaxe", "handaxe", "light hammer", "warhammer"),
        damage_resistances=("poison",),
        saving_throw_advantages=("poison",),
        special_traits=("Dwarven Resilience",),
    ),
    "elf": RaceDefinition(
        name="Elf",
        ability_score_bonuses={"dex": 2},
        speed=3,
        darkvision_range=6,
        skill_proficiencies=("perception",),
        weapon_proficiencies=("longsword", "shortsword", "shortbow", "longbow"),
        special_traits=("Fey Ancestry", "Trance"),
    ),
    "halfling": RaceDefinition(
        name="Halfling",
        ability_score_bonuses={"dex": 2},
        speed=2,
        size="Small",
        special_traits=("Lucky", "Brave", "Halfling Nimbleness"),
    ),
}

CUSTOM_RACE = RaceDefinition(
    name="CustomRace",
    speed=3,
    special_traits=("Unsupported race fallback",),
)


def get_race_definition(
    race_name: str | None,
    allow_custom_fallback: bool = False,
) -> RaceDefinition:
    """Return a supported race, or CustomRace only for explicit importer fallback."""

    if race_name is None or not str(race_name).strip():
        if allow_custom_fallback:
            return _warn_custom_race("missing")
        raise ValueError("race_name is required")

    key = _race_key(race_name)
    definition = SUPPORTED_RACES.get(key)
    if definition is not None:
        return definition

    if allow_custom_fallback:
        return _warn_custom_race(str(race_name))

    raise ValueError(
        f"Unsupported race '{race_name}'. CustomRace fallback requires importer confirmation."
    )


def is_supported_race(race_name: str | None) -> bool:
    if race_name is None:
        return False
    return _race_key(race_name) in SUPPORTED_RACES


def _race_key(race_name: str) -> str:
    return str(race_name).strip().casefold().replace("_", " ")


def _warn_custom_race(race_name: str) -> RaceDefinition:
    warnings.warn(
        (
            f"Unsupported race '{race_name}' imported as CustomRace. "
            "Combat uses only generic race defaults."
        ),
        UserWarning,
        stacklevel=2,
    )
    return CUSTOM_RACE
