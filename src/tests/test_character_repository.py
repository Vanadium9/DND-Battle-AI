from pathlib import Path
from uuid import uuid4

import pytest

from character import (
    CharacterRepository,
    CharacterValidationError,
    InternalCharacter,
)


def test_character_repository_saves_character() -> None:
    repository = _repository()
    character = _valid_character(id="")

    saved = repository.save_character(character)

    assert saved.id
    assert (repository.storage_dir / f"{saved.id}.json").exists()


def test_character_repository_loads_character() -> None:
    repository = _repository()
    saved = repository.save_character(_valid_character(name="Elaine"))

    loaded = repository.get_character(saved.id)

    assert loaded == saved
    assert loaded is not None
    assert loaded.name == "Elaine"


def test_character_repository_lists_characters() -> None:
    repository = _repository()
    fighter = repository.save_character(_valid_character(id="fighter-1", name="Zed"))
    wizard = repository.save_character(
        _valid_character(
            id="wizard-1",
            name="Ada",
            class_name="Wizard",
            subclass_name="School of Evocation",
            race_name="Elf",
            spell_slots={"1": 2},
            spells=({"name": "Magic Missile", "level": 1},),
            prepared_spells=("Magic Missile",),
        )
    )

    characters = repository.list_characters()

    assert [character.id for character in characters] == [wizard.id, fighter.id]


def test_character_repository_rejects_invalid_character() -> None:
    repository = _repository()
    invalid = _valid_character(class_name="Barbarian")

    with pytest.raises(CharacterValidationError):
        repository.save_character(invalid)

    assert repository.list_characters() == []


def test_character_repository_duplicates_character() -> None:
    repository = _repository()
    saved = repository.save_character(_valid_character(name="Hero"))

    duplicate = repository.duplicate_character(saved.id)

    assert duplicate.id != saved.id
    assert duplicate.name == "Hero Copy"
    assert repository.get_character(duplicate.id) == duplicate


def _repository() -> CharacterRepository:
    return CharacterRepository(
        Path("checkpoints") / f"test_characters_{uuid4().hex}",
    )


def _valid_character(
    *,
    id: str = "hero-1",
    name: str = "Hero",
    class_name: str = "Fighter",
    subclass_name: str | None = "Champion",
    race_name: str = "Human",
    spell_slots: dict[str, int] | None = None,
    spells: tuple[dict[str, object] | str, ...] = (),
    prepared_spells: tuple[str, ...] = (),
) -> InternalCharacter:
    return InternalCharacter(
        id=id,
        name=name,
        class_name=class_name,
        subclass_name=subclass_name,
        level=3,
        experience=900,
        race_name=race_name,
        role="MELEE_DAMAGE",
        stats={
            "str": 16,
            "dex": 12,
            "con": 14,
            "int": 10,
            "wis": 10,
            "cha": 10,
        },
        hp=28,
        ac=16,
        speed=30,
        proficiency_bonus=2,
        weapons=(
            {
                "name": "Longsword",
                "range": 1,
                "damage": "1d8",
                "damage_type": "slashing",
            },
        ),
        armor={"name": "Chain Mail", "ac": 16},
        class_features=("Fighting Style", "Second Wind", "Action Surge"),
        subclass_features=("Improved Critical",),
        race_traits={"speed": 30, "size": "Medium"},
        feats=("Ability Score Improvement",),
        spells=spells,
        prepared_spells=prepared_spells,
        spell_slots=spell_slots or {},
        resources={"Second Wind": 1, "Action Surge": 1},
        inventory=({"name": "Potion of Healing", "quantity": 1},),
        resistances=(),
        immunities=(),
        vulnerabilities=(),
    )
