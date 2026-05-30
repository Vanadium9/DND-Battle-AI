import pytest

from ui.screens.character_builder_screen import (
    CharacterBuilderData,
    build_internal_character_from_builder,
    validate_builder_options,
)


def test_builder_calculates_fighter_derived_values() -> None:
    character = build_internal_character_from_builder(
        CharacterBuilderData(
            id="fighter",
            name="Fighter",
            race_name="Human",
            class_name="Fighter",
            subclass_name="Champion",
            level=5,
            stats={
                "str": 16,
                "dex": 14,
                "con": 14,
                "int": 10,
                "wis": 10,
                "cha": 10,
            },
            fighting_style="Defense",
            weapons=("Greatsword",),
            armor_name="Chain Mail",
        )
    )

    assert character.proficiency_bonus == 3
    assert character.hp > 0
    assert character.ac == 17
    assert character.resources["second_wind"] == 1
    assert character.resources["action_surge"] == 1
    assert "Extra Attack" in character.class_features


def test_builder_rejects_fireball_for_wizard_level_one() -> None:
    with pytest.raises(ValueError, match="Fireball"):
        validate_builder_options(
            CharacterBuilderData(
                name="Wizard",
                race_name="Elf",
                class_name="Wizard",
                level=1,
                prepared_spells=("Fireball",),
            )
        )


def test_builder_rejects_unsupported_class_race_subclass_and_item() -> None:
    with pytest.raises(ValueError, match="Class"):
        validate_builder_options(
            CharacterBuilderData(name="Bad", class_name="Barbarian")
        )

    with pytest.raises(ValueError, match="Race"):
        validate_builder_options(
            CharacterBuilderData(name="Bad", race_name="Tiefling")
        )

    with pytest.raises(ValueError, match="Unsupported subclass"):
        validate_builder_options(
            CharacterBuilderData(
                name="Bad",
                class_name="Fighter",
                subclass_name="Champion",
                level=1,
            )
        )

    with pytest.raises(ValueError, match="Unsupported item"):
        validate_builder_options(
            CharacterBuilderData(
                name="Bad",
                inventory=({"name": "Deck of Many Things", "quantity": 1},),
            )
        )


def test_builder_allows_supported_cleric_spells_and_slots() -> None:
    character = build_internal_character_from_builder(
        CharacterBuilderData(
            id="cleric",
            name="Cleric",
            race_name="Dwarf",
            class_name="Cleric",
            subclass_name="Life Domain",
            level=3,
            stats={
                "str": 12,
                "dex": 10,
                "con": 14,
                "int": 10,
                "wis": 16,
                "cha": 10,
            },
            prepared_spells=("Cure Wounds", "Healing Word"),
            weapons=("Mace",),
            armor_name="Scale Mail",
        )
    )

    assert character.spell_slots == {"1": 4, "2": 2}
    assert character.spell_save_dc == 13
    assert character.spell_attack_bonus == 5
    assert "Channel Divinity" in character.class_features
