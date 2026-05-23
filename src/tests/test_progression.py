from character import CharacterSchema
from combat import (
    Character,
    ClassFeature,
    CombatEnvironment,
    GridMap,
    Position,
    Stats,
    Team,
    WeaponAttack,
    apply_level_up,
    can_level_up,
    get_level_for_xp,
    get_proficiency_bonus,
)
from rules.progression import spell_slots_for_level


def make_character(
    class_name: str = "Fighter",
    subclass_name: str | None = None,
    level: int = 1,
    experience: int = 0,
) -> Character:
    return Character(
        name="Hero",
        hp=10,
        max_hp=10,
        ac=12,
        position=Position(0, 0),
        speed=3,
        stats=Stats(),
        team=Team.PLAYERS,
        class_name=class_name,
        subclass_name=subclass_name,
        level=level,
        experience=experience,
        weapons=[WeaponAttack(name="Sword", range=1, damage=2)],
    )


def test_xp_thresholds_map_to_supported_levels() -> None:
    assert get_level_for_xp(-1) == 1
    assert get_level_for_xp(0) == 1
    assert get_level_for_xp(299) == 1
    assert get_level_for_xp(300) == 2
    assert get_level_for_xp(899) == 2
    assert get_level_for_xp(900) == 3
    assert get_level_for_xp(2699) == 3
    assert get_level_for_xp(2700) == 4
    assert get_level_for_xp(6499) == 4
    assert get_level_for_xp(6500) == 5
    assert get_level_for_xp(999999) == 5


def test_proficiency_bonus_changes_at_level_five() -> None:
    assert [get_proficiency_bonus(level) for level in range(1, 5)] == [2, 2, 2, 2]
    assert get_proficiency_bonus(5) == 3


def test_apply_level_up_never_exceeds_level_five() -> None:
    character = make_character(level=4, experience=999999)

    apply_level_up(character)

    assert character.level == 5
    assert character.proficiency_bonus == 3
    assert can_level_up(character) is False


def test_level_up_updates_fighter_features_and_resources() -> None:
    character = make_character(
        class_name="Fighter",
        subclass_name="Champion",
        level=1,
        experience=900,
    )

    assert can_level_up(character)
    apply_level_up(character)

    feature_names = {feature.name for feature in character.class_features}
    assert character.level == 3
    assert character.proficiency_bonus == 2
    assert {"Second Wind", "Action Surge", "Improved Critical"}.issubset(feature_names)
    assert all(isinstance(feature, ClassFeature) for feature in character.class_features)
    assert set(character.resources) == {"action_surge", "second_wind"}


def test_spellcaster_level_up_updates_spell_slots() -> None:
    character = make_character(
        class_name="Wizard",
        subclass_name="School of Evocation",
        level=1,
        experience=6500,
    )

    apply_level_up(character)

    feature_names = {feature.name for feature in character.class_features}
    assert character.level == 5
    assert character.proficiency_bonus == 3
    assert "Sculpt Spells" in feature_names
    assert character.spell_slots == spell_slots_for_level(5)
    assert character.spell_slots_remaining == spell_slots_for_level(5)
    assert "cast_spell" in character.common_actions


def test_character_schema_reflects_progression_fields() -> None:
    character = make_character(
        class_name="Cleric",
        subclass_name="Life Domain",
        level=2,
        experience=300,
    )

    schema = CharacterSchema.from_character(character)

    assert schema.name == "Hero"
    assert schema.progression.level == 2
    assert schema.progression.experience == 300
    assert schema.progression.class_name == "Cleric"
    assert schema.progression.subclass_name == "Life Domain"


def test_combat_environment_does_not_auto_level_up_mid_encounter() -> None:
    character = make_character(level=1, experience=6500)
    enemy = Character(
        name="Enemy",
        hp=10,
        max_hp=10,
        ac=12,
        position=Position(1, 0),
        speed=3,
        stats=Stats(),
        team=Team.ENEMIES,
    )

    environment = CombatEnvironment(
        characters=[character, enemy],
        grid_map=GridMap(width=3, height=3),
        use_initiative=False,
        log_to_console=False,
    )

    assert environment.combat_state.characters[0].level == 1
    assert can_level_up(environment.combat_state.characters[0])
