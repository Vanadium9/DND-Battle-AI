from combat import (
    ClassFeature,
    FighterArcher,
    FighterChampionGreatsword,
    Goblin,
    GridMap,
    Orc,
    Position,
    Team,
    WeaponAttack,
    create_test_encounter,
)


def test_fighter_champion_greatsword_preset() -> None:
    fighter = FighterChampionGreatsword()
    weapon = fighter.weapons[0]

    assert fighter.team is Team.PLAYERS
    assert fighter.class_name == "Fighter"
    assert fighter.level == 3
    assert fighter.proficiency_bonus == 2
    assert fighter.stats.str > fighter.stats.dex
    assert isinstance(weapon, WeaponAttack)
    assert weapon.range == 1
    assert weapon.damage == "2d6"
    assert weapon.ability_score == "str"
    assert weapon.damage_ability_score == "str"
    assert all(isinstance(feature, ClassFeature) for feature in fighter.class_features)
    assert {feature.name for feature in fighter.class_features} == {
        "Action Surge",
        "Second Wind",
    }
    assert set(fighter.resources) == {"action_surge", "second_wind"}


def test_fighter_archer_preset() -> None:
    archer = FighterArcher()
    weapon = archer.weapons[0]

    assert archer.team is Team.PLAYERS
    assert archer.class_name == "Fighter"
    assert archer.stats.dex > archer.stats.str
    assert isinstance(weapon, WeaponAttack)
    assert weapon.range == 6
    assert weapon.damage == "1d8"
    assert weapon.ability_score == "dex"
    assert weapon.damage_ability_score == "dex"


def test_enemy_presets() -> None:
    goblin = Goblin()
    orc = Orc()

    assert goblin.team is Team.ENEMIES
    assert orc.team is Team.ENEMIES
    assert goblin.class_name is None
    assert orc.class_name is None
    assert len(goblin.weapons) == 2
    assert any(weapon.range == 1 for weapon in goblin.weapons)
    assert any(weapon.range > 1 for weapon in goblin.weapons)
    assert not goblin.class_features
    assert not orc.class_features
    assert orc.max_hp > goblin.max_hp
    assert orc.stats.str > goblin.stats.str


def test_create_test_encounter() -> None:
    encounter = create_test_encounter()

    assert isinstance(encounter.grid_map, GridMap)
    assert len(encounter.characters) == 4
    assert len(encounter.characters_for_team(Team.PLAYERS)) == 2
    assert len(encounter.characters_for_team(Team.ENEMIES)) == 2
    assert all(
        encounter.grid_map.in_bounds(character.position)
        for character in encounter.characters
    )
    assert {character.position for character in encounter.characters} == {
        Position(0, 1),
        Position(0, 3),
        Position(5, 1),
        Position(5, 3),
    }
