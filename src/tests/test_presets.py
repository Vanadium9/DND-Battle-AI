from combat import (
    ClassFeature,
    FighterArcher,
    FighterChampionArcher,
    FighterChampionGreatsword,
    FighterLevel1Basic,
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
    assert fighter.level == 5
    assert fighter.proficiency_bonus == 3
    assert fighter.stats.str > fighter.stats.dex
    assert fighter.subclass_name == "Champion"
    assert fighter.fighting_style == "Great Weapon Fighting"
    assert fighter.wearing_armor is True
    assert isinstance(weapon, WeaponAttack)
    assert weapon.range == 1
    assert weapon.damage == "2d6"
    assert weapon.ability_score == "str"
    assert weapon.damage_ability_score == "str"
    assert weapon.two_handed is True
    assert weapon.heavy is True
    assert all(isinstance(feature, ClassFeature) for feature in fighter.class_features)
    assert {feature.name for feature in fighter.class_features} == {
        "Action Surge",
        "Ability Score Improvement",
        "Extra Attack",
        "Fighting Style",
        "Improved Critical",
        "Martial Archetype: Champion",
        "Second Wind",
    }
    assert set(fighter.resources) == {"action_surge", "second_wind"}


def test_fighter_archer_preset() -> None:
    archer = FighterArcher()
    weapon = archer.weapons[0]

    assert archer.team is Team.PLAYERS
    assert archer.class_name == "Fighter"
    assert archer.level == 5
    assert archer.subclass_name == "Champion"
    assert archer.fighting_style == "Archery"
    assert archer.stats.dex > archer.stats.str
    assert isinstance(weapon, WeaponAttack)
    assert weapon.range == 6
    assert weapon.damage == "1d8"
    assert weapon.ability_score == "dex"
    assert weapon.damage_ability_score == "dex"


def test_new_fighter_presets() -> None:
    archer = FighterChampionArcher()
    basic = FighterLevel1Basic()

    assert archer.level == 5
    assert archer.fighting_style == "Archery"
    assert basic.level == 1
    assert basic.subclass_name is None
    assert basic.fighting_style == "Defense"
    assert basic.ac == 17
    assert {feature.name for feature in basic.class_features} == {
        "Fighting Style",
        "Second Wind",
    }


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
