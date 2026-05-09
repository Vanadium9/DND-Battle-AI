from combat import (
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
    weapon = fighter.abilities[0]

    assert fighter.team is Team.PLAYERS
    assert fighter.stats.str > fighter.stats.dex
    assert isinstance(weapon, WeaponAttack)
    assert weapon.range == 1
    assert weapon.damage == "2d6+4"


def test_fighter_archer_preset() -> None:
    archer = FighterArcher()
    weapon = archer.abilities[0]

    assert archer.team is Team.PLAYERS
    assert archer.stats.dex > archer.stats.str
    assert isinstance(weapon, WeaponAttack)
    assert weapon.range == 6
    assert weapon.damage == "1d8+4"


def test_enemy_presets() -> None:
    goblin = Goblin()
    orc = Orc()

    assert goblin.team is Team.ENEMIES
    assert orc.team is Team.ENEMIES
    assert len(goblin.abilities) == 2
    assert any(ability.range == 1 for ability in goblin.abilities)
    assert any(ability.range > 1 for ability in goblin.abilities)
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
