from types import SimpleNamespace

from combat import (
    AttackAction,
    Character,
    CombatEnvironment,
    GoblinMelee,
    GridMap,
    OrcWarrior,
    Position,
    Stats,
    Team,
    WeaponAttack,
    can_level_up,
)
from rules.xp import award_party_xp, calculate_encounter_xp, get_xp_for_cr


def player(
    name: str,
    position: Position,
    experience: int = 0,
    weapon_damage: int = 50,
) -> Character:
    return Character(
        name=name,
        hp=20,
        max_hp=20,
        ac=12,
        position=position,
        speed=3,
        stats=Stats(str=16),
        team=Team.PLAYERS,
        class_name="Fighter",
        level=1,
        experience=experience,
        weapons=[
            WeaponAttack(
                name="Sword",
                range=1,
                damage=weapon_damage,
                attack_bonus=20,
                ability_score="str",
                damage_ability_score=None,
            )
        ],
    )


def test_xp_for_cr_table() -> None:
    assert get_xp_for_cr(0) == 10
    assert get_xp_for_cr("1/8") == 25
    assert get_xp_for_cr(0.25) == 50
    assert get_xp_for_cr("1/2") == 100
    assert get_xp_for_cr(5) == 1800


def test_calculate_encounter_xp_uses_monster_xp_or_cr_table() -> None:
    monsters = [
        GoblinMelee(),
        OrcWarrior(),
        SimpleNamespace(challenge_rating="1/8", xp_value=0),
    ]

    assert calculate_encounter_xp(monsters) == 175


def test_award_party_xp_splits_between_party_members() -> None:
    first = player("First", Position(0, 0))
    second = player("Second", Position(0, 1))

    total_xp = award_party_xp([first, second], [GoblinMelee()])

    assert total_xp == 50
    assert first.experience == 25
    assert second.experience == 25


def test_level_up_after_combat_is_possible_without_auto_level_up() -> None:
    hero = player("Hero", Position(0, 0), experience=250)
    monster = OrcWarrior(Position(1, 0))
    environment = CombatEnvironment(
        characters=[hero, monster],
        grid_map=GridMap(width=3, height=3),
        use_initiative=False,
        log_to_console=False,
    )

    result = environment.step(AttackAction(actor_id=0, target_id=1))
    hero_after = environment.combat_state.characters[0]

    assert result.success
    assert environment.is_done()
    assert environment.last_awarded_xp == 100
    assert hero_after.experience == 350
    assert hero_after.level == 1
    assert can_level_up(hero_after)
    assert result.reward < environment.last_awarded_xp


def test_xp_is_not_awarded_to_players_on_defeat() -> None:
    monster = OrcWarrior(Position(1, 0))
    hero = player("Hero", Position(0, 0), experience=250)
    hero.hp = 1
    monster.weapons[0].damage = 10
    monster.weapons[0].attack_bonus = 20
    environment = CombatEnvironment(
        characters=[monster, hero],
        grid_map=GridMap(width=3, height=3),
        use_initiative=False,
        log_to_console=False,
    )

    result = environment.step(AttackAction(actor_id=0, target_id=1))
    hero_after = environment.combat_state.characters[1]

    assert result.success
    assert environment.is_done()
    assert environment.get_winner() is Team.ENEMIES
    assert environment.last_awarded_xp == 0
    assert hero_after.experience == 250
