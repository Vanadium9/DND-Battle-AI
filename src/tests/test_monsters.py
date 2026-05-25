import pytest

from combat import (
    BASIC_MONSTER_PRESETS,
    COMMON_ACTION_ATTACK,
    COMMON_ACTION_END_TURN,
    AttackAction,
    Bandit,
    Character,
    CombatState,
    DamageType,
    FireElementalSimple,
    GoblinArcher,
    GoblinMelee,
    GridMap,
    OrcWarrior,
    Position,
    SkeletonArcher,
    Stats,
    Team,
    Wolf,
)


@pytest.mark.parametrize("monster_factory", BASIC_MONSTER_PRESETS)
def test_basic_monster_preset_creates_valid_enemy(monster_factory) -> None:
    monster = monster_factory()

    assert monster.hp > 0
    assert monster.max_hp >= monster.hp
    assert monster.ac > 0
    assert monster.speed > 0
    assert isinstance(monster.stats, Stats)
    assert monster.weapons
    assert monster.challenge_rating is not None
    assert monster.xp_value > 0
    assert monster.role
    assert COMMON_ACTION_ATTACK in monster.common_actions
    assert COMMON_ACTION_END_TURN in monster.common_actions


@pytest.mark.parametrize(
    "monster_factory",
    [
        GoblinMelee,
        GoblinArcher,
        OrcWarrior,
        SkeletonArcher,
        Bandit,
        Wolf,
        FireElementalSimple,
    ],
)
def test_basic_monster_has_valid_attack(monster_factory) -> None:
    monster = monster_factory(Position(1, 1))
    target = Character(
        name="Target",
        hp=20,
        max_hp=20,
        ac=12,
        position=Position(2, 1),
        speed=3,
        stats=Stats(),
        team=Team.PLAYERS,
    )
    state = CombatState(
        characters=[monster, target],
        grid_map=GridMap(width=5, height=5),
    )

    assert any(
        AttackAction(actor_id=0, target_id=1, weapon=weapon).is_valid(state)
        for weapon in monster.weapons
    )


def test_fire_elemental_has_fire_immunity() -> None:
    elemental = FireElementalSimple()

    assert DamageType.FIRE in elemental.immunities


def test_skeleton_archer_has_poison_defense() -> None:
    skeleton = SkeletonArcher()

    assert (
        DamageType.POISON in skeleton.immunities
        or DamageType.POISON in skeleton.resistances
    )


def test_wolf_has_melee_attack_and_high_speed() -> None:
    wolf = Wolf()

    assert wolf.speed >= 4
    assert any(weapon.range <= 1 for weapon in wolf.weapons)


def test_goblin_archer_has_ranged_weapon() -> None:
    archer = GoblinArcher()

    assert any(weapon.range > 1 for weapon in archer.weapons)
