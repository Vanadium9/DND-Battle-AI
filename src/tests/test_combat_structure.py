from combat import (
    Character,
    CombatState,
    Goblin,
    GridMap,
    Position,
    Stats,
    Team,
    WeaponAttack,
)
from agents import ActionCategory, MainActionType, build_action_masks


def test_weapon_attacks_are_separate_from_class_features() -> None:
    goblin = Goblin()

    assert goblin.weapons
    assert all(isinstance(weapon, WeaponAttack) for weapon in goblin.weapons)
    assert not goblin.class_features


def test_common_action_masks_respect_character_common_actions() -> None:
    weapon = WeaponAttack(name="Sword", range=1, damage=2, attack_bonus=20)
    hero = Character(
        name="Hero",
        hp=10,
        max_hp=10,
        ac=12,
        position=Position(0, 0),
        speed=2,
        stats=Stats(),
        team=Team.PLAYERS,
        weapons=[weapon],
        common_actions=["end_turn"],
    )
    enemy = Character(
        name="Enemy",
        hp=10,
        max_hp=10,
        ac=12,
        position=Position(1, 0),
        speed=2,
        stats=Stats(),
        team=Team.ENEMIES,
    )
    state = CombatState(characters=[hero, enemy], grid_map=GridMap(width=3, height=3))

    masks = build_action_masks(state, actor_id=0)

    assert not masks["action_category"][ActionCategory.MOVEMENT]
    assert not masks["main_action_type"][MainActionType.ATTACK]
    assert masks["action_category"][ActionCategory.END_TURN]
