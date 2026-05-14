from combat import (
    AttackAction,
    Character,
    ClassFeature,
    CombatState,
    Goblin,
    GridMap,
    Position,
    Resource,
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


def test_character_accepts_legacy_weapon_abilities_but_attacks_from_weapons() -> None:
    weapon = WeaponAttack(name="Legacy Sword", range=1, damage=3, attack_bonus=20)
    hero = Character(
        name="Hero",
        hp=10,
        max_hp=10,
        ac=12,
        position=Position(0, 0),
        speed=2,
        stats=Stats(),
        team=Team.PLAYERS,
        abilities=[weapon],
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

    assert hero.weapons == [weapon]
    assert AttackAction(actor_id=0, target_id=1).is_valid(state)


def test_attack_uses_stats_proficiency_and_weapon_bonus_for_damage() -> None:
    weapon = WeaponAttack(
        name="Axe",
        range=1,
        damage=1,
        attack_bonus=20,
        ability_score="str",
        damage_ability_score="str",
    )
    hero = Character(
        name="Hero",
        hp=10,
        max_hp=10,
        ac=12,
        position=Position(0, 0),
        speed=2,
        stats=Stats(str=18),
        team=Team.PLAYERS,
        proficiency_bonus=3,
        weapons=[weapon],
    )
    enemy = Character(
        name="Enemy",
        hp=20,
        max_hp=20,
        ac=12,
        position=Position(1, 0),
        speed=2,
        stats=Stats(),
        team=Team.ENEMIES,
    )
    state = CombatState(characters=[hero, enemy], grid_map=GridMap(width=3, height=3))

    result = AttackAction(actor_id=0, target_id=1, weapon=weapon).execute(state)

    assert result.success
    assert enemy.hp == 15


def test_class_resources_reset_without_owning_common_attack_logic() -> None:
    feature = ClassFeature(
        name="Action Surge",
        resource_name="action_surge",
        required_level=2,
    )
    resource = Resource(name="action_surge", max_uses=1)
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
        class_name="Fighter",
        class_features=[feature],
        resources={"action_surge": resource},
        weapons=[weapon],
    )

    resource.spend()
    hero.reset_combat_resources()

    assert resource.uses_remaining == 1
    assert AttackAction(actor_id=0, target_id=0, weapon=weapon) is not None


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
