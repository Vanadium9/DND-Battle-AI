import pytest

from agents import MainActionType, build_action_masks
from combat import (
    Bomb,
    Character,
    CharacterBuildRequest,
    CombatState,
    GridMap,
    HealerKit,
    ItemDefinition,
    ItemEffect,
    PotionOfHealing,
    Position,
    Stats,
    Team,
    UseObjectAction,
    WeaponAttack,
    build_character,
)


def character(
    name: str,
    position: Position,
    team: Team,
    hp: int = 10,
    max_hp: int = 10,
) -> Character:
    return Character(
        name=name,
        hp=hp,
        max_hp=max_hp,
        ac=12,
        position=position,
        speed=3,
        stats=Stats(dex=14),
        team=team,
        weapons=[WeaponAttack(name="Dagger", range=1, damage=1)],
    )


def test_potion_heals_and_spends_action(monkeypatch) -> None:
    hero = character("Hero", Position(0, 0), Team.PLAYERS, hp=4, max_hp=12)
    potion = PotionOfHealing()
    hero.inventory = [potion]
    state = CombatState(characters=[hero], grid_map=GridMap(width=3, height=3))
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: 2)

    result = UseObjectAction(actor_id=0, item=potion, target_id=0).execute(state)

    assert result.success
    assert hero.hp == 10
    assert hero.action_economy.action_available is False
    assert hero.action_economy.bonus_action_available is True
    assert hero.action_economy.reaction_available is True


def test_potion_quantity_is_spent() -> None:
    hero = character("Hero", Position(0, 0), Team.PLAYERS, hp=4, max_hp=12)
    potion = PotionOfHealing()
    hero.inventory = [potion]
    state = CombatState(characters=[hero], grid_map=GridMap(width=3, height=3))

    result = UseObjectAction(actor_id=0, item=potion, target_id=0).execute(state)

    assert result.success
    assert potion.quantity == 0


def test_bomb_deals_fire_damage_and_is_consumed(monkeypatch) -> None:
    hero = character("Alchemist", Position(0, 0), Team.PLAYERS)
    bomb = Bomb()
    hero.inventory = [bomb]
    enemy = character("Enemy", Position(3, 0), Team.ENEMIES, hp=20, max_hp=20)
    state = CombatState(
        characters=[hero, enemy],
        grid_map=GridMap(width=5, height=3),
    )
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: 1)

    result = UseObjectAction(
        actor_id=0,
        item=bomb,
        target_cell=enemy.position,
    ).execute(state)

    assert result.success
    assert enemy.hp < enemy.max_hp
    assert bomb.quantity == 0


def test_item_without_quantity_is_masked_and_invalid() -> None:
    hero = character("Hero", Position(0, 0), Team.PLAYERS, hp=4, max_hp=12)
    potion = PotionOfHealing(quantity=0)
    hero.inventory = [potion]
    state = CombatState(characters=[hero], grid_map=GridMap(width=3, height=3))

    masks = build_action_masks(state, actor_id=0)

    assert not masks["main_action_type"][MainActionType.USE_OBJECT]
    assert not UseObjectAction(actor_id=0, item=potion, target_id=0).is_valid(state)


def test_healer_kit_stabilizes_target() -> None:
    hero = character("Hero", Position(0, 0), Team.PLAYERS)
    kit = HealerKit(quantity=1)
    hero.inventory = [kit]
    ally = character("Ally", Position(1, 0), Team.PLAYERS, hp=0, max_hp=10)
    state = CombatState(
        characters=[hero, ally],
        grid_map=GridMap(width=3, height=3),
    )

    result = UseObjectAction(actor_id=0, item=kit, target_id=1).execute(state)

    assert result.success
    assert ally.stable is True
    assert kit.quantity == 0


def test_character_builder_accepts_only_implemented_items() -> None:
    character_with_item = build_character(
        CharacterBuildRequest(
            name="Builder Hero",
            class_name="Fighter",
            inventory=("Potion of Healing",),
        )
    )

    assert character_with_item.inventory[0].name == "Potion of Healing"

    with pytest.raises(ValueError, match="not implemented"):
        build_character(
            CharacterBuildRequest(
                name="Builder Hero",
                class_name="Fighter",
                inventory=(
                    ItemDefinition(
                        name="Unsupported Wand",
                        effect=ItemEffect(damage=1),
                        implemented=False,
                    ),
                ),
            )
        )
