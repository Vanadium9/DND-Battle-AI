"""Inventory helpers and supported MVP combat items."""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable

from combat.aoe import AoEShape
from combat.damage import DamageType
from combat.items import (
    CombatItem,
    ItemActionCost,
    ItemDefinition,
    ItemEffect,
    ItemTargetType,
    item_has_quantity,
)


def PotionOfHealing(quantity: int = 1) -> ItemDefinition:
    return ItemDefinition(
        name="Potion of Healing",
        item_type="potion",
        quantity=quantity,
        action_cost=ItemActionCost.ACTION,
        target_type=ItemTargetType.SELF,
        range=0,
        effect=ItemEffect(healing="2d4+2"),
        consumable=True,
        implemented=True,
    )


def Bomb(quantity: int = 1) -> ItemDefinition:
    return ItemDefinition(
        name="Bomb",
        item_type="thrown",
        quantity=quantity,
        action_cost=ItemActionCost.ACTION,
        target_type=ItemTargetType.POINT,
        range=4,
        effect=ItemEffect(
            damage="2d6",
            damage_type=DamageType.FIRE,
            save_ability="dex",
            save_half_damage=True,
        ),
        consumable=True,
        implemented=True,
        area_shape=AoEShape.RADIUS,
        area_size=1,
        thrown=True,
    )


def AlchemistFire(quantity: int = 1) -> ItemDefinition:
    return ItemDefinition(
        name="Alchemist Fire",
        item_type="thrown",
        quantity=quantity,
        action_cost=ItemActionCost.ACTION,
        target_type=ItemTargetType.ENEMY,
        range=4,
        effect=ItemEffect(damage="1d6", damage_type=DamageType.FIRE),
        consumable=True,
        implemented=True,
        thrown=True,
    )


def HealerKit(quantity: int = 10) -> ItemDefinition:
    return ItemDefinition(
        name="Healer Kit",
        item_type="tool",
        quantity=quantity,
        action_cost=ItemActionCost.ACTION,
        target_type=ItemTargetType.ALLY,
        range=1,
        effect=ItemEffect(stabilize=True),
        consumable=True,
        implemented=True,
    )


def _lookup_key(value: object) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


ITEM_DEFINITIONS: dict[str, ItemDefinition] = {
    _lookup_key("Potion of Healing"): PotionOfHealing(),
    _lookup_key("Bomb"): Bomb(),
    _lookup_key("Alchemist Fire"): AlchemistFire(),
    _lookup_key("Healer Kit"): HealerKit(),
}


def get_item_definition(name: str) -> ItemDefinition | None:
    definition = ITEM_DEFINITIONS.get(_lookup_key(name))
    return deepcopy(definition) if definition is not None else None


def get_supported_item_definitions() -> tuple[ItemDefinition, ...]:
    return tuple(deepcopy(item) for item in ITEM_DEFINITIONS.values() if item.implemented)


def validate_item_selection(items: Iterable[str | ItemDefinition | CombatItem]) -> None:
    for item in items:
        definition = _coerce_definition(item)
        if definition is None or not definition.implemented:
            name = getattr(item, "name", item)
            raise ValueError(f"Item '{name}' is not implemented and cannot be selected.")


def resolve_inventory_items(
    items: Iterable[str | ItemDefinition | CombatItem],
) -> list[ItemDefinition]:
    resolved: list[ItemDefinition] = []
    for item in items:
        definition = _coerce_definition(item)
        if definition is None or not definition.implemented:
            name = getattr(item, "name", item)
            raise ValueError(f"Item '{name}' is not implemented and cannot be selected.")
        resolved.append(definition)
    return resolved


def available_inventory_items(character: object) -> list[ItemDefinition]:
    inventory = getattr(character, "inventory", ())
    if not isinstance(inventory, (list, tuple)):
        return []
    return [
        item
        for item in inventory
        if isinstance(item, ItemDefinition) and item.implemented and item_has_quantity(item)
    ]


def add_item(character: object, item: str | ItemDefinition | CombatItem) -> ItemDefinition:
    definition = _coerce_definition(item)
    if definition is None or not definition.implemented:
        name = getattr(item, "name", item)
        raise ValueError(f"Item '{name}' is not implemented and cannot be added.")
    inventory = getattr(character, "inventory", None)
    if not isinstance(inventory, list):
        setattr(character, "inventory", [])
        inventory = getattr(character, "inventory")
    inventory.append(definition)
    return definition


def _coerce_definition(item: str | ItemDefinition | CombatItem) -> ItemDefinition | None:
    if isinstance(item, ItemDefinition):
        return deepcopy(item)
    if isinstance(item, str):
        return get_item_definition(item)
    return None
