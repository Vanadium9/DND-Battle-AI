"""Combat item definitions and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from combat.aoe import AoEShape, coerce_aoe_shape
from combat.damage import DamageType


class ItemActionCost(str, Enum):
    """Action economy resource spent by an item."""

    ACTION = "action"
    BONUS_ACTION = "bonus_action"
    REACTION = "reaction"
    FREE_INTERACTION = "free_interaction"


class ItemTargetType(str, Enum):
    """Supported item target category."""

    SELF = "self"
    ALLY = "ally"
    ENEMY = "enemy"
    POINT = "point"


@dataclass
class ItemEffect:
    """Data-only item effect payload."""

    healing: int | str | None = None
    damage: int | str | None = None
    damage_type: DamageType | str | None = None
    save_ability: str | None = None
    save_half_damage: bool = False
    stabilize: bool = False


@dataclass
class ItemDefinition:
    """A combat-usable item definition stored in a character inventory."""

    name: str
    item_type: str = "consumable"
    quantity: int = 1
    action_cost: ItemActionCost | str = ItemActionCost.ACTION
    target_type: ItemTargetType | str = ItemTargetType.SELF
    range: int = 1
    effect: ItemEffect | None = None
    consumable: bool = False
    implemented: bool = True
    damage: int | str | None = None
    damage_type: DamageType | str | None = None
    save_ability: str | None = None
    save_half_damage: bool = False
    area_shape: AoEShape | str | None = None
    area_size: int = 0
    thrown: bool = False

    @property
    def has_aoe(self) -> bool:
        return coerce_aoe_shape(self.area_shape) is not None and self.area_size > 0

    @property
    def consumed_on_use(self) -> bool:
        return self.consumable

    @consumed_on_use.setter
    def consumed_on_use(self, value: bool) -> None:
        self.consumable = bool(value)


CombatItem = ItemDefinition


def normalize_action_cost(value: ItemActionCost | str | None) -> ItemActionCost:
    if value is None:
        return ItemActionCost.ACTION
    if isinstance(value, ItemActionCost):
        return value
    normalized = _lookup_key(value)
    aliases = {
        "action": ItemActionCost.ACTION,
        "bonusaction": ItemActionCost.BONUS_ACTION,
        "bonus": ItemActionCost.BONUS_ACTION,
        "reaction": ItemActionCost.REACTION,
        "freeinteraction": ItemActionCost.FREE_INTERACTION,
        "freeobjectinteraction": ItemActionCost.FREE_INTERACTION,
        "free": ItemActionCost.FREE_INTERACTION,
    }
    return aliases.get(normalized, ItemActionCost.ACTION)


def normalize_target_type(value: ItemTargetType | str | None) -> ItemTargetType:
    if value is None:
        return ItemTargetType.SELF
    if isinstance(value, ItemTargetType):
        return value
    normalized = _lookup_key(value)
    aliases = {
        "self": ItemTargetType.SELF,
        "ally": ItemTargetType.ALLY,
        "friendly": ItemTargetType.ALLY,
        "enemy": ItemTargetType.ENEMY,
        "hostile": ItemTargetType.ENEMY,
        "point": ItemTargetType.POINT,
        "cell": ItemTargetType.POINT,
    }
    return aliases.get(normalized, ItemTargetType.SELF)


def resolve_item(character: Any, item: CombatItem | str | None) -> CombatItem | None:
    """Resolve an item object or name against a character inventory."""

    inventory = _inventory(character)
    if isinstance(item, ItemDefinition):
        if item in inventory or not inventory:
            return item
        for candidate in inventory:
            if _item_name(candidate) == _item_name(item):
                return _coerce_item(candidate)
        return None
    if item is None:
        return None
    item_key = _lookup_key(item)
    for candidate in inventory:
        if _lookup_key(_item_name(candidate)) == item_key:
            return _coerce_item(candidate)
    return None


def supported_item_aoe_shape(item: CombatItem | None) -> AoEShape | None:
    """Return the supported AoE shape for an item."""

    if item is None:
        return None
    return coerce_aoe_shape(item.area_shape)


def item_has_quantity(item: CombatItem | None) -> bool:
    return item is not None and int(getattr(item, "quantity", 0)) > 0


def item_healing(item: CombatItem) -> int | str | None:
    if item.effect is not None and item.effect.healing is not None:
        return item.effect.healing
    return None


def item_damage(item: CombatItem) -> int | str | None:
    if item.effect is not None and item.effect.damage is not None:
        return item.effect.damage
    return item.damage


def item_damage_type(item: CombatItem) -> object:
    if item.effect is not None and item.effect.damage_type is not None:
        return item.effect.damage_type
    return item.damage_type


def item_save_ability(item: CombatItem) -> str | None:
    if item.effect is not None and item.effect.save_ability is not None:
        return item.effect.save_ability
    return item.save_ability


def item_save_half_damage(item: CombatItem) -> bool:
    if item.effect is not None:
        return item.effect.save_half_damage
    return item.save_half_damage


def item_stabilizes(item: CombatItem) -> bool:
    return bool(item.effect is not None and item.effect.stabilize)


def consume_item(item: CombatItem) -> None:
    """Spend one quantity from a consumable item."""

    if not item.consumable:
        return
    item.quantity = max(0, int(item.quantity) - 1)


def _inventory(character: Any) -> list[Any]:
    inventory = getattr(character, "inventory", None)
    if isinstance(inventory, list):
        return inventory
    items = getattr(character, "items", None)
    if isinstance(items, list):
        return items
    return []


def _coerce_item(item: Any) -> CombatItem | None:
    if isinstance(item, ItemDefinition):
        return item
    return None


def _item_name(item: Any) -> str:
    return str(getattr(item, "name", item))


def _lookup_key(value: object) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())
