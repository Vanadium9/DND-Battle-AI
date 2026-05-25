"""Combat item definitions and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from combat.aoe import AoEShape, coerce_aoe_shape
from combat.damage import DamageType


@dataclass
class ItemEffect:
    """Data-only item effect payload."""

    damage: int | str | None = None
    damage_type: DamageType | str | None = None
    save_ability: str | None = None
    save_half_damage: bool = False


@dataclass
class CombatItem:
    """A simple combat-usable item definition."""

    name: str
    range: int = 1
    action_cost: str = "action"
    damage: int | str | None = None
    damage_type: DamageType | str | None = None
    effect: ItemEffect | None = None
    save_ability: str | None = None
    save_half_damage: bool = False
    area_shape: AoEShape | str | None = None
    area_size: int = 0
    implemented: bool = True
    consumed_on_use: bool = False

    @property
    def has_aoe(self) -> bool:
        return coerce_aoe_shape(self.area_shape) is not None and self.area_size > 0


def resolve_item(character: Any, item: CombatItem | str | None) -> CombatItem | None:
    """Resolve an item object or name against a character's item list."""

    items = getattr(character, "items", ())
    if isinstance(item, CombatItem):
        if item in items or not items:
            return item
        for candidate in items:
            if _item_name(candidate) == _item_name(item):
                return _coerce_item(candidate)
        return None
    if item is None:
        return None
    item_key = _lookup_key(item)
    for candidate in items:
        if _lookup_key(_item_name(candidate)) == item_key:
            return _coerce_item(candidate)
    return None


def supported_item_aoe_shape(item: CombatItem | None) -> AoEShape | None:
    """Return the supported AoE shape for an item."""

    if item is None:
        return None
    return coerce_aoe_shape(item.area_shape)


def _coerce_item(item: Any) -> CombatItem | None:
    if isinstance(item, CombatItem):
        return item
    return None


def _item_name(item: Any) -> str:
    return str(getattr(item, "name", item))


def _lookup_key(value: object) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())
