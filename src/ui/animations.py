"""Presentation-only battle animation helpers for the desktop UI."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from combat import (
    AttackAction,
    CastSpellAction,
    CombatAction,
    CombatState,
    MoveAction,
    OpportunityAttackAction,
    Position,
    UseObjectAction,
)
from combat.aoe import (
    AoEShape,
    AoETargeting,
    coerce_aoe_direction,
    coerce_aoe_shape,
    positions_for_aoe,
)
from combat.common_actions import ActionResult
from combat.replay import StateSnapshot


MIN_ANIMATION_SPEED_MS = 300
MAX_ANIMATION_SPEED_MS = 1500
DEFAULT_ANIMATION_SPEED_MS = 600


class BattleAnimationKind(str, Enum):
    """Supported visual-only animation types."""

    MOVEMENT = "movement"
    MELEE_ATTACK = "melee_attack"
    RANGED_ATTACK = "ranged_attack"
    SPELL = "spell"
    DAMAGE = "damage"
    HEALING = "healing"
    DEATH = "death"


@dataclass(frozen=True)
class BattleAnimation:
    """One visual effect derived from a completed combat step."""

    kind: BattleAnimationKind
    actor_id: int | None = None
    target_ids: tuple[int, ...] = ()
    start: Position | None = None
    end: Position | None = None
    cells: tuple[Position, ...] = ()
    color: str = "#ffffff"


@dataclass(frozen=True)
class BattleAnimationFrame:
    """Current animation frame consumed by BattleMapWidget."""

    animations: tuple[BattleAnimation, ...] = ()
    progress: float = 0.0


def normalize_animation_speed(value: object) -> int:
    """Clamp animation/autobattle speed to a readable GUI range."""

    try:
        speed = int(value)
    except (TypeError, ValueError):
        speed = DEFAULT_ANIMATION_SPEED_MS
    return max(MIN_ANIMATION_SPEED_MS, min(MAX_ANIMATION_SPEED_MS, speed))


def animation_duration_ms(speed_ms: int) -> int:
    """Return a short visual duration that leaves breathing room before next auto-step."""

    return max(160, int(normalize_animation_speed(speed_ms) * 0.72))


def build_battle_animations(
    before: StateSnapshot | None,
    after_state: CombatState,
    action: CombatAction,
    result: ActionResult,
) -> tuple[BattleAnimation, ...]:
    """Build presentation-only animations from the state delta around one action."""

    if before is None or not result.success:
        return ()

    animations: list[BattleAnimation] = []
    actor_before = _character_snapshot(before, action.actor_id)
    actor_start = _snapshot_position(actor_before)
    actor_end = _state_position(after_state, action.actor_id)

    if isinstance(action, MoveAction) and actor_start is not None and actor_end is not None:
        if actor_start != actor_end:
            animations.append(
                BattleAnimation(
                    BattleAnimationKind.MOVEMENT,
                    actor_id=action.actor_id,
                    start=actor_start,
                    end=actor_end,
                    color="#2f80ed",
                )
            )
    elif isinstance(action, (AttackAction, OpportunityAttackAction)):
        target_start = _snapshot_target_position(before, getattr(action, "target_id", None))
        if actor_start is not None and target_start is not None:
            weapon = getattr(action, "weapon", None)
            weapon_range = int(getattr(weapon, "range", 0) or 0)
            if weapon_range <= 1 and _grid_distance(actor_start, target_start) <= 1:
                animations.append(
                    BattleAnimation(
                        BattleAnimationKind.MELEE_ATTACK,
                        actor_id=action.actor_id,
                        target_ids=(int(action.target_id),),
                        start=actor_start,
                        end=target_start,
                        color="#f2c94c",
                    )
                )
            else:
                animations.append(
                    BattleAnimation(
                        BattleAnimationKind.RANGED_ATTACK,
                        actor_id=action.actor_id,
                        target_ids=(int(action.target_id),),
                        start=actor_start,
                        end=target_start,
                        color="#f2994a",
                    )
                )
    elif isinstance(action, CastSpellAction):
        animations.extend(_spell_animations(before, action, actor_start))
    elif isinstance(action, UseObjectAction):
        animations.extend(_item_animations(before, action, actor_start))

    animations.extend(_delta_animations(before, after_state))
    return tuple(animations)


def _spell_animations(
    before: StateSnapshot,
    action: CastSpellAction,
    actor_start: Position | None,
) -> list[BattleAnimation]:
    if actor_start is None:
        return []
    spell = action.spell
    target_id = getattr(action, "target_id", None)
    target_position = _snapshot_target_position(before, target_id)
    cells = _aoe_cells(
        actor_start,
        getattr(spell, "area_shape", None),
        getattr(spell, "area_size", 0),
        action.target_cell,
        action.direction,
    )
    if cells:
        return [
            BattleAnimation(
                BattleAnimationKind.SPELL,
                actor_id=action.actor_id,
                target_ids=_ids_in_cells(before, cells),
                start=actor_start,
                end=action.target_cell or target_position,
                cells=tuple(sorted(cells, key=lambda position: (position.y, position.x))),
                color="#9b51e0",
            )
        ]
    if target_position is None:
        return []
    return [
        BattleAnimation(
            BattleAnimationKind.SPELL,
            actor_id=action.actor_id,
            target_ids=(int(target_id),) if target_id is not None else (),
            start=actor_start,
            end=target_position,
            cells=(target_position,),
            color="#9b51e0",
        )
    ]


def _item_animations(
    before: StateSnapshot,
    action: UseObjectAction,
    actor_start: Position | None,
) -> list[BattleAnimation]:
    if actor_start is None:
        return []
    item = action.item
    target_id = getattr(action, "target_id", None)
    target_position = _snapshot_target_position(before, target_id)
    cells = _aoe_cells(
        actor_start,
        getattr(item, "area_shape", None),
        getattr(item, "area_size", 0),
        action.target_cell,
        action.direction,
    )
    if cells:
        return [
            BattleAnimation(
                BattleAnimationKind.SPELL,
                actor_id=action.actor_id,
                target_ids=_ids_in_cells(before, cells),
                start=actor_start,
                end=action.target_cell or target_position,
                cells=tuple(sorted(cells, key=lambda position: (position.y, position.x))),
                color="#eb5757",
            )
        ]
    if target_position is None:
        target_position = action.target_cell
    if target_position is None:
        return []
    kind = BattleAnimationKind.RANGED_ATTACK if bool(getattr(item, "thrown", False)) else BattleAnimationKind.SPELL
    return [
        BattleAnimation(
            kind,
            actor_id=action.actor_id,
            target_ids=(int(target_id),) if target_id is not None else (),
            start=actor_start,
            end=target_position,
            cells=(target_position,),
            color="#eb5757" if kind is BattleAnimationKind.RANGED_ATTACK else "#27ae60",
        )
    ]


def _delta_animations(
    before: StateSnapshot,
    after_state: CombatState,
) -> list[BattleAnimation]:
    animations: list[BattleAnimation] = []
    for character_id, after_character in enumerate(after_state.characters):
        before_character = _character_snapshot(before, character_id)
        if before_character is None:
            continue
        before_hp = int(before_character.get("hp", 0))
        after_hp = int(after_character.hp)
        before_alive = bool(before_character.get("alive", False))
        if before_hp > after_hp:
            animations.append(
                BattleAnimation(
                    BattleAnimationKind.DAMAGE,
                    target_ids=(character_id,),
                    end=after_character.position,
                    cells=(after_character.position,),
                    color="#eb5757",
                )
            )
        elif after_hp > before_hp:
            animations.append(
                BattleAnimation(
                    BattleAnimationKind.HEALING,
                    target_ids=(character_id,),
                    end=after_character.position,
                    cells=(after_character.position,),
                    color="#27ae60",
                )
            )
        if before_alive and not after_character.is_alive:
            animations.append(
                BattleAnimation(
                    BattleAnimationKind.DEATH,
                    target_ids=(character_id,),
                    end=after_character.position,
                    cells=(after_character.position,),
                    color="#111827",
                )
            )
    return animations


def _aoe_cells(
    origin: Position,
    shape_value: object,
    size_value: object,
    target_cell: Position | None,
    direction_value: object,
) -> set[Position]:
    shape = coerce_aoe_shape(shape_value)
    try:
        size = int(size_value)
    except (TypeError, ValueError):
        size = 0
    if shape is None or size <= 0:
        return set()
    if shape is AoEShape.RADIUS:
        if target_cell is None:
            return set()
        return positions_for_aoe(
            AoETargeting(
                shape=shape,
                origin=origin,
                size=size,
                target_cell=target_cell,
            )
        )
    direction = coerce_aoe_direction(direction_value)
    if direction is None:
        return set()
    return positions_for_aoe(
        AoETargeting(
            shape=shape,
            origin=origin,
            size=size,
            direction=direction,
        )
    )


def _ids_in_cells(snapshot: StateSnapshot, cells: set[Position]) -> tuple[int, ...]:
    ids: list[int] = []
    for character in snapshot.get("characters", []):
        position = _snapshot_position(character)
        if position in cells:
            ids.append(int(character["id"]))
    return tuple(ids)


def _snapshot_target_position(
    snapshot: StateSnapshot,
    target_id: int | None,
) -> Position | None:
    if target_id is None:
        return None
    return _snapshot_position(_character_snapshot(snapshot, int(target_id)))


def _snapshot_position(character: dict[str, Any] | None) -> Position | None:
    if character is None:
        return None
    position = character.get("position")
    if not isinstance(position, dict):
        return None
    return Position(int(position.get("x", 0)), int(position.get("y", 0)))


def _state_position(state: CombatState, character_id: int) -> Position | None:
    character = state.character_at(character_id)
    return character.position if character is not None else None


def _character_snapshot(
    snapshot: StateSnapshot,
    character_id: int,
) -> dict[str, Any] | None:
    for character in snapshot.get("characters", []):
        if int(character.get("id", -1)) == int(character_id):
            return character
    return None


def _grid_distance(start: Position, end: Position) -> int:
    return max(abs(start.x - end.x), abs(start.y - end.y))
