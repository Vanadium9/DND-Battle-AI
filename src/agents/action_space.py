"""Hierarchical action space helpers for PPO agents."""

from __future__ import annotations

from enum import IntEnum

import torch

from combat.actions import AttackAction, CombatAction, EndTurnAction, MoveAction
from combat.models import Character, CombatState, Position, WeaponAttack


class ActionType(IntEnum):
    """Top-level action choice."""

    MOVE = 0
    MAIN_ACTION_ATTACK = 1
    END_TURN = 2
    BONUS_ACTION = 3
    REACTION = 4


ACTION_TYPE_COUNT = len(ActionType)


def build_action_masks(state: CombatState, actor_id: int) -> dict[str, torch.Tensor]:
    """Build boolean masks for the hierarchical action components."""

    action_type_mask = torch.zeros(ACTION_TYPE_COUNT, dtype=torch.bool)
    target_mask = torch.zeros(len(state.characters), dtype=torch.bool)
    move_mask = torch.zeros(_move_space_size(state), dtype=torch.bool)

    actor = state.character_at(actor_id)
    if actor is None or actor.is_dead or not _is_active_actor(state, actor_id):
        return {
            "action_type": action_type_mask,
            "target_index": target_mask,
            "move_index": move_mask,
        }

    move_mask = _build_move_mask(state, actor)
    target_mask = _build_target_mask(state, actor_id, actor)

    action_type_mask[int(ActionType.MOVE)] = bool(move_mask.any())
    action_type_mask[int(ActionType.MAIN_ACTION_ATTACK)] = (
        actor.action_economy.action_available and bool(target_mask.any())
    )
    action_type_mask[int(ActionType.END_TURN)] = True

    # Reserved for future concrete actions. They remain masked until implemented.
    action_type_mask[int(ActionType.BONUS_ACTION)] = False
    action_type_mask[int(ActionType.REACTION)] = False

    return {
        "action_type": action_type_mask,
        "target_index": target_mask,
        "move_index": move_mask,
    }


def decode_action(
    action_type: int | ActionType,
    target_index: int,
    move_index: int,
    state: CombatState,
    actor_id: int,
) -> CombatAction:
    """Decode hierarchical PPO outputs into a combat action."""

    selected_type = _coerce_action_type(action_type)
    masks = build_action_masks(state, actor_id)

    if not masks["action_type"][int(selected_type)]:
        raise ValueError(f"Action type {selected_type.name} is masked for actor {actor_id}")

    if selected_type is ActionType.MOVE:
        if move_index < 0 or move_index >= len(masks["move_index"]):
            raise ValueError(f"move_index {move_index} is out of range")
        if not masks["move_index"][move_index]:
            raise ValueError(f"move_index {move_index} is masked for actor {actor_id}")
        return MoveAction(
            actor_id=actor_id,
            destination=_position_from_move_index(state, move_index),
        )

    if selected_type is ActionType.MAIN_ACTION_ATTACK:
        if target_index < 0 or target_index >= len(masks["target_index"]):
            raise ValueError(f"target_index {target_index} is out of range")
        if not masks["target_index"][target_index]:
            raise ValueError(f"target_index {target_index} is masked for actor {actor_id}")

        actor = state.character_at(actor_id)
        target = state.character_at(target_index)
        weapon = _first_valid_weapon_for_target(state, actor, target)
        if weapon is None:
            raise ValueError(f"target_index {target_index} has no valid weapon attack")
        return AttackAction(actor_id=actor_id, target_id=target_index, weapon=weapon)

    if selected_type is ActionType.END_TURN:
        return EndTurnAction(actor_id=actor_id)

    raise ValueError(f"Action type {selected_type.name} is reserved and not implemented")


def _build_move_mask(state: CombatState, actor: Character) -> torch.Tensor:
    move_mask = torch.zeros(_move_space_size(state), dtype=torch.bool)
    if state.grid_map is None or actor.action_economy.movement_remaining <= 0:
        return move_mask

    movement_cells = state.grid_map.movement_cells(
        actor.position,
        actor.action_economy.movement_remaining,
        state.characters,
    )
    movement_cells.discard(actor.position)

    for position in movement_cells:
        move_mask[_move_index_from_position(state, position)] = True
    return move_mask


def _build_target_mask(
    state: CombatState,
    actor_id: int,
    actor: Character,
) -> torch.Tensor:
    target_mask = torch.zeros(len(state.characters), dtype=torch.bool)
    if not actor.action_economy.action_available:
        return target_mask

    for target_id, target in enumerate(state.characters):
        if target_id == actor_id or target.team == actor.team or target.is_dead:
            continue
        if _first_valid_weapon_for_target(state, actor, target) is not None:
            target_mask[target_id] = True
    return target_mask


def _first_valid_weapon_for_target(
    state: CombatState,
    actor: Character | None,
    target: Character | None,
) -> WeaponAttack | None:
    if actor is None or target is None or actor.is_dead or target.is_dead:
        return None

    distance = _distance(actor.position, target.position, state)
    for ability in actor.available_abilities:
        if isinstance(ability, WeaponAttack) and distance <= ability.range:
            return ability
    return None


def _move_space_size(state: CombatState) -> int:
    if state.grid_map is None:
        return 0
    return state.grid_map.width * state.grid_map.height


def _move_index_from_position(state: CombatState, position: Position) -> int:
    if state.grid_map is None:
        raise ValueError("Cannot index movement without a grid map")
    return position.y * state.grid_map.width + position.x


def _position_from_move_index(state: CombatState, move_index: int) -> Position:
    if state.grid_map is None:
        raise ValueError("Cannot decode movement without a grid map")
    x = move_index % state.grid_map.width
    y = move_index // state.grid_map.width
    return Position(x, y)


def _distance(first: Position, second: Position, state: CombatState) -> int:
    if state.grid_map is not None:
        return state.grid_map.manhattan_distance(first, second)
    return abs(first.x - second.x) + abs(first.y - second.y)


def _is_active_actor(state: CombatState, actor_id: int) -> bool:
    return bool(state.characters) and actor_id == state.turn_index % len(state.characters)


def _coerce_action_type(action_type: int | ActionType) -> ActionType:
    try:
        return ActionType(action_type)
    except ValueError as error:
        raise ValueError(f"Unknown action_type: {action_type}") from error
