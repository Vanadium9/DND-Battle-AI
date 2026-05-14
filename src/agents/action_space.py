"""Hierarchical action space helpers for PPO agents."""

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum

import torch

from combat.actions import (
    COMMON_ACTION_ATTACK,
    COMMON_ACTION_CAST_SPELL,
    COMMON_ACTION_DASH,
    COMMON_ACTION_DISENGAGE,
    COMMON_ACTION_DODGE,
    COMMON_ACTION_END_TURN,
    COMMON_ACTION_GRAPPLE,
    COMMON_ACTION_HELP,
    COMMON_ACTION_HIDE,
    COMMON_ACTION_IMPROVISED,
    COMMON_ACTION_MOVE,
    COMMON_ACTION_READY,
    COMMON_ACTION_SEARCH,
    COMMON_ACTION_SHOVE,
    COMMON_ACTION_STABILIZE,
    COMMON_ACTION_USE_OBJECT,
    AttackAction,
    CastSpellAction,
    CombatAction,
    DashAction,
    DisengageAction,
    DodgeAction,
    EndTurnAction,
    GrappleAction,
    HelpAction,
    HideAction,
    ImprovisedAction,
    MoveAction,
    ReadyAction,
    SearchAction,
    ShoveAction,
    StabilizeAction,
    UseObjectAction,
)
from combat.models import Character, CombatState, Position, SpellAbility, WeaponAttack


class ActionCategory(IntEnum):
    """Top-level action economy bucket."""

    MAIN_ACTION = 0
    BONUS_ACTION = 1
    MOVEMENT = 2
    REACTION = 3
    END_TURN = 4


class MainActionType(IntEnum):
    """Concrete action that consumes the action resource."""

    ATTACK = 0
    CAST_SPELL = 1
    DASH = 2
    DISENGAGE = 3
    DODGE = 4
    HELP = 5
    HIDE = 6
    SEARCH = 7
    USE_OBJECT = 8
    READY = 9
    GRAPPLE = 10
    SHOVE = 11
    STABILIZE = 12
    IMPROVISED = 13


ACTION_CATEGORY_COUNT = len(ActionCategory)
MAIN_ACTION_TYPE_COUNT = len(MainActionType)
MIN_OPTION_COUNT = 8

SHOVE_PRONE_OPTION = 0
SHOVE_PUSH_OPTION = 1
SEARCH_PERCEPTION_OPTION = 0
SEARCH_INVESTIGATION_OPTION = 1


def build_action_masks(state: CombatState, actor_id: int) -> dict[str, torch.Tensor]:
    """Build boolean masks for the D&D-like hierarchical action components."""

    actor = state.character_at(actor_id)
    masks = _empty_masks(state, actor)
    if actor is None or actor.is_dead or not _is_active_actor(state, actor_id):
        return masks

    main_action_type_mask = _build_main_action_type_mask(state, actor_id, actor)
    move_mask = _build_move_mask(state, actor)
    target_mask = _build_target_mask(state, actor_id, actor, main_action_type_mask)
    option_mask = _build_option_mask(state, actor_id, actor, main_action_type_mask)

    action_category_mask = torch.zeros(ACTION_CATEGORY_COUNT, dtype=torch.bool)
    action_category_mask[int(ActionCategory.MAIN_ACTION)] = bool(main_action_type_mask.any())
    action_category_mask[int(ActionCategory.BONUS_ACTION)] = _has_bonus_action(actor)
    action_category_mask[int(ActionCategory.MOVEMENT)] = bool(move_mask.any())
    action_category_mask[int(ActionCategory.REACTION)] = _has_reaction(actor)
    action_category_mask[int(ActionCategory.END_TURN)] = (
        actor.is_alive and COMMON_ACTION_END_TURN in actor.common_actions
    )

    return {
        "action_category": action_category_mask,
        "main_action_type": main_action_type_mask,
        "target_index": target_mask,
        "move_index": move_mask,
        "option_index": option_mask,
    }


def decode_action(
    action_category: int | ActionCategory,
    main_action_type: int | MainActionType,
    target_index: int,
    move_index: int,
    option_index: int,
    state: CombatState,
    actor_id: int,
) -> CombatAction:
    """Decode hierarchical PPO outputs into a concrete combat action."""

    selected_category = _coerce_action_category(action_category)
    masks = build_action_masks(state, actor_id)
    _validate_masked_index(
        selected_category,
        masks["action_category"],
        "action_category",
        actor_id,
    )

    if selected_category is ActionCategory.MOVEMENT:
        _validate_masked_index(move_index, masks["move_index"], "move_index", actor_id)
        return MoveAction(
            actor_id=actor_id,
            destination=_position_from_move_index(state, move_index),
        )

    if selected_category is ActionCategory.END_TURN:
        return EndTurnAction(actor_id=actor_id)

    if selected_category in {ActionCategory.BONUS_ACTION, ActionCategory.REACTION}:
        raise ValueError(f"Action category {selected_category.name} is reserved.")

    selected_main_action = _coerce_main_action_type(main_action_type)
    _validate_masked_index(
        selected_main_action,
        masks["main_action_type"],
        "main_action_type",
        actor_id,
    )
    if _main_action_uses_target(selected_main_action):
        _validate_masked_index(target_index, masks["target_index"], "target_index", actor_id)
    if _main_action_uses_option(selected_main_action):
        _validate_masked_index(option_index, masks["option_index"], "option_index", actor_id)

    if selected_main_action is MainActionType.ATTACK:
        return _decode_attack(state, actor_id, target_index, option_index)

    if selected_main_action is MainActionType.CAST_SPELL:
        return _decode_cast_spell(state, actor_id, target_index, option_index)

    if selected_main_action is MainActionType.DASH:
        return DashAction(actor_id=actor_id)

    if selected_main_action is MainActionType.DISENGAGE:
        return DisengageAction(actor_id=actor_id)

    if selected_main_action is MainActionType.DODGE:
        return DodgeAction(actor_id=actor_id)

    if selected_main_action is MainActionType.HELP:
        return HelpAction(
            actor_id=actor_id,
            target_id=_target_or_first_valid(
                target_index,
                state,
                actor_id,
                lambda target_id, target: target_id != actor_id and not target.is_dead,
            ),
        )

    if selected_main_action is MainActionType.HIDE:
        return HideAction(actor_id=actor_id)

    if selected_main_action is MainActionType.SEARCH:
        skill = "investigation" if option_index == SEARCH_INVESTIGATION_OPTION else "perception"
        return SearchAction(actor_id=actor_id, skill=skill)

    if selected_main_action is MainActionType.USE_OBJECT:
        return UseObjectAction(
            actor_id=actor_id,
            object_name=_object_name_for_option(state.characters[actor_id], option_index),
        )

    if selected_main_action is MainActionType.READY:
        return ReadyAction(actor_id=actor_id)

    if selected_main_action is MainActionType.GRAPPLE:
        return GrappleAction(
            actor_id=actor_id,
            target_id=_target_or_first_special_melee(
                target_index,
                state,
                actor_id,
                COMMON_ACTION_GRAPPLE,
            ),
        )

    if selected_main_action is MainActionType.SHOVE:
        shove_effect = "push" if option_index == SHOVE_PUSH_OPTION else "prone"
        return ShoveAction(
            actor_id=actor_id,
            target_id=_target_or_first_special_melee(
                target_index,
                state,
                actor_id,
                COMMON_ACTION_SHOVE,
            ),
            shove_effect=shove_effect,
        )

    if selected_main_action is MainActionType.STABILIZE:
        return StabilizeAction(
            actor_id=actor_id,
            target_id=_target_or_first_valid(
                target_index,
                state,
                actor_id,
                lambda target_id, target: target_id != actor_id
                and target.hp <= 0
                and _distance(state.characters[actor_id].position, target.position, state) <= 1,
            ),
        )

    if selected_main_action is MainActionType.IMPROVISED:
        return ImprovisedAction(actor_id=actor_id)

    raise ValueError(f"Main action type {selected_main_action.name} is not implemented")


def _empty_masks(
    state: CombatState,
    actor: Character | None,
) -> dict[str, torch.Tensor]:
    return {
        "action_category": torch.zeros(ACTION_CATEGORY_COUNT, dtype=torch.bool),
        "main_action_type": torch.zeros(MAIN_ACTION_TYPE_COUNT, dtype=torch.bool),
        "target_index": torch.zeros(len(state.characters), dtype=torch.bool),
        "move_index": torch.zeros(_move_space_size(state), dtype=torch.bool),
        "option_index": torch.zeros(_option_space_size(actor), dtype=torch.bool),
    }


def _build_main_action_type_mask(
    state: CombatState,
    actor_id: int,
    actor: Character,
) -> torch.Tensor:
    main_action_type_mask = torch.zeros(MAIN_ACTION_TYPE_COUNT, dtype=torch.bool)

    main_action_type_mask[int(MainActionType.ATTACK)] = _has_weapon_attack_target(
        state,
        actor_id,
        actor,
    )
    main_action_type_mask[int(MainActionType.CAST_SPELL)] = _can_cast_spell(state, actor)
    main_action_type_mask[int(MainActionType.DASH)] = _can_spend_action(
        actor,
        COMMON_ACTION_DASH,
    )
    main_action_type_mask[int(MainActionType.DISENGAGE)] = (
        _can_spend_action(actor, COMMON_ACTION_DISENGAGE)
        and not actor.disengaged_until_end_of_turn
    )
    main_action_type_mask[int(MainActionType.DODGE)] = (
        _can_spend_action(actor, COMMON_ACTION_DODGE)
        and not actor.dodging_until_start_of_next_turn
    )
    main_action_type_mask[int(MainActionType.HELP)] = _can_help(state, actor_id, actor)
    main_action_type_mask[int(MainActionType.HIDE)] = (
        _can_spend_action(actor, COMMON_ACTION_HIDE) and not actor.hidden
    )
    main_action_type_mask[int(MainActionType.SEARCH)] = _can_spend_action(
        actor,
        COMMON_ACTION_SEARCH,
    )
    main_action_type_mask[int(MainActionType.USE_OBJECT)] = _can_spend_action(
        actor,
        COMMON_ACTION_USE_OBJECT,
    )
    main_action_type_mask[int(MainActionType.READY)] = _can_ready(actor)
    main_action_type_mask[int(MainActionType.GRAPPLE)] = _has_special_melee_target(
        state,
        actor_id,
        actor,
        COMMON_ACTION_GRAPPLE,
    )
    main_action_type_mask[int(MainActionType.SHOVE)] = _has_special_melee_target(
        state,
        actor_id,
        actor,
        COMMON_ACTION_SHOVE,
    )
    main_action_type_mask[int(MainActionType.STABILIZE)] = _has_stabilize_target(
        state,
        actor_id,
        actor,
    )
    main_action_type_mask[int(MainActionType.IMPROVISED)] = _can_spend_action(
        actor,
        COMMON_ACTION_IMPROVISED,
    )
    return main_action_type_mask


def _build_move_mask(state: CombatState, actor: Character) -> torch.Tensor:
    move_mask = torch.zeros(_move_space_size(state), dtype=torch.bool)
    if (
        state.grid_map is None
        or actor.action_economy.movement_remaining <= 0
        or actor.action_economy.grappled
        or COMMON_ACTION_MOVE not in actor.common_actions
    ):
        return move_mask

    movement_remaining = actor.action_economy.movement_remaining
    if actor.prone:
        movement_remaining -= max(1, max(0, actor.speed) // 2)
    if movement_remaining <= 0:
        return move_mask

    movement_cells = state.grid_map.movement_cells(
        actor.position,
        movement_remaining,
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
    main_action_type_mask: torch.Tensor,
) -> torch.Tensor:
    target_mask = torch.zeros(len(state.characters), dtype=torch.bool)

    if main_action_type_mask[int(MainActionType.ATTACK)]:
        for target_id, target in enumerate(state.characters):
            if _first_valid_weapon_for_target(state, actor, target) is not None:
                target_mask[target_id] = True

    if main_action_type_mask[int(MainActionType.CAST_SPELL)]:
        for target_id, target in enumerate(state.characters):
            if any(
                _can_target_spell(state, actor, target, spell)
                for spell in _available_spells(actor)
            ):
                target_mask[target_id] = True

    if main_action_type_mask[int(MainActionType.HELP)]:
        for target_id, target in enumerate(state.characters):
            if target_id != actor_id and not target.is_dead:
                target_mask[target_id] = True

    if (
        main_action_type_mask[int(MainActionType.GRAPPLE)]
        or main_action_type_mask[int(MainActionType.SHOVE)]
    ):
        for target_id, target in enumerate(state.characters):
            if _is_special_melee_target(state, actor_id, actor, target_id, target):
                target_mask[target_id] = True

    if main_action_type_mask[int(MainActionType.STABILIZE)]:
        for target_id, target in enumerate(state.characters):
            if (
                target_id != actor_id
                and target.hp <= 0
                and _distance(actor.position, target.position, state) <= 1
            ):
                target_mask[target_id] = True

    return target_mask


def _build_option_mask(
    state: CombatState,
    actor_id: int,
    actor: Character,
    main_action_type_mask: torch.Tensor,
) -> torch.Tensor:
    option_mask = torch.zeros(_option_space_size(actor), dtype=torch.bool)

    if main_action_type_mask[int(MainActionType.ATTACK)]:
        for weapon_index, weapon in enumerate(actor.weapons):
            if _weapon_has_target(state, actor_id, actor, weapon):
                option_mask[weapon_index] = True

    if main_action_type_mask[int(MainActionType.CAST_SPELL)]:
        for spell_index, spell in enumerate(_available_spells(actor)):
            if _spell_has_valid_target_or_no_target(state, actor, spell):
                option_mask[spell_index] = True

    if main_action_type_mask[int(MainActionType.SHOVE)]:
        option_mask[SHOVE_PRONE_OPTION] = True
        option_mask[SHOVE_PUSH_OPTION] = True

    if main_action_type_mask[int(MainActionType.SEARCH)]:
        option_mask[SEARCH_PERCEPTION_OPTION] = True
        option_mask[SEARCH_INVESTIGATION_OPTION] = True

    if main_action_type_mask[int(MainActionType.USE_OBJECT)]:
        for object_index in range(_object_count(actor)):
            option_mask[object_index] = True

    return option_mask


def _decode_attack(
    state: CombatState,
    actor_id: int,
    target_index: int,
    option_index: int,
) -> AttackAction:
    actor = state.character_at(actor_id)
    target = state.character_at(target_index)
    weapon = _weapon_at_option(actor, option_index)
    if weapon is None:
        raise ValueError(f"option_index {option_index} has no valid weapon for actor {actor_id}")
    if not _is_valid_weapon_target(state, actor, target, weapon):
        raise ValueError(f"target_index {target_index} is not valid for {weapon.name}")
    return AttackAction(actor_id=actor_id, target_id=target_index, weapon=weapon)


def _decode_cast_spell(
    state: CombatState,
    actor_id: int,
    target_index: int,
    option_index: int,
) -> CastSpellAction:
    actor = state.character_at(actor_id)
    spell = _spell_at_option(actor, option_index)
    if actor is None or spell is None:
        raise ValueError(f"option_index {option_index} has no valid spell for actor {actor_id}")
    target_id = None
    if spell.damage is not None:
        target = state.character_at(target_index)
        if not _can_target_spell(state, actor, target, spell):
            raise ValueError(f"target_index {target_index} is not valid for {spell.name}")
        target_id = target_index
    return CastSpellAction(actor_id=actor_id, spell=spell, target_id=target_id)


def _first_valid_weapon_for_target(
    state: CombatState,
    actor: Character | None,
    target: Character | None,
) -> WeaponAttack | None:
    if actor is None:
        return None
    for weapon in actor.weapons:
        if _is_valid_weapon_target(state, actor, target, weapon):
            return weapon
    return None


def _weapon_at_option(actor: Character | None, option_index: int) -> WeaponAttack | None:
    if actor is None or option_index < 0 or option_index >= len(actor.weapons):
        return None
    weapon = actor.weapons[option_index]
    return weapon if weapon.available else None


def _is_valid_weapon_target(
    state: CombatState,
    actor: Character | None,
    target: Character | None,
    weapon: WeaponAttack,
) -> bool:
    if actor is None or target is None:
        return False
    if (
        actor.is_dead
        or target.is_dead
        or target is actor
        or target.team == actor.team
        or not weapon.available
        or COMMON_ACTION_ATTACK not in actor.common_actions
        or not actor.action_economy.action_available
    ):
        return False
    return (
        _distance(actor.position, target.position, state) <= weapon.range
        and _has_line_of_sight(state, actor.position, target.position)
    )


def _weapon_has_target(
    state: CombatState,
    actor_id: int,
    actor: Character,
    weapon: WeaponAttack,
) -> bool:
    return any(
        target_id != actor_id and _is_valid_weapon_target(state, actor, target, weapon)
        for target_id, target in enumerate(state.characters)
    )


def _move_space_size(state: CombatState) -> int:
    if state.grid_map is None:
        return 0
    return state.grid_map.width * state.grid_map.height


def _option_space_size(actor: Character | None) -> int:
    if actor is None:
        return MIN_OPTION_COUNT
    return max(
        MIN_OPTION_COUNT,
        len(actor.weapons),
        len(_available_spells(actor)),
        _object_count(actor),
        2,
    )


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


def _can_spend_action(actor: Character, action_name: str) -> bool:
    return actor.action_economy.action_available and action_name in actor.common_actions


def _can_cast_spell(state: CombatState, actor: Character) -> bool:
    if not _can_spend_action(actor, COMMON_ACTION_CAST_SPELL):
        return False
    if not _spell_system_available(actor):
        return False
    return any(
        _spell_has_valid_target_or_no_target(state, actor, spell)
        for spell in _available_spells(actor)
    )


def _available_spells(actor: Character) -> list[SpellAbility]:
    return [
        ability
        for ability in actor.available_abilities
        if isinstance(ability, SpellAbility) and _has_spell_slot(actor, ability)
    ]


def _spell_at_option(actor: Character | None, option_index: int) -> SpellAbility | None:
    if actor is None:
        return None
    spells = _available_spells(actor)
    if option_index < 0 or option_index >= len(spells):
        return None
    return spells[option_index]


def _spell_has_valid_target_or_no_target(
    state: CombatState,
    actor: Character,
    spell: SpellAbility,
) -> bool:
    if spell.damage is None:
        return True
    return any(_can_target_spell(state, actor, target, spell) for target in state.characters)


def _can_target_spell(
    state: CombatState,
    actor: Character,
    target: Character | None,
    spell: SpellAbility,
) -> bool:
    if target is None:
        return False
    return (
        target is not actor
        and target.team != actor.team
        and target.is_alive
        and _distance(actor.position, target.position, state) <= spell.range
        and _has_line_of_sight(state, actor.position, target.position)
    )


def _spell_system_available(actor: Character) -> bool:
    return any(
        hasattr(actor, attribute_name)
        for attribute_name in (
            "spell_slots",
            "spell_slots_remaining",
            "spellcasting",
        )
    )


def _has_spell_slot(actor: Character, spell: SpellAbility) -> bool:
    if not _spell_system_available(actor):
        return False
    if spell.spell_level <= 0:
        return True

    for attribute_name in ("spell_slots_remaining", "spell_slots"):
        slots = getattr(actor, attribute_name, None)
        if isinstance(slots, dict):
            return int(slots.get(spell.spell_level, 0)) > 0
        if isinstance(slots, int):
            return slots > 0
    return True


def _can_help(state: CombatState, actor_id: int, actor: Character) -> bool:
    if not _can_spend_action(actor, COMMON_ACTION_HELP):
        return False
    return any(
        target_id != actor_id and not target.is_dead
        for target_id, target in enumerate(state.characters)
    )


def _has_weapon_attack_target(
    state: CombatState,
    actor_id: int,
    actor: Character,
) -> bool:
    if (
        not _can_spend_action(actor, COMMON_ACTION_ATTACK)
        or COMMON_ACTION_ATTACK not in actor.common_actions
    ):
        return False
    return any(
        target_id != actor_id
        and _first_valid_weapon_for_target(state, actor, target) is not None
        for target_id, target in enumerate(state.characters)
    )


def _has_special_melee_target(
    state: CombatState,
    actor_id: int,
    actor: Character,
    action_name: str,
) -> bool:
    if (
        not _can_spend_action(actor, action_name)
        or COMMON_ACTION_ATTACK not in actor.common_actions
    ):
        return False
    return any(
        _is_special_melee_target(state, actor_id, actor, target_id, target)
        for target_id, target in enumerate(state.characters)
    )


def _is_special_melee_target(
    state: CombatState,
    actor_id: int,
    actor: Character,
    target_id: int,
    target: Character,
) -> bool:
    return (
        target_id != actor_id
        and target.team != actor.team
        and target.hp > 0
        and _distance(actor.position, target.position, state) <= 1
        and _has_line_of_sight(state, actor.position, target.position)
    )


def _has_stabilize_target(
    state: CombatState,
    actor_id: int,
    actor: Character,
) -> bool:
    if not _can_spend_action(actor, COMMON_ACTION_STABILIZE):
        return False
    return any(
        target_id != actor_id
        and target.hp <= 0
        and _distance(actor.position, target.position, state) <= 1
        for target_id, target in enumerate(state.characters)
    )


def _can_ready(actor: Character) -> bool:
    return (
        _can_spend_action(actor, COMMON_ACTION_READY)
        and actor.action_economy.reaction_available
    )


def _has_bonus_action(actor: Character) -> bool:
    return actor.action_economy.bonus_action_available and False


def _has_reaction(actor: Character) -> bool:
    return actor.action_economy.reaction_available and False


def _target_or_first_valid(
    target_index: int,
    state: CombatState,
    actor_id: int,
    predicate: Callable[[int, Character], bool],
) -> int:
    if 0 <= target_index < len(state.characters):
        target = state.characters[target_index]
        if predicate(target_index, target):
            return target_index
    for candidate_id, candidate in enumerate(state.characters):
        if predicate(candidate_id, candidate):
            return candidate_id
    raise ValueError(f"No valid target for actor {actor_id}")


def _target_or_first_special_melee(
    target_index: int,
    state: CombatState,
    actor_id: int,
    action_name: str,
) -> int:
    actor = state.character_at(actor_id)
    if actor is None:
        raise ValueError(f"Actor {actor_id} not found")
    if action_name not in actor.common_actions:
        raise ValueError(f"{action_name} is masked for actor {actor_id}")
    return _target_or_first_valid(
        target_index,
        state,
        actor_id,
        lambda candidate_id, candidate: COMMON_ACTION_ATTACK in actor.common_actions
        and _is_special_melee_target(state, actor_id, actor, candidate_id, candidate),
    )


def _object_count(actor: Character) -> int:
    objects = getattr(actor, "items", None)
    if isinstance(objects, (list, tuple)):
        return max(1, len(objects))
    return 1


def _object_name_for_option(actor: Character, option_index: int) -> str:
    objects = getattr(actor, "items", None)
    if isinstance(objects, (list, tuple)) and 0 <= option_index < len(objects):
        item = objects[option_index]
        return str(getattr(item, "name", item))
    return "object"


def _has_line_of_sight(
    state: CombatState,
    origin: Position,
    target: Position,
) -> bool:
    grid_map = state.grid_map
    if grid_map is None:
        return True
    for method_name in ("has_line_of_sight", "line_of_sight"):
        method = getattr(grid_map, method_name, None)
        if callable(method):
            return bool(method(origin, target))
    return True


def _validate_masked_index(
    selected_index: int | IntEnum,
    mask: torch.Tensor,
    name: str,
    actor_id: int,
) -> None:
    index = int(selected_index)
    if index < 0 or index >= len(mask):
        raise ValueError(f"{name} {index} is out of range")
    if not mask[index]:
        raise ValueError(f"{name} {index} is masked for actor {actor_id}")


def _main_action_uses_target(main_action_type: MainActionType) -> bool:
    return main_action_type in {
        MainActionType.ATTACK,
        MainActionType.CAST_SPELL,
        MainActionType.HELP,
        MainActionType.GRAPPLE,
        MainActionType.SHOVE,
        MainActionType.STABILIZE,
    }


def _main_action_uses_option(main_action_type: MainActionType) -> bool:
    return main_action_type in {
        MainActionType.ATTACK,
        MainActionType.CAST_SPELL,
        MainActionType.SEARCH,
        MainActionType.USE_OBJECT,
        MainActionType.SHOVE,
    }


def _coerce_action_category(action_category: int | ActionCategory) -> ActionCategory:
    try:
        return ActionCategory(action_category)
    except ValueError as error:
        raise ValueError(f"Unknown action_category: {action_category}") from error


def _coerce_main_action_type(
    main_action_type: int | MainActionType,
) -> MainActionType:
    try:
        return MainActionType(main_action_type)
    except ValueError as error:
        raise ValueError(f"Unknown main_action_type: {main_action_type}") from error
