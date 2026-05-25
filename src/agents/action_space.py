"""Hierarchical action space helpers for PPO agents."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum

import torch

from combat.aoe import (
    AOE_DIRECTIONS,
    AoEDirection,
    AoEShape,
    AoETargeting,
    affected_creatures,
    coerce_aoe_direction,
    direction_from_positions,
    positions_for_aoe,
)
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
    ActionSurgeAction,
    AttackAction,
    CastSpellAction,
    ChannelDivinityPreserveLifeAction,
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
    SecondWindAction,
    ShoveAction,
    StabilizeAction,
    UseObjectAction,
)
from combat.class_features import (
    available_implemented_class_features,
    implemented_feature_active_actions,
)
from combat.cover import CoverType
from combat.items import CombatItem, resolve_item, supported_item_aoe_shape
from combat.models import Character, CombatState, Position, SpellAbility, WeaponAttack
from combat.spellcasting import (
    available_castable_spells,
    can_target_spell as can_target_spell_with_rules,
    spell_has_aoe,
    spell_aoe_shape,
    spell_requires_direction,
    spell_requires_target_cell,
    spell_system_available,
)


class ActionCategory(IntEnum):
    """Top-level action economy bucket."""

    MAIN_ACTION = 0
    BONUS_ACTION = 1
    MOVEMENT = 2
    REACTION = 3
    END_TURN = 4
    CLASS_FEATURE = 5


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


@dataclass(frozen=True)
class SpellOption:
    """Action-space spell option with optional AoE direction."""

    spell: SpellAbility
    direction: AoEDirection | None = None


def build_action_masks(state: CombatState, actor_id: int) -> dict[str, torch.Tensor]:
    """Build boolean masks for the D&D-like hierarchical action components."""

    actor = state.character_at(actor_id)
    masks = _empty_masks(state, actor)
    if actor is None or not actor.can_take_turn or not _is_active_actor(state, actor_id):
        return masks

    main_action_type_mask = _build_main_action_type_mask(state, actor_id, actor)
    move_mask = _build_move_mask(state, actor)
    target_mask = _build_target_mask(state, actor_id, actor, main_action_type_mask)
    target_cell_mask = _build_target_cell_mask(state, actor, main_action_type_mask)
    direction_mask = _build_direction_mask(state, actor, main_action_type_mask)
    option_mask = _build_option_mask(state, actor_id, actor, main_action_type_mask)

    action_category_mask = torch.zeros(ACTION_CATEGORY_COUNT, dtype=torch.bool)
    action_category_mask[int(ActionCategory.MAIN_ACTION)] = bool(main_action_type_mask.any())
    action_category_mask[int(ActionCategory.BONUS_ACTION)] = _has_bonus_action(state, actor)
    action_category_mask[int(ActionCategory.MOVEMENT)] = bool(move_mask.any())
    action_category_mask[int(ActionCategory.REACTION)] = _has_reaction(state, actor)
    action_category_mask[int(ActionCategory.END_TURN)] = (
        actor.is_alive and COMMON_ACTION_END_TURN in actor.common_actions
    )
    action_category_mask[int(ActionCategory.CLASS_FEATURE)] = _has_class_feature_action(
        state,
        actor_id,
        actor,
    )

    return {
        "action_category": action_category_mask,
        "main_action_type": main_action_type_mask,
        "target_index": target_mask,
        "move_index": move_mask,
        "target_cell_index": target_cell_mask,
        "direction_index": direction_mask,
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
    target_cell_index: int | None = None,
    direction_index: int | None = None,
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

    if selected_category is ActionCategory.BONUS_ACTION:
        actor = state.character_at(actor_id)
        bonus_spell = _spell_at_option(actor, option_index, "bonus_action")
        if actor is not None and bonus_spell is not None:
            target_id = None
            if bonus_spell.damage is not None or bonus_spell.healing is not None:
                target = state.character_at(target_index)
                if not _can_target_spell(state, actor, target, bonus_spell):
                    raise ValueError(
                        f"target_index {target_index} is not valid for {bonus_spell.name}"
                    )
                target_id = target_index
            return CastSpellAction(actor_id=actor_id, spell=bonus_spell, target_id=target_id)
        if actor is not None and "second_wind" in implemented_feature_active_actions(
            actor,
            "bonus_action",
        ):
            return SecondWindAction(actor_id=actor_id)
        raise ValueError(f"Action category {selected_category.name} is masked for actor {actor_id}.")

    if selected_category is ActionCategory.CLASS_FEATURE:
        actor = state.character_at(actor_id)
        if actor is not None and "preserve_life" in _class_feature_actions(
            state,
            actor_id,
            actor,
        ):
            return ChannelDivinityPreserveLifeAction(
                actor_id=actor_id,
                target_id=_target_or_first_valid(
                    target_index,
                    state,
                    actor_id,
                    lambda _target_id, target: _can_preserve_life_target(state, actor, target),
                ),
            )
        if actor is not None and "action_surge" in _class_feature_actions(
            state,
            actor_id,
            actor,
        ):
            return ActionSurgeAction(actor_id=actor_id)
        raise ValueError(f"Action category {selected_category.name} is masked for actor {actor_id}.")

    if selected_category is ActionCategory.REACTION:
        actor = state.character_at(actor_id)
        reaction_spell = _spell_at_option(actor, option_index, "reaction")
        if actor is not None and reaction_spell is not None:
            target_id = actor_id if reaction_spell.target_type == "self" else None
            if reaction_spell.damage is not None or reaction_spell.healing is not None:
                target = state.character_at(target_index)
                if not _can_target_spell(state, actor, target, reaction_spell):
                    raise ValueError(
                        f"target_index {target_index} is not valid for {reaction_spell.name}"
                    )
                target_id = target_index
            return CastSpellAction(actor_id=actor_id, spell=reaction_spell, target_id=target_id)
        raise ValueError(f"Action category {selected_category.name} is masked for actor {actor_id}.")

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
        return _decode_cast_spell(
            state,
            actor_id,
            target_index,
            move_index if target_cell_index is None else target_cell_index,
            option_index,
            direction_index,
        )

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
        actor = state.character_at(actor_id)
        item = _item_for_option(actor, option_index)
        if actor is not None and item is not None:
            shape = supported_item_aoe_shape(item)
            if shape is AoEShape.RADIUS:
                selected_cell_index = move_index if target_cell_index is None else target_cell_index
                target_cell = _position_from_move_index(state, selected_cell_index)
                if not _can_target_cell_with_item(state, actor, target_cell, item):
                    raise ValueError(f"target_cell {target_cell} is not valid for {item.name}")
                return UseObjectAction(
                    actor_id=actor_id,
                    object_name=item.name,
                    item=item,
                    target_cell=target_cell,
                )
            if shape in {AoEShape.CONE, AoEShape.LINE}:
                selected_direction_index = target_index if direction_index is None else direction_index
                direction = coerce_aoe_direction(selected_direction_index)
                if direction is None or not _can_direction_with_item(state, actor, direction, item):
                    raise ValueError(f"direction is not valid for {item.name}")
                return UseObjectAction(
                    actor_id=actor_id,
                    object_name=item.name,
                    item=item,
                    direction=direction,
                )
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
        "target_cell_index": torch.zeros(_move_space_size(state), dtype=torch.bool),
        "direction_index": torch.zeros(len(AOE_DIRECTIONS), dtype=torch.bool),
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
    main_action_type_mask[int(MainActionType.HIDE)] = _can_hide(state, actor)
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
                for spell in _available_spells(actor, "action")
            ):
                target_mask[target_id] = True

    if _has_bonus_spell_target(state, actor):
        for target_id, target in enumerate(state.characters):
            if any(
                _can_target_spell(state, actor, target, spell)
                for spell in _available_spells(actor, "bonus_action")
            ):
                target_mask[target_id] = True

    if _has_reaction_spell_target(state, actor):
        for target_id, target in enumerate(state.characters):
            if any(
                _can_target_spell(state, actor, target, spell)
                for spell in _available_spells(actor, "reaction")
            ):
                target_mask[target_id] = True

    if "preserve_life" in _class_feature_actions(state, actor_id, actor):
        for target_id, target in enumerate(state.characters):
            if _can_preserve_life_target(state, actor, target):
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


def _build_target_cell_mask(
    state: CombatState,
    actor: Character,
    main_action_type_mask: torch.Tensor,
) -> torch.Tensor:
    target_cell_mask = torch.zeros(_move_space_size(state), dtype=torch.bool)
    if state.grid_map is None:
        return target_cell_mask

    if main_action_type_mask[int(MainActionType.CAST_SPELL)]:
        for spell in _available_spells(actor, "action"):
            if not spell_requires_target_cell(spell):
                continue
            for position in _grid_positions(state):
                if _can_target_cell_with_spell(state, actor, position, spell):
                    target_cell_mask[_move_index_from_position(state, position)] = True

    if main_action_type_mask[int(MainActionType.USE_OBJECT)]:
        for item in _available_items(actor):
            if supported_item_aoe_shape(item) is not AoEShape.RADIUS:
                continue
            for position in _grid_positions(state):
                if _can_target_cell_with_item(state, actor, position, item):
                    target_cell_mask[_move_index_from_position(state, position)] = True

    return target_cell_mask


def _build_direction_mask(
    state: CombatState,
    actor: Character,
    main_action_type_mask: torch.Tensor,
) -> torch.Tensor:
    direction_mask = torch.zeros(len(AOE_DIRECTIONS), dtype=torch.bool)

    if main_action_type_mask[int(MainActionType.CAST_SPELL)]:
        for spell in _available_spells(actor, "action"):
            if not spell_requires_direction(spell):
                continue
            for direction_index, direction in enumerate(AOE_DIRECTIONS):
                if _can_direction_with_spell(state, actor, direction, spell):
                    direction_mask[direction_index] = True

    if main_action_type_mask[int(MainActionType.USE_OBJECT)]:
        for item in _available_items(actor):
            if supported_item_aoe_shape(item) not in {AoEShape.CONE, AoEShape.LINE}:
                continue
            for direction_index, direction in enumerate(AOE_DIRECTIONS):
                if _can_direction_with_item(state, actor, direction, item):
                    direction_mask[direction_index] = True

    return direction_mask


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
        for spell_index, spell_option in enumerate(_spell_options(actor, "action")):
            if _spell_option_is_valid(state, actor, spell_option):
                option_mask[spell_index] = True

    if _has_bonus_spell_target(state, actor):
        for spell_index, spell in enumerate(_available_spells(actor, "bonus_action")):
            if _spell_has_valid_target_or_no_target(state, actor, spell):
                option_mask[spell_index] = True

    if _has_reaction_spell_target(state, actor):
        for spell_index, spell in enumerate(_available_spells(actor, "reaction")):
            if _spell_has_valid_target_or_no_target(state, actor, spell):
                option_mask[spell_index] = True

    if main_action_type_mask[int(MainActionType.SHOVE)]:
        option_mask[SHOVE_PRONE_OPTION] = True
        option_mask[SHOVE_PUSH_OPTION] = True

    if main_action_type_mask[int(MainActionType.SEARCH)]:
        option_mask[SEARCH_PERCEPTION_OPTION] = True
        option_mask[SEARCH_INVESTIGATION_OPTION] = True

    if main_action_type_mask[int(MainActionType.USE_OBJECT)]:
        for object_index, item in enumerate(_available_items(actor)):
            if _item_option_is_valid(state, actor, item):
                option_mask[object_index] = True
        if not _available_items(actor):
            option_mask[0] = True

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
    target_cell_index: int,
    option_index: int,
    direction_index: int | None = None,
) -> CastSpellAction:
    actor = state.character_at(actor_id)
    spell_option = _spell_option_at_index(actor, option_index, "action")
    if actor is None or spell_option is None:
        raise ValueError(f"option_index {option_index} has no valid spell for actor {actor_id}")
    spell = spell_option.spell
    if spell_requires_target_cell(spell):
        target_cell = _position_from_move_index(state, target_cell_index)
        if not _can_target_cell_with_spell(state, actor, target_cell, spell):
            raise ValueError(f"target_cell {target_cell} is not valid for {spell.name}")
        return CastSpellAction(actor_id=actor_id, spell=spell, target_cell=target_cell)
    if spell_requires_direction(spell):
        direction = (
            coerce_aoe_direction(direction_index)
            if direction_index is not None
            else spell_option.direction
        )
        if direction is None or not _can_direction_with_spell(state, actor, direction, spell):
            raise ValueError(f"direction is not valid for {spell.name}")
        return CastSpellAction(actor_id=actor_id, spell=spell, direction=direction)
    target_id = None
    if spell.damage is not None or spell.healing is not None:
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
        not actor.can_take_turn
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
        and _cover_between(state, actor.position, target.position) is not CoverType.FULL_COVER
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
        len(_spell_options(actor)),
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


def _grid_positions(state: CombatState) -> tuple[Position, ...]:
    if state.grid_map is None:
        return ()
    return tuple(
        Position(x, y)
        for y in range(state.grid_map.height)
        for x in range(state.grid_map.width)
    )


def _distance(first: Position, second: Position, state: CombatState) -> int:
    if state.grid_map is not None:
        return state.grid_map.manhattan_distance(first, second)
    return abs(first.x - second.x) + abs(first.y - second.y)


def _is_active_actor(state: CombatState, actor_id: int) -> bool:
    return actor_id == state.active_actor_id


def _can_spend_action(actor: Character, action_name: str) -> bool:
    return (
        actor.can_take_turn
        and actor.action_economy.action_available
        and action_name in actor.common_actions
    )


def _can_cast_spell(state: CombatState, actor: Character) -> bool:
    if not _can_spend_action(actor, COMMON_ACTION_CAST_SPELL):
        return False
    if not _spell_system_available(actor):
        return False
    return any(
        _spell_has_valid_target_or_no_target(state, actor, spell)
        for spell in _available_spells(actor, "action")
    )


def _available_spells(
    actor: Character,
    action_cost: str | None = None,
) -> list[SpellAbility]:
    spells = available_castable_spells(actor)
    if action_cost is None:
        return spells
    return [spell for spell in spells if spell.action_cost == action_cost]


def _spell_options(
    actor: Character,
    action_cost: str | None = None,
) -> list[SpellOption]:
    options: list[SpellOption] = []
    for spell in _available_spells(actor, action_cost):
        if spell_requires_direction(spell):
            options.extend(SpellOption(spell=spell, direction=direction) for direction in AOE_DIRECTIONS)
        else:
            options.append(SpellOption(spell=spell))
    return options


def _spell_option_at_index(
    actor: Character | None,
    option_index: int,
    action_cost: str | None = None,
) -> SpellOption | None:
    if actor is None:
        return None
    options = _spell_options(actor, action_cost)
    if option_index < 0 or option_index >= len(options):
        return None
    return options[option_index]


def _spell_at_option(
    actor: Character | None,
    option_index: int,
    action_cost: str | None = None,
) -> SpellAbility | None:
    if actor is None:
        return None
    spells = _available_spells(actor, action_cost)
    if option_index < 0 or option_index >= len(spells):
        return None
    return spells[option_index]


def _spell_option_is_valid(
    state: CombatState,
    actor: Character,
    spell_option: SpellOption,
) -> bool:
    spell = spell_option.spell
    if spell_requires_target_cell(spell):
        return any(
            _can_target_cell_with_spell(state, actor, position, spell)
            for position in _grid_positions(state)
        )
    if spell_requires_direction(spell):
        return spell_option.direction is not None and _can_direction_with_spell(
            state,
            actor,
            spell_option.direction,
            spell,
        )
    return _spell_has_valid_target_or_no_target(state, actor, spell)


def _spell_has_valid_target_or_no_target(
    state: CombatState,
    actor: Character,
    spell: SpellAbility,
) -> bool:
    if spell.damage is None and spell.healing is None:
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
    single_target_valid = can_target_spell_with_rules(
        actor,
        target,
        spell,
        distance=_distance(actor.position, target.position, state),
        has_line_of_sight=_has_line_of_sight(state, actor.position, target.position),
        has_full_cover=_cover_between(state, actor.position, target.position)
        is CoverType.FULL_COVER,
    )
    if not single_target_valid:
        return False
    if spell_has_aoe(spell):
        return _can_area_spell_target(state, actor, target, spell)
    return True


def _can_area_spell_target(
    state: CombatState,
    actor: Character,
    target: Character,
    spell: SpellAbility,
) -> bool:
    shape = spell_aoe_shape(spell)
    if shape is None:
        return False
    if shape is AoEShape.RADIUS:
        return _can_target_cell_with_spell(state, actor, target.position, spell)
    direction = direction_from_positions(actor.position, target.position)
    return _can_direction_with_spell(state, actor, direction, spell)


def _can_target_cell_with_spell(
    state: CombatState,
    actor: Character,
    target_cell: Position,
    spell: SpellAbility,
) -> bool:
    if state.grid_map is not None and not state.grid_map.in_bounds(target_cell):
        return False
    if _distance(actor.position, target_cell, state) > spell.range:
        return False
    if not _has_line_of_sight(state, actor.position, target_cell):
        return False
    if _cover_between(state, actor.position, target_cell) is CoverType.FULL_COVER:
        return False
    targeting = AoETargeting(
        shape=AoEShape.RADIUS,
        origin=actor.position,
        size=spell.area_size,
        target_cell=target_cell,
    )
    return _aoe_has_affected_creature(state, targeting)


def _can_direction_with_spell(
    state: CombatState,
    actor: Character,
    direction: AoEDirection,
    spell: SpellAbility,
) -> bool:
    shape = spell_aoe_shape(spell)
    if shape not in {AoEShape.CONE, AoEShape.LINE}:
        return False
    targeting = AoETargeting(
        shape=shape,
        origin=actor.position,
        size=spell.area_size,
        direction=direction,
    )
    return _aoe_has_affected_creature(state, targeting)


def _can_target_cell_with_item(
    state: CombatState,
    actor: Character,
    target_cell: Position,
    item: CombatItem,
) -> bool:
    if state.grid_map is not None and not state.grid_map.in_bounds(target_cell):
        return False
    if _distance(actor.position, target_cell, state) > item.range:
        return False
    if not _has_line_of_sight(state, actor.position, target_cell):
        return False
    if _cover_between(state, actor.position, target_cell) is CoverType.FULL_COVER:
        return False
    targeting = AoETargeting(
        shape=AoEShape.RADIUS,
        origin=actor.position,
        size=item.area_size,
        target_cell=target_cell,
    )
    return _aoe_has_affected_creature(state, targeting)


def _can_direction_with_item(
    state: CombatState,
    actor: Character,
    direction: AoEDirection,
    item: CombatItem,
) -> bool:
    shape = supported_item_aoe_shape(item)
    if shape not in {AoEShape.CONE, AoEShape.LINE}:
        return False
    targeting = AoETargeting(
        shape=shape,
        origin=actor.position,
        size=item.area_size,
        direction=direction,
    )
    return _aoe_has_affected_creature(state, targeting)


def _aoe_has_affected_creature(
    state: CombatState,
    targeting: AoETargeting,
) -> bool:
    positions = positions_for_aoe(targeting)
    if state.grid_map is not None:
        positions = {
            position for position in positions if state.grid_map.in_bounds(position)
        }
    return bool(affected_creatures(state.characters, positions))


def _spell_system_available(actor: Character) -> bool:
    return spell_system_available(actor)


def _has_spell_slot(actor: Character, spell: SpellAbility) -> bool:
    return spell in _available_spells(actor)


def _can_help(state: CombatState, actor_id: int, actor: Character) -> bool:
    if not _can_spend_action(actor, COMMON_ACTION_HELP):
        return False
    return any(
        target_id != actor_id and not target.is_dead
        for target_id, target in enumerate(state.characters)
    )


def _can_hide(state: CombatState, actor: Character) -> bool:
    if not _can_spend_action(actor, COMMON_ACTION_HIDE) or actor.hidden:
        return False
    enemies = [
        character
        for character in state.characters
        if character is not actor and character.team != actor.team and character.is_alive
    ]
    if not enemies:
        return True
    return all(
        not _has_line_of_sight(state, enemy.position, actor.position)
        or _cover_between(state, enemy.position, actor.position) is not CoverType.NO_COVER
        for enemy in enemies
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


def _has_bonus_action(state: CombatState, actor: Character) -> bool:
    return actor.action_economy.bonus_action_available and (
        bool(implemented_feature_active_actions(actor, "bonus_action"))
        or _has_bonus_spell_target(state, actor)
    )


def _has_reaction(state: CombatState, actor: Character) -> bool:
    return actor.action_economy.reaction_available and (
        bool(implemented_feature_active_actions(actor, "reaction"))
        or _has_reaction_spell_target(state, actor)
    )


def _has_class_feature_action(
    state: CombatState,
    actor_id: int,
    actor: Character,
) -> bool:
    return actor.can_take_turn and bool(_class_feature_actions(state, actor_id, actor))


def _class_feature_actions(
    state: CombatState,
    actor_id: int,
    actor: Character,
) -> tuple[str, ...]:
    actions: list[str] = []
    for feature in available_implemented_class_features(actor):
        if feature.active_action is None:
            continue
        if feature.active_action == "preserve_life":
            if (
                feature.action_cost == "action"
                and actor.action_economy.action_available
                and any(
                    _can_preserve_life_target(state, actor, target)
                    for target in state.characters
                )
            ):
                actions.append(feature.active_action)
            continue
        if feature.action_cost is not None:
            continue
        actions.append(feature.active_action)
    return tuple(actions)


def _has_bonus_spell_target(state: CombatState, actor: Character) -> bool:
    if not actor.action_economy.bonus_action_available:
        return False
    return any(
        _spell_has_valid_target_or_no_target(state, actor, spell)
        for spell in _available_spells(actor, "bonus_action")
    )


def _has_reaction_spell_target(state: CombatState, actor: Character) -> bool:
    if not actor.action_economy.reaction_available:
        return False
    return any(
        _spell_has_valid_target_or_no_target(state, actor, spell)
        for spell in _available_spells(actor, "reaction")
    )


def _can_preserve_life_target(
    state: CombatState,
    actor: Character,
    target: Character,
) -> bool:
    return (
        target.team == actor.team
        and target.is_alive
        and target.hp < max(1, target.max_hp // 2)
        and _distance(actor.position, target.position, state) <= 6
        and _has_line_of_sight(state, actor.position, target.position)
    )


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
    return max(1, len(_available_items(actor)))


def _object_name_for_option(actor: Character, option_index: int) -> str:
    items = _available_items(actor)
    if 0 <= option_index < len(items):
        return items[option_index].name
    return "object"


def _item_for_option(actor: Character | None, option_index: int) -> CombatItem | None:
    if actor is None:
        return None
    items = _available_items(actor)
    if 0 <= option_index < len(items):
        return items[option_index]
    return None


def _available_items(actor: Character) -> list[CombatItem]:
    items = getattr(actor, "items", None)
    if not isinstance(items, (list, tuple)):
        return []
    return [
        item
        for item in (resolve_item(actor, candidate) for candidate in items)
        if item is not None and item.implemented
    ]


def _item_option_is_valid(
    state: CombatState,
    actor: Character,
    item: CombatItem,
) -> bool:
    shape = supported_item_aoe_shape(item)
    if shape is AoEShape.RADIUS:
        return any(
            _can_target_cell_with_item(state, actor, position, item)
            for position in _grid_positions(state)
        )
    if shape in {AoEShape.CONE, AoEShape.LINE}:
        return any(
            _can_direction_with_item(state, actor, direction, item)
            for direction in AOE_DIRECTIONS
        )
    return True


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


def _cover_between(
    state: CombatState,
    origin: Position,
    target: Position,
) -> CoverType:
    grid_map = state.grid_map
    if grid_map is None:
        return CoverType.NO_COVER
    return grid_map.get_cover_between(origin, target)


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
