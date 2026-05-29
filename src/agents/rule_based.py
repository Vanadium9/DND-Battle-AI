"""Rule-based baseline policies for PPO comparison and self-play opponents."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any, Iterable

import torch

from agents.action_space import (
    ACTION_CATEGORY_COUNT,
    MAIN_ACTION_TYPE_COUNT,
    MIN_OPTION_COUNT,
    ActionCategory,
    MainActionType,
    build_action_masks,
    decode_action,
)
from combat import CombatAction, CombatState, CoverType, Position, Team, TerrainType
from combat.items import item_healing
from combat.spellcasting import available_castable_spells


@dataclass
class RuleBasedAgent:
    """Base class for mask-aware rule-based baseline policies."""

    seed: int | None = None
    target_count: int = 8
    move_count: int = 64
    option_count: int = MIN_OPTION_COUNT
    action_category_count: int = ACTION_CATEGORY_COUNT
    main_action_type_count: int = MAIN_ACTION_TYPE_COUNT
    rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def act(
        self,
        observation: Any = None,
        masks: dict[str, torch.Tensor] | None = None,
        *,
        state: CombatState,
        actor_id: int,
        actor: Any | None = None,
        deterministic: bool = True,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        masks = _masks(state, actor_id, masks)
        return _first_legal_output(state, actor_id, masks, self.default_plan(state, actor_id, masks))

    def default_plan(
        self,
        state: CombatState,
        actor_id: int,
        masks: dict[str, torch.Tensor],
    ) -> Iterable[ActionPlan]:
        return (
            ActionPlan(ActionCategory.MAIN_ACTION, MainActionType.ATTACK),
            ActionPlan(ActionCategory.MOVEMENT),
            ActionPlan(ActionCategory.MAIN_ACTION, MainActionType.DODGE),
            ActionPlan(ActionCategory.END_TURN),
        )


class AggressiveMeleeAgent(RuleBasedAgent):
    """Close distance and use melee attacks whenever possible."""

    def default_plan(
        self,
        state: CombatState,
        actor_id: int,
        masks: dict[str, torch.Tensor],
    ) -> Iterable[ActionPlan]:
        actor = state.character_at(actor_id)
        if actor is None:
            return (ActionPlan(ActionCategory.END_TURN),)
        enemies = _enemies(state, actor)
        nearest_targets = _target_ids_by_distance(state, actor, enemies)
        melee_options = _weapon_options(actor, ranged=False)
        return (
            ActionPlan(
                ActionCategory.MAIN_ACTION,
                MainActionType.ATTACK,
                target_indices=nearest_targets,
                option_indices=melee_options,
            ),
            ActionPlan(
                ActionCategory.MOVEMENT,
                move_indices=_move_indices_toward_enemy(state, actor, masks),
            ),
            ActionPlan(ActionCategory.MAIN_ACTION, MainActionType.DASH),
            ActionPlan(ActionCategory.MAIN_ACTION, MainActionType.ATTACK),
            ActionPlan(ActionCategory.END_TURN),
        )


class RangedKitingAgent(RuleBasedAgent):
    """Prefer ranged attacks, move away from melee threats, and avoid bad terrain."""

    preferred_min_distance: int = 3

    def default_plan(
        self,
        state: CombatState,
        actor_id: int,
        masks: dict[str, torch.Tensor],
    ) -> Iterable[ActionPlan]:
        actor = state.character_at(actor_id)
        if actor is None:
            return (ActionPlan(ActionCategory.END_TURN),)
        enemies = _enemies(state, actor)
        if any(_distance(state, actor.position, enemy.position) <= 1 for enemy in enemies):
            return (
                ActionPlan(
                    ActionCategory.MOVEMENT,
                    move_indices=_kite_move_indices(state, actor, masks, self.preferred_min_distance),
                ),
                *_ranged_attack_plans(state, actor, enemies),
                ActionPlan(ActionCategory.MAIN_ACTION, MainActionType.DISENGAGE),
                ActionPlan(ActionCategory.END_TURN),
            )
        return (
            *_ranged_attack_plans(state, actor, enemies),
            ActionPlan(
                ActionCategory.MOVEMENT,
                move_indices=_kite_move_indices(state, actor, masks, self.preferred_min_distance),
            ),
            ActionPlan(ActionCategory.MAIN_ACTION, MainActionType.DODGE),
            ActionPlan(ActionCategory.END_TURN),
        )


class CoverAwareRangedAgent(RangedKitingAgent):
    """Ranged baseline that tries to occupy cover before trading attacks."""

    def default_plan(
        self,
        state: CombatState,
        actor_id: int,
        masks: dict[str, torch.Tensor],
    ) -> Iterable[ActionPlan]:
        actor = state.character_at(actor_id)
        if actor is None:
            return (ActionPlan(ActionCategory.END_TURN),)
        enemies = _enemies(state, actor)
        cover_moves = _cover_move_indices(state, actor, masks)
        if cover_moves and not _has_cover_from_enemy(state, actor):
            return (
                ActionPlan(ActionCategory.MOVEMENT, move_indices=cover_moves),
                *_ranged_attack_plans(state, actor, enemies),
                ActionPlan(ActionCategory.END_TURN),
            )
        return (
            *_ranged_attack_plans(state, actor, enemies),
            ActionPlan(ActionCategory.MOVEMENT, move_indices=cover_moves),
            ActionPlan(ActionCategory.END_TURN),
        )


class SimpleHealerAgent(RuleBasedAgent):
    """Use available healing spells or items for badly injured allies."""

    low_hp_ratio: float = 0.5

    def default_plan(
        self,
        state: CombatState,
        actor_id: int,
        masks: dict[str, torch.Tensor],
    ) -> Iterable[ActionPlan]:
        actor = state.character_at(actor_id)
        if actor is None:
            return (ActionPlan(ActionCategory.END_TURN),)
        wounded = _wounded_ally_ids(state, actor, self.low_hp_ratio)
        healing_action_options = _healing_spell_options(actor, "action")
        healing_bonus_options = _healing_spell_options(actor, "bonus_action")
        healing_item_options = _healing_item_options(actor)
        return (
            ActionPlan(
                ActionCategory.BONUS_ACTION,
                target_indices=wounded,
                option_indices=healing_bonus_options,
            ),
            ActionPlan(
                ActionCategory.MAIN_ACTION,
                MainActionType.CAST_SPELL,
                target_indices=wounded,
                option_indices=healing_action_options,
            ),
            ActionPlan(
                ActionCategory.MAIN_ACTION,
                MainActionType.USE_OBJECT,
                target_indices=wounded,
                option_indices=healing_item_options,
            ),
            ActionPlan(ActionCategory.CLASS_FEATURE, target_indices=wounded),
            *_ranged_attack_plans(state, actor, _enemies(state, actor)),
            ActionPlan(ActionCategory.MAIN_ACTION, MainActionType.DODGE),
            ActionPlan(ActionCategory.END_TURN),
        )


class SimpleCasterAgent(RuleBasedAgent):
    """Cast damaging spells first, then fall back to weapon attacks."""

    def default_plan(
        self,
        state: CombatState,
        actor_id: int,
        masks: dict[str, torch.Tensor],
    ) -> Iterable[ActionPlan]:
        actor = state.character_at(actor_id)
        if actor is None:
            return (ActionPlan(ActionCategory.END_TURN),)
        enemies = _enemies(state, actor)
        damaging_spell_options = _damaging_spell_options(actor)
        return (
            ActionPlan(
                ActionCategory.MAIN_ACTION,
                MainActionType.CAST_SPELL,
                target_indices=_target_ids_by_distance(state, actor, enemies),
                option_indices=damaging_spell_options,
                move_indices=_allowed_indices(masks, "target_cell_index"),
            ),
            *_ranged_attack_plans(state, actor, enemies),
            ActionPlan(ActionCategory.MAIN_ACTION, MainActionType.DODGE),
            ActionPlan(ActionCategory.END_TURN),
        )


class RandomLegalAgent(RuleBasedAgent):
    """Randomly choose among legal mask-compatible actions."""

    def act(
        self,
        observation: Any = None,
        masks: dict[str, torch.Tensor] | None = None,
        *,
        state: CombatState,
        actor_id: int,
        actor: Any | None = None,
        deterministic: bool = False,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        masks = _masks(state, actor_id, masks)
        candidates = _all_legal_outputs(state, actor_id, masks)
        if not candidates:
            return _end_turn_output()
        if deterministic:
            return candidates[0]
        return self.rng.choice(candidates)


@dataclass(frozen=True)
class ActionPlan:
    category: ActionCategory
    main_action: MainActionType | None = None
    target_indices: Iterable[int] | None = None
    option_indices: Iterable[int] | None = None
    move_indices: Iterable[int] | None = None


def _ranged_attack_plans(
    state: CombatState,
    actor: Any,
    enemies: list[Any],
) -> tuple[ActionPlan, ...]:
    return (
        ActionPlan(
            ActionCategory.MAIN_ACTION,
            MainActionType.ATTACK,
            target_indices=_target_ids_by_distance(state, actor, enemies),
            option_indices=_weapon_options(actor, ranged=True),
        ),
        ActionPlan(
            ActionCategory.MAIN_ACTION,
            MainActionType.ATTACK,
            target_indices=_target_ids_by_distance(state, actor, enemies),
        ),
    )


def _first_legal_output(
    state: CombatState,
    actor_id: int,
    masks: dict[str, torch.Tensor],
    plans: Iterable[ActionPlan],
) -> dict[str, torch.Tensor]:
    for plan in plans:
        for output in _outputs_for_plan(state, actor_id, masks, plan):
            if _is_legal_output(state, actor_id, output):
                return output
    return _end_turn_output()


def _all_legal_outputs(
    state: CombatState,
    actor_id: int,
    masks: dict[str, torch.Tensor],
) -> list[dict[str, torch.Tensor]]:
    plans: list[ActionPlan] = []
    for category_index in _allowed_indices(masks, "action_category"):
        category = ActionCategory(category_index)
        if category is ActionCategory.MAIN_ACTION:
            for main_index in _allowed_indices(masks, "main_action_type"):
                plans.append(ActionPlan(category, MainActionType(main_index)))
        else:
            plans.append(ActionPlan(category))
    outputs: list[dict[str, torch.Tensor]] = []
    seen: set[tuple[int, int, int, int, int]] = set()
    for plan in plans:
        for output in _outputs_for_plan(state, actor_id, masks, plan):
            key = _output_key(output)
            if key not in seen and _is_legal_output(state, actor_id, output):
                seen.add(key)
                outputs.append(output)
    return outputs


def _outputs_for_plan(
    state: CombatState,
    actor_id: int,
    masks: dict[str, torch.Tensor],
    plan: ActionPlan,
) -> Iterable[dict[str, torch.Tensor]]:
    if not _mask_allows(masks, "action_category", int(plan.category)):
        return ()
    main_action = plan.main_action or MainActionType.ATTACK
    if plan.category is ActionCategory.MAIN_ACTION and not _mask_allows(
        masks,
        "main_action_type",
        int(main_action),
    ):
        return ()

    target_indices = _candidate_indices(
        plan.target_indices,
        _allowed_indices(masks, "target_index"),
    )
    option_indices = _candidate_indices(
        plan.option_indices,
        _allowed_indices(masks, "option_index"),
    )
    move_key = (
        "target_cell_index"
        if plan.category is ActionCategory.MAIN_ACTION
        and main_action in {MainActionType.CAST_SPELL, MainActionType.USE_OBJECT}
        and _allowed_indices(masks, "target_cell_index")
        else "move_index"
    )
    move_indices = _candidate_indices(plan.move_indices, _allowed_indices(masks, move_key))
    if plan.category is ActionCategory.END_TURN:
        target_indices = [0]
        option_indices = [0]
        move_indices = [0]
    if plan.category is ActionCategory.MOVEMENT:
        target_indices = [0]
        option_indices = [0]
    for target_index in target_indices:
        for option_index in option_indices:
            for move_index in move_indices:
                yield _action_output(
                    action_category=int(plan.category),
                    main_action_type=int(main_action),
                    target_index=target_index,
                    move_index=move_index,
                    option_index=option_index,
                )


def _is_legal_output(
    state: CombatState,
    actor_id: int,
    output: dict[str, torch.Tensor],
) -> bool:
    try:
        decode_action(
            int(output["action_category"].item()),
            int(output["main_action_type"].item()),
            int(output["target_index"].item()),
            int(output["move_index"].item()),
            int(output["option_index"].item()),
            state,
            actor_id,
        )
    except ValueError:
        return False
    return True


def _candidate_indices(
    preferred: Iterable[int] | None,
    fallback: list[int],
) -> list[int]:
    combined: list[int] = []
    for value in tuple(preferred or ()):
        if int(value) not in combined:
            combined.append(int(value))
    for value in fallback:
        if int(value) not in combined:
            combined.append(int(value))
    return combined or [0]


def _allowed_indices(
    masks: dict[str, torch.Tensor],
    key: str,
) -> list[int]:
    mask = masks.get(key)
    if mask is None:
        return []
    prepared = mask.detach().cpu().bool()
    if prepared.ndim == 2:
        prepared = prepared[0]
    if prepared.ndim != 1:
        return []
    return [int(index.item()) for index in torch.nonzero(prepared, as_tuple=False).reshape(-1)]


def _mask_allows(masks: dict[str, torch.Tensor], key: str, index: int) -> bool:
    return index in _allowed_indices(masks, key)


def _masks(
    state: CombatState,
    actor_id: int,
    masks: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    return build_action_masks(state, actor_id) if masks is None else masks


def _action_output(
    *,
    action_category: int,
    main_action_type: int = 0,
    target_index: int = 0,
    move_index: int = 0,
    option_index: int = 0,
) -> dict[str, torch.Tensor]:
    return {
        "action_category": torch.tensor(action_category, dtype=torch.long),
        "main_action_type": torch.tensor(main_action_type, dtype=torch.long),
        "target_index": torch.tensor(target_index, dtype=torch.long),
        "move_index": torch.tensor(move_index, dtype=torch.long),
        "option_index": torch.tensor(option_index, dtype=torch.long),
        "log_prob": torch.tensor(0.0, dtype=torch.float32),
        "entropy": torch.tensor(0.0, dtype=torch.float32),
        "value": torch.tensor(0.0, dtype=torch.float32),
    }


def _end_turn_output() -> dict[str, torch.Tensor]:
    return _action_output(action_category=int(ActionCategory.END_TURN))


def _output_key(output: dict[str, torch.Tensor]) -> tuple[int, int, int, int, int]:
    return (
        int(output["action_category"].item()),
        int(output["main_action_type"].item()),
        int(output["target_index"].item()),
        int(output["move_index"].item()),
        int(output["option_index"].item()),
    )


def _enemies(state: CombatState, actor: Any) -> list[Any]:
    return [
        character
        for character in state.characters
        if character.team != actor.team and character.is_alive
    ]


def _target_ids_by_distance(
    state: CombatState,
    actor: Any,
    targets: Iterable[Any],
) -> list[int]:
    indexed = [
        (target_id, target)
        for target_id, target in enumerate(state.characters)
        if target in targets
    ]
    return [
        target_id
        for target_id, _target in sorted(
            indexed,
            key=lambda item: (
                _distance(state, actor.position, item[1].position),
                item[1].hp,
                item[1].name,
            ),
        )
    ]


def _wounded_ally_ids(
    state: CombatState,
    actor: Any,
    low_hp_ratio: float,
) -> list[int]:
    wounded = [
        (target_id, ally)
        for target_id, ally in enumerate(state.characters)
        if ally.team == actor.team
        and ally.is_alive
        and ally.hp < ally.max_hp
        and (ally.hp / max(1, ally.max_hp)) <= low_hp_ratio
    ]
    return [
        target_id
        for target_id, _ally in sorted(
            wounded,
            key=lambda item: (item[1].hp / max(1, item[1].max_hp), item[1].hp),
        )
    ]


def _weapon_options(actor: Any, *, ranged: bool) -> list[int]:
    options = []
    for index, weapon in enumerate(getattr(actor, "weapons", ())):
        weapon_range = int(getattr(weapon, "range", 1))
        if ranged and weapon_range > 1:
            options.append(index)
        elif not ranged and weapon_range <= 1:
            options.append(index)
    return options


def _healing_spell_options(actor: Any, action_cost: str) -> list[int]:
    spells = [
        spell
        for spell in available_castable_spells(actor)
        if getattr(spell, "action_cost", "action") == action_cost
    ]
    return [
        index
        for index, spell in enumerate(spells)
        if getattr(spell, "healing", None) is not None
    ]


def _damaging_spell_options(actor: Any) -> list[int]:
    spells = [
        spell
        for spell in available_castable_spells(actor)
        if getattr(spell, "action_cost", "action") == "action"
    ]
    return [
        index
        for index, spell in enumerate(spells)
        if getattr(spell, "damage", None) is not None
    ]


def _healing_item_options(actor: Any) -> list[int]:
    inventory = getattr(actor, "inventory", ())
    return [
        index
        for index, item in enumerate(inventory)
        if getattr(item, "implemented", False)
        and int(getattr(item, "quantity", 0)) > 0
        and item_healing(item) is not None
    ]


def _move_indices_toward_enemy(
    state: CombatState,
    actor: Any,
    masks: dict[str, torch.Tensor],
) -> list[int]:
    enemies = _enemies(state, actor)
    if not enemies or state.grid_map is None:
        return _allowed_indices(masks, "move_index")
    return _sort_move_indices(
        state,
        actor,
        masks,
        key=lambda position: (
            min(_distance(state, position, enemy.position) for enemy in enemies),
            _terrain_cost(state, position),
        ),
        reverse=False,
    )


def _kite_move_indices(
    state: CombatState,
    actor: Any,
    masks: dict[str, torch.Tensor],
    preferred_min_distance: int,
) -> list[int]:
    enemies = _enemies(state, actor)
    if not enemies or state.grid_map is None:
        return _allowed_indices(masks, "move_index")
    return _sort_move_indices(
        state,
        actor,
        masks,
        key=lambda position: (
            min(_distance(state, position, enemy.position) for enemy in enemies),
            int(_is_cover_cell(state, position)),
            -_terrain_cost(state, position),
        ),
        reverse=True,
        predicate=lambda position: min(
            _distance(state, position, enemy.position) for enemy in enemies
        )
        >= preferred_min_distance,
    )


def _cover_move_indices(
    state: CombatState,
    actor: Any,
    masks: dict[str, torch.Tensor],
) -> list[int]:
    if state.grid_map is None:
        return []
    return _sort_move_indices(
        state,
        actor,
        masks,
        key=lambda position: (
            int(_is_cover_cell(state, position)),
            min(
                (
                    _distance(state, position, enemy.position)
                    for enemy in _enemies(state, actor)
                ),
                default=0,
            ),
            -_terrain_cost(state, position),
        ),
        reverse=True,
        predicate=lambda position: _is_cover_cell(state, position),
    )


def _sort_move_indices(
    state: CombatState,
    actor: Any,
    masks: dict[str, torch.Tensor],
    *,
    key: Any,
    reverse: bool,
    predicate: Any | None = None,
) -> list[int]:
    indexed: list[tuple[int, Position]] = []
    for move_index in _allowed_indices(masks, "move_index"):
        position = _position_from_move_index(state, move_index)
        if predicate is not None and not predicate(position):
            continue
        indexed.append((move_index, position))
    return [
        move_index
        for move_index, _position in sorted(
            indexed,
            key=lambda item: key(item[1]),
            reverse=reverse,
        )
    ]


def _has_cover_from_enemy(state: CombatState, actor: Any) -> bool:
    if state.grid_map is None:
        return False
    return any(
        state.grid_map.get_cover_between(enemy.position, actor.position)
        is not CoverType.NO_COVER
        for enemy in _enemies(state, actor)
    )


def _is_cover_cell(state: CombatState, position: Position) -> bool:
    if state.grid_map is None or not state.grid_map.in_bounds(position):
        return False
    return state.grid_map.terrain_at(position) in {
        TerrainType.LOW_COVER,
        TerrainType.HIGH_COVER,
    }


def _terrain_cost(state: CombatState, position: Position) -> int:
    if state.grid_map is None or not state.grid_map.in_bounds(position):
        return 99
    cost = state.grid_map.movement_cost(position)
    return 99 if cost is None else int(cost)


def _position_from_move_index(state: CombatState, move_index: int) -> Position:
    if state.grid_map is None:
        return Position()
    return Position(move_index % state.grid_map.width, move_index // state.grid_map.width)


def _distance(state: CombatState, first: Position, second: Position) -> int:
    if state.grid_map is not None:
        return state.grid_map.manhattan_distance(first, second)
    return abs(first.x - second.x) + abs(first.y - second.y)


__all__ = [
    "AggressiveMeleeAgent",
    "CoverAwareRangedAgent",
    "RandomLegalAgent",
    "RangedKitingAgent",
    "RuleBasedAgent",
    "SimpleCasterAgent",
    "SimpleHealerAgent",
]
