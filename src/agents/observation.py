"""Observation encoding for neural network agents."""

from __future__ import annotations

from typing import Iterable

import torch

from combat.abilities import SpellAbility, WeaponAttack
from combat.common_actions import (
    COMMON_ACTION_ATTACK,
    COMMON_ACTION_CAST_SPELL,
    COMMON_ACTION_DASH,
    COMMON_ACTION_DISENGAGE,
    COMMON_ACTION_DODGE,
    COMMON_ACTION_GRAPPLE,
    COMMON_ACTION_HELP,
    COMMON_ACTION_HIDE,
    COMMON_ACTION_SHOVE,
)
from combat.class_features import (
    available_implemented_class_features,
    implemented_class_features,
    implemented_feature_active_actions,
)
from combat.cover import CoverType
from combat.models import Character, CombatState, Position, Team
from combat.terrain import TerrainType


MAX_NEARBY_CHARACTERS = 4
BASE_CHARACTER_FEATURE_SIZE = 20
ACTOR_CLASS_FEATURE_SIZE = 6
ACTOR_MAP_FEATURE_SIZE = 10
OTHER_MAP_FEATURE_SIZE = 2
ACTOR_FEATURE_SIZE = (
    BASE_CHARACTER_FEATURE_SIZE
    + 18
    + ACTOR_MAP_FEATURE_SIZE
    + ACTOR_CLASS_FEATURE_SIZE
)
OTHER_CHARACTER_FEATURE_SIZE = BASE_CHARACTER_FEATURE_SIZE + 9 + OTHER_MAP_FEATURE_SIZE
CHARACTER_FEATURE_SIZE = OTHER_CHARACTER_FEATURE_SIZE
OBSERVATION_SIZE = (
    ACTOR_FEATURE_SIZE
    + OTHER_CHARACTER_FEATURE_SIZE * MAX_NEARBY_CHARACTERS * 2
)


def encode_observation(state: CombatState, actor_id: int) -> torch.Tensor:
    """Encode combat state from one actor's perspective as a fixed vector."""

    actor = state.character_at(actor_id)
    if actor is None:
        raise ValueError(f"Actor {actor_id} not found")

    allies = _nearest_character_entries(
        state,
        actor,
        (
            (index, character)
            for index, character in enumerate(state.characters)
            if index != actor_id and character.team == actor.team
        ),
    )
    enemies = _nearest_character_entries(
        state,
        actor,
        (
            (index, character)
            for index, character in enumerate(state.characters)
            if character.team != actor.team
        ),
    )

    features = [
        *_encode_actor(actor, actor_id, state),
        *_encode_padded_group(allies, actor, state),
        *_encode_padded_group(enemies, actor, state),
    ]
    return torch.tensor(features, dtype=torch.float32)


def _encode_actor(
    actor: Character,
    actor_id: int,
    state: CombatState,
) -> list[float]:
    return [
        *_encode_base_character(actor, actor, state, present=True),
        float(actor.action_economy.free_object_interaction_available),
        float(actor.prone),
        float(actor.grappled),
        float(actor.hidden),
        float(actor.dodging_until_start_of_next_turn),
        float(actor.disengaged_until_end_of_turn),
        float(actor.prepared_action is not None),
        float(len(actor.weapons)),
        float(_has_spells(actor)),
        float(_can_cast_spell(state, actor)),
        float(_can_attack(state, actor)),
        float(_can_dash(actor)),
        float(_can_disengage(actor)),
        float(_can_dodge(actor)),
        float(_can_hide(state, actor)),
        float(_can_help(state, actor_id, actor)),
        float(_can_grapple(state, actor_id, actor)),
        float(_can_shove(state, actor_id, actor)),
        *_encode_actor_map_features(actor, state),
        *_encode_actor_class_features(actor),
    ]


def _encode_actor_class_features(actor: Character) -> list[float]:
    implemented_features = implemented_class_features(actor)
    available_features = available_implemented_class_features(actor)
    active_actions = implemented_feature_active_actions(actor)
    available_active_actions = tuple(
        action
        for feature in available_features
        if feature.active_action is not None
        for action in (feature.active_action,)
    )
    return [
        float(len(implemented_features)),
        float(len(available_features)),
        float(_has_implemented_feature(actor, "Spellcasting")),
        float(_has_implemented_feature(actor, "Ability Score Improvement")),
        float(bool(active_actions)),
        float(bool(available_active_actions)),
    ]


def _encode_padded_group(
    entries: list[tuple[int, Character]],
    actor: Character,
    state: CombatState,
) -> list[float]:
    features: list[float] = []
    for index in range(MAX_NEARBY_CHARACTERS):
        if index < len(entries):
            _, character = entries[index]
            features.extend(_encode_other_character(character, actor, state))
        else:
            features.extend([0.0] * OTHER_CHARACTER_FEATURE_SIZE)
    return features


def _encode_other_character(
    character: Character,
    actor: Character,
    state: CombatState,
) -> list[float]:
    return [
        *_encode_base_character(character, actor, state, present=True),
        float(character.prone),
        float(character.grappled),
        float(character.hidden),
        float(character.dodging_until_start_of_next_turn),
        float(_is_in_melee_reach(actor, character, state)),
        float(_can_attack_target(state, actor, character)),
        float(_can_help_against_target(state, actor, character)),
        float(_can_grapple_target(state, actor, character)),
        float(_can_shove_target(state, actor, character)),
        _cover_value(_cover_between(state, actor.position, character.position)),
        float(_has_line_of_sight(state, actor.position, character.position)),
    ]


def _encode_actor_map_features(actor: Character, state: CombatState) -> list[float]:
    directions = (
        Position(actor.position.x, actor.position.y - 1),
        Position(actor.position.x + 1, actor.position.y),
        Position(actor.position.x, actor.position.y + 1),
        Position(actor.position.x - 1, actor.position.y),
    )
    terrain_values = [_terrain_value(state, position) for position in directions]
    movement_costs = [_movement_cost_value(state, position) for position in directions]
    reachable_costs = _reachable_movement_costs(state, actor)
    positive_costs = [cost for position, cost in reachable_costs.items() if position != actor.position]
    average_cost = sum(positive_costs) / len(positive_costs) if positive_costs else 0.0
    return [
        *terrain_values,
        *movement_costs,
        float(len(positive_costs)),
        float(average_cost),
    ]


def _encode_base_character(
    character: Character,
    reference: Character,
    state: CombatState,
    present: bool,
) -> list[float]:
    hp_ratio = character.hp / character.max_hp if character.max_hp > 0 else 0.0
    movement_ratio = (
        character.action_economy.movement_remaining / character.speed
        if character.speed > 0
        else 0.0
    )
    distance = _distance(reference.position, character.position, state)
    return [
        float(present),
        float(character.hp),
        float(character.max_hp),
        float(hp_ratio),
        float(character.ac),
        float(character.position.x),
        float(character.position.y),
        _team_flag(character.team, Team.PLAYERS),
        _team_flag(character.team, Team.ENEMIES),
        float(_has_melee_attack(character)),
        float(_has_ranged_attack(character)),
        float(character.is_alive),
        float(character.is_dead),
        float(character.action_economy.action_available),
        float(character.action_economy.bonus_action_available),
        float(character.action_economy.reaction_available),
        float(character.action_economy.movement_remaining),
        float(character.speed),
        float(movement_ratio),
        float(distance if present else 0),
    ]


def _nearest_character_entries(
    state: CombatState,
    actor: Character,
    entries: Iterable[tuple[int, Character]],
) -> list[tuple[int, Character]]:
    return sorted(
        entries,
        key=lambda item: (
            _distance(actor.position, item[1].position, state),
            item[1].position.x,
            item[1].position.y,
            item[1].name,
        ),
    )[:MAX_NEARBY_CHARACTERS]


def _distance(first: Position, second: Position, state: CombatState) -> int:
    if state.grid_map is not None:
        return state.grid_map.manhattan_distance(first, second)
    return abs(first.x - second.x) + abs(first.y - second.y)


def _team_flag(team: Team, expected: Team) -> float:
    return 1.0 if team is expected else 0.0


def _has_melee_attack(character: Character) -> bool:
    return any(
        isinstance(weapon, WeaponAttack) and weapon.range <= 1
        for weapon in character.weapons
    )


def _has_ranged_attack(character: Character) -> bool:
    return any(
        isinstance(weapon, WeaponAttack) and weapon.range > 1
        for weapon in character.weapons
    )


def _has_spells(character: Character) -> bool:
    return any(isinstance(ability, SpellAbility) for ability in character.abilities)


def _has_implemented_feature(character: Character, feature_name: str) -> bool:
    feature_key = _lookup_key(feature_name)
    return any(
        _lookup_key(feature.name) == feature_key
        for feature in implemented_class_features(character)
    )


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
        for spell in _available_spells(actor)
    )


def _can_attack(state: CombatState, actor: Character) -> bool:
    return any(
        _can_attack_target(state, actor, target)
        for target in state.characters
        if target is not actor
    )


def _can_dash(actor: Character) -> bool:
    return _can_spend_action(actor, COMMON_ACTION_DASH)


def _can_disengage(actor: Character) -> bool:
    return (
        _can_spend_action(actor, COMMON_ACTION_DISENGAGE)
        and not actor.disengaged_until_end_of_turn
    )


def _can_dodge(actor: Character) -> bool:
    return (
        _can_spend_action(actor, COMMON_ACTION_DODGE)
        and not actor.dodging_until_start_of_next_turn
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


def _can_help(state: CombatState, actor_id: int, actor: Character) -> bool:
    if not _can_spend_action(actor, COMMON_ACTION_HELP):
        return False
    return any(
        target_id != actor_id and not target.is_dead
        for target_id, target in enumerate(state.characters)
    )


def _can_grapple(state: CombatState, actor_id: int, actor: Character) -> bool:
    return any(
        target_id != actor_id and _can_grapple_target(state, actor, target)
        for target_id, target in enumerate(state.characters)
    )


def _can_shove(state: CombatState, actor_id: int, actor: Character) -> bool:
    return any(
        target_id != actor_id and _can_shove_target(state, actor, target)
        for target_id, target in enumerate(state.characters)
    )


def _can_attack_target(
    state: CombatState,
    actor: Character,
    target: Character,
) -> bool:
    if not _can_spend_action(actor, COMMON_ACTION_ATTACK):
        return False
    return any(
        _is_valid_weapon_target(state, actor, target, weapon)
        for weapon in actor.available_weapons
    )


def _is_valid_weapon_target(
    state: CombatState,
    actor: Character,
    target: Character,
    weapon: WeaponAttack,
) -> bool:
    return (
        target is not actor
        and target.team != actor.team
        and target.is_alive
        and weapon.available
        and _distance(actor.position, target.position, state) <= weapon.range
        and _has_line_of_sight(state, actor.position, target.position)
        and _cover_between(state, actor.position, target.position) is not CoverType.FULL_COVER
    )


def _can_help_against_target(
    state: CombatState,
    actor: Character,
    target: Character,
) -> bool:
    return (
        _can_spend_action(actor, COMMON_ACTION_HELP)
        and target is not actor
        and target.team != actor.team
        and target.is_alive
    )


def _can_grapple_target(
    state: CombatState,
    actor: Character,
    target: Character,
) -> bool:
    return _can_special_melee_target(
        state,
        actor,
        target,
        COMMON_ACTION_GRAPPLE,
    )


def _can_shove_target(
    state: CombatState,
    actor: Character,
    target: Character,
) -> bool:
    return _can_special_melee_target(
        state,
        actor,
        target,
        COMMON_ACTION_SHOVE,
    )


def _can_special_melee_target(
    state: CombatState,
    actor: Character,
    target: Character,
    action_name: str,
) -> bool:
    return (
        _can_spend_action(actor, action_name)
        and COMMON_ACTION_ATTACK in actor.common_actions
        and target is not actor
        and target.team != actor.team
        and target.is_alive
        and _is_in_melee_reach(actor, target, state)
        and _has_line_of_sight(state, actor.position, target.position)
    )


def _is_in_melee_reach(
    actor: Character,
    target: Character,
    state: CombatState,
) -> bool:
    return _distance(actor.position, target.position, state) <= 1


def _available_spells(actor: Character) -> list[SpellAbility]:
    return [
        ability
        for ability in actor.available_abilities
        if isinstance(ability, SpellAbility) and _has_spell_slot(actor, ability)
    ]


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
    target: Character,
    spell: SpellAbility,
) -> bool:
    return (
        target is not actor
        and target.team != actor.team
        and target.is_alive
        and _distance(actor.position, target.position, state) <= spell.range
        and _has_line_of_sight(state, actor.position, target.position)
        and _cover_between(state, actor.position, target.position) is not CoverType.FULL_COVER
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


def _cover_value(cover: CoverType) -> float:
    return float(
        {
            CoverType.NO_COVER: 0,
            CoverType.HALF_COVER: 1,
            CoverType.THREE_QUARTERS_COVER: 2,
            CoverType.FULL_COVER: 3,
        }[cover]
    )


def _terrain_value(state: CombatState, position: Position) -> float:
    grid_map = state.grid_map
    if grid_map is None or not grid_map.in_bounds(position):
        return float(_terrain_index(TerrainType.BLOCKED))
    return float(_terrain_index(grid_map.terrain_at(position)))


def _movement_cost_value(state: CombatState, position: Position) -> float:
    grid_map = state.grid_map
    if grid_map is None or not grid_map.in_bounds(position):
        return 0.0
    movement_cost = grid_map.movement_cost(position)
    return float(movement_cost or 0)


def _reachable_movement_costs(state: CombatState, actor: Character) -> dict[Position, int]:
    grid_map = state.grid_map
    if grid_map is None:
        return {}
    return grid_map.movement_costs_from(
        actor.position,
        actor.action_economy.movement_remaining,
        state.characters,
    )


def _terrain_index(terrain_type: TerrainType) -> int:
    return {
        TerrainType.NORMAL: 0,
        TerrainType.DIFFICULT_TERRAIN: 1,
        TerrainType.BLOCKED: 2,
        TerrainType.LOW_COVER: 3,
        TerrainType.HIGH_COVER: 4,
    }[terrain_type]


def _lookup_key(value: object) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())
