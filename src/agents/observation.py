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
from combat.damage import (
    DAMAGE_TYPES,
    character_immunities,
    character_resistances,
    character_vulnerabilities,
    coerce_damage_type,
)
from combat.cover import CoverType
from combat.models import Character, CombatState, Position, Team
from combat.spellcasting import (
    available_castable_spells,
    can_target_spell as can_target_spell_with_rules,
    spell_system_available,
)
from combat.terrain import TerrainType
from agents.entity_observation import (
    CONDITION_FLAG_NAMES,
    EntityObservation,
    FEAT_FLAG_NAMES,
    INVENTORY_ITEM_FLAG_NAMES,
    LOCAL_MAP_CELL_COUNT,
    LOCAL_MAP_RADIUS,
    MAP_CELL_FEATURE_SIZE,
    MAP_FEATURE_SIZE,
    PREPARED_SPELL_FLAG_NAMES,
    TERRAIN_FEATURE_TYPES,
    active_concentration_flag,
    available_damage_type_flags,
    class_id,
    class_resource_flags,
    condition_flags,
    current_cover_status,
    feat_flags,
    global_feature_values,
    inventory_usable_item_flags,
    normalized_challenge_rating,
    normalized_level,
    normalized_proficiency_bonus,
    normalized_xp_value,
    prepared_spell_flags,
    race_id,
    reachable_by_actor,
    role_id,
    spell_slot_features,
    subclass_id,
    terrain_around_features,
    threat_estimate,
    visible_enemies_count,
)


MAX_NEARBY_CHARACTERS = 4
MAX_ENTITY_COUNT = MAX_NEARBY_CHARACTERS * 2
BASE_CHARACTER_FEATURE_SIZE = 20
DAMAGE_TYPE_FEATURE_SIZE = len(DAMAGE_TYPES)
ACTOR_DAMAGE_ACTION_FEATURE_SIZE = DAMAGE_TYPE_FEATURE_SIZE
ACTOR_COMMON_ACTION_FEATURE_SIZE = 18
FEAT_FEATURE_SIZE = len(FEAT_FLAG_NAMES)
CLASS_RESOURCE_FEATURE_SIZE = 4
SPELL_SLOT_FEATURE_SIZE = 6
PREPARED_SPELL_FEATURE_SIZE = len(PREPARED_SPELL_FLAG_NAMES)
INVENTORY_ITEM_FEATURE_SIZE = len(INVENTORY_ITEM_FLAG_NAMES)
TERRAIN_AROUND_FEATURE_SIZE = 4 * len(TERRAIN_FEATURE_TYPES)
ACTOR_REAL_GAME_FEATURE_SIZE = (
    6
    + FEAT_FEATURE_SIZE
    + 4
    + CLASS_RESOURCE_FEATURE_SIZE
    + SPELL_SLOT_FEATURE_SIZE
    + PREPARED_SPELL_FEATURE_SIZE
    + INVENTORY_ITEM_FEATURE_SIZE
    + 1
    + TERRAIN_AROUND_FEATURE_SIZE
    + 1
)
OTHER_DAMAGE_PROFILE_FEATURE_SIZE = DAMAGE_TYPE_FEATURE_SIZE * 3
OTHER_COMMON_ACTION_FEATURE_SIZE = 9
OTHER_CONDITION_FEATURE_SIZE = len(CONDITION_FLAG_NAMES)
OTHER_ENTITY_PROFILE_FEATURE_SIZE = 3 + 2 + OTHER_CONDITION_FEATURE_SIZE + 1 + 2 + 1
OTHER_MAP_FEATURE_SIZE = 4
ACTOR_CLASS_FEATURE_SIZE = 6
ACTOR_MAP_FEATURE_SIZE = 10
GLOBAL_FEATURE_SIZE = 7
ACTOR_COMMON_ACTION_OFFSET = BASE_CHARACTER_FEATURE_SIZE
ACTOR_REAL_GAME_OFFSET = ACTOR_COMMON_ACTION_OFFSET + ACTOR_COMMON_ACTION_FEATURE_SIZE
ACTOR_DAMAGE_ACTION_OFFSET = ACTOR_REAL_GAME_OFFSET + ACTOR_REAL_GAME_FEATURE_SIZE
ACTOR_MAP_FEATURE_OFFSET = ACTOR_DAMAGE_ACTION_OFFSET + ACTOR_DAMAGE_ACTION_FEATURE_SIZE
ACTOR_CLASS_FEATURE_OFFSET = ACTOR_MAP_FEATURE_OFFSET + ACTOR_MAP_FEATURE_SIZE
OTHER_COMMON_ACTION_OFFSET = BASE_CHARACTER_FEATURE_SIZE
OTHER_DAMAGE_PROFILE_OFFSET = OTHER_COMMON_ACTION_OFFSET + OTHER_COMMON_ACTION_FEATURE_SIZE
OTHER_ENTITY_PROFILE_OFFSET = OTHER_DAMAGE_PROFILE_OFFSET + OTHER_DAMAGE_PROFILE_FEATURE_SIZE
OTHER_MAP_FEATURE_OFFSET = OTHER_ENTITY_PROFILE_OFFSET + OTHER_ENTITY_PROFILE_FEATURE_SIZE
ACTOR_FEATURE_SIZE = (
    BASE_CHARACTER_FEATURE_SIZE
    + ACTOR_COMMON_ACTION_FEATURE_SIZE
    + ACTOR_REAL_GAME_FEATURE_SIZE
    + ACTOR_DAMAGE_ACTION_FEATURE_SIZE
    + ACTOR_MAP_FEATURE_SIZE
    + ACTOR_CLASS_FEATURE_SIZE
)
OTHER_CHARACTER_FEATURE_SIZE = (
    BASE_CHARACTER_FEATURE_SIZE
    + OTHER_COMMON_ACTION_FEATURE_SIZE
    + OTHER_DAMAGE_PROFILE_FEATURE_SIZE
    + OTHER_ENTITY_PROFILE_FEATURE_SIZE
    + OTHER_MAP_FEATURE_SIZE
)
CHARACTER_FEATURE_SIZE = OTHER_CHARACTER_FEATURE_SIZE
OBSERVATION_SIZE = (
    ACTOR_FEATURE_SIZE
    + OTHER_CHARACTER_FEATURE_SIZE * MAX_NEARBY_CHARACTERS * 2
    + GLOBAL_FEATURE_SIZE
)
PPO_INPUT_SIZE = OBSERVATION_SIZE
ENTITY_EXTRA_FEATURE_SIZE = 1
ENTITY_FEATURE_SIZE = OTHER_CHARACTER_FEATURE_SIZE + ENTITY_EXTRA_FEATURE_SIZE
GNN_NODE_FEATURE_SIZE = ENTITY_FEATURE_SIZE
ENTITY_MASK_SIZE = MAX_ENTITY_COUNT
ENTITY_GLOBAL_FEATURE_SIZE = GLOBAL_FEATURE_SIZE + 1


def encode_observation(state: CombatState, actor_id: int) -> torch.Tensor:
    """Encode combat state as a flattened MLP-compatible observation vector."""

    return flatten_entity_observation(encode_entity_observation(state, actor_id))


def encode_entity_observation(state: CombatState, actor_id: int) -> EntityObservation:
    """Encode combat state from one actor's perspective as entity-based tensors."""

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

    entity_rows, entity_mask = _encode_entity_rows(allies, enemies, actor, state)
    return EntityObservation(
        actor_features=torch.tensor(
            _encode_actor(actor, actor_id, state),
            dtype=torch.float32,
        ),
        entities_features=torch.tensor(entity_rows, dtype=torch.float32),
        map_features=torch.tensor(_encode_map_features(state, actor), dtype=torch.float32),
        global_features=torch.tensor(
            _entity_global_feature_values(state, actor_id),
            dtype=torch.float32,
        ),
        entity_mask=torch.tensor(entity_mask, dtype=torch.float32),
    )


def flatten_entity_observation(observation: EntityObservation) -> torch.Tensor:
    """Flatten entity-based observations for the legacy MLP/PPO policy."""

    return torch.cat(
        (
            observation.actor_features.reshape(-1),
            observation.entities_features[:, :OTHER_CHARACTER_FEATURE_SIZE].reshape(-1),
            observation.global_features[:GLOBAL_FEATURE_SIZE].reshape(-1),
        ),
        dim=0,
    ).to(dtype=torch.float32)


def _encode_entity_rows(
    allies: list[tuple[int, Character]],
    enemies: list[tuple[int, Character]],
    actor: Character,
    state: CombatState,
) -> tuple[list[list[float]], list[float]]:
    rows: list[list[float]] = []
    mask: list[float] = []
    for entries in (allies, enemies):
        for index in range(MAX_NEARBY_CHARACTERS):
            if index < len(entries):
                _, character = entries[index]
                rows.append(
                    [
                        *_encode_other_character(character, actor, state),
                        float(_has_spells(character)),
                    ]
                )
                mask.append(1.0)
            else:
                rows.append([0.0] * ENTITY_FEATURE_SIZE)
                mask.append(0.0)
    return rows, mask


def _entity_global_feature_values(state: CombatState, actor_id: int) -> list[float]:
    actor = state.character_at(actor_id)
    current_team = 0.0 if actor is None or actor.team is Team.PLAYERS else 1.0
    return [*global_feature_values(state, actor_id), current_team]


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
        *_encode_actor_real_game_features(actor, state),
        *_encode_available_damage_types(actor),
        *_encode_actor_map_features(actor, state),
        *_encode_actor_class_features(actor),
    ]


def _encode_actor_real_game_features(
    actor: Character,
    state: CombatState,
) -> list[float]:
    return [
        normalized_level(actor),
        normalized_proficiency_bonus(actor),
        class_id(actor),
        subclass_id(actor),
        race_id(actor),
        role_id(actor),
        *feat_flags(actor),
        float(actor.action_economy.action_available),
        float(actor.action_economy.bonus_action_available),
        float(actor.action_economy.reaction_available),
        float(actor.action_economy.movement_remaining),
        *class_resource_flags(actor),
        *spell_slot_features(actor),
        *prepared_spell_flags(actor),
        *inventory_usable_item_flags(actor),
        current_cover_status(state, actor),
        *terrain_around_features(state, actor),
        visible_enemies_count(state, actor),
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
        *_encode_damage_profile(character),
        *_encode_other_entity_profile(character, actor, state),
        *_encode_other_map_features(character, actor, state),
    ]


def _encode_other_entity_profile(
    character: Character,
    actor: Character,
    state: CombatState,
) -> list[float]:
    hp_ratio = character.hp / character.max_hp if character.max_hp > 0 else 0.0
    return [
        class_id(character),
        subclass_id(character),
        role_id(character),
        normalized_challenge_rating(character),
        normalized_xp_value(character),
        *condition_flags(character),
        active_concentration_flag(character),
        float(character.ac),
        float(hp_ratio),
        threat_estimate(character, actor, state),
    ]


def _encode_other_map_features(
    character: Character,
    actor: Character,
    state: CombatState,
) -> list[float]:
    return [
        float(_has_line_of_sight(state, actor.position, character.position)),
        _cover_value(_cover_between(state, actor.position, character.position)),
        float(_distance(actor.position, character.position, state)),
        reachable_by_actor(state, actor, character),
    ]


def _encode_map_features(state: CombatState, actor: Character) -> list[list[float]]:
    grid_map = state.grid_map
    reachable_costs = _reachable_movement_costs(state, actor)
    rows: list[list[float]] = []
    for dy in range(-LOCAL_MAP_RADIUS, LOCAL_MAP_RADIUS + 1):
        for dx in range(-LOCAL_MAP_RADIUS, LOCAL_MAP_RADIUS + 1):
            position = Position(actor.position.x + dx, actor.position.y + dy)
            terrain_type = terrain_at_for_map_features(state, position)
            movement_cost = 0.0
            blocked = True
            if grid_map is not None and grid_map.in_bounds(position):
                cost = grid_map.movement_cost(position)
                movement_cost = float(cost or 0)
                blocked = cost is None
            cover_cell = terrain_type in {
                TerrainType.LOW_COVER,
                TerrainType.HIGH_COVER,
            }
            visible = _has_line_of_sight(state, actor.position, position)
            rows.append(
                [
                    *[float(terrain_type is candidate) for candidate in TERRAIN_FEATURE_TYPES],
                    float(blocked),
                    float(cover_cell),
                    movement_cost,
                    float(position in reachable_costs),
                    float(visible),
                ]
            )
    return rows


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


def _encode_available_damage_types(character: Character) -> list[float]:
    return available_damage_type_flags(character)


def _encode_damage_profile(character: Character) -> list[float]:
    return [
        *_damage_type_flags(character_resistances(character)),
        *_damage_type_flags(character_immunities(character)),
        *_damage_type_flags(character_vulnerabilities(character)),
    ]


def _damage_type_flags(damage_types: set[object]) -> list[float]:
    normalized = {
        damage_type
        for value in damage_types
        for damage_type in (coerce_damage_type(value),)
        if damage_type is not None
    }
    return [float(damage_type in normalized) for damage_type in DAMAGE_TYPES]


def _has_spells(character: Character) -> bool:
    return bool(character.cantrips or character.prepared_spells) or any(
        isinstance(ability, SpellAbility) for ability in character.abilities
    )


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
        for spell in _available_spells(actor, "action")
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


def _available_spells(
    actor: Character,
    action_cost: str | None = None,
) -> list[SpellAbility]:
    spells = available_castable_spells(actor)
    if action_cost is None:
        return spells
    return [spell for spell in spells if spell.action_cost == action_cost]


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
    target: Character,
    spell: SpellAbility,
) -> bool:
    return can_target_spell_with_rules(
        actor,
        target,
        spell,
        distance=_distance(actor.position, target.position, state),
        has_line_of_sight=_has_line_of_sight(state, actor.position, target.position),
        has_full_cover=_cover_between(state, actor.position, target.position)
        is CoverType.FULL_COVER,
    )


def _spell_system_available(actor: Character) -> bool:
    return spell_system_available(actor)


def _has_spell_slot(actor: Character, spell: SpellAbility) -> bool:
    return spell in _available_spells(actor)


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


def terrain_at_for_map_features(state: CombatState, position: Position) -> TerrainType:
    grid_map = state.grid_map
    if grid_map is None or not grid_map.in_bounds(position):
        return TerrainType.BLOCKED
    return grid_map.terrain_at(position)


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
