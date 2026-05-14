"""Observation encoding for neural network agents."""

from __future__ import annotations

from typing import Iterable

import torch

from combat.models import Character, CombatState, Position, Team, WeaponAttack


MAX_NEARBY_CHARACTERS = 4
CHARACTER_FEATURE_SIZE = 20
OBSERVATION_SIZE = CHARACTER_FEATURE_SIZE * (1 + MAX_NEARBY_CHARACTERS * 2)


def encode_observation(state: CombatState, actor_id: int) -> torch.Tensor:
    """Encode combat state from one actor's perspective as a fixed vector."""

    actor = state.character_at(actor_id)
    if actor is None:
        raise ValueError(f"Actor {actor_id} not found")

    allies = _nearest_characters(
        state,
        actor,
        (
            character
            for index, character in enumerate(state.characters)
            if index != actor_id and character.team == actor.team
        ),
    )
    enemies = _nearest_characters(
        state,
        actor,
        (
            character
            for character in state.characters
            if character.team != actor.team
        ),
    )

    features = [
        *_encode_character(actor, actor, state, present=True),
        *_encode_padded_group(allies, actor, state),
        *_encode_padded_group(enemies, actor, state),
    ]
    return torch.tensor(features, dtype=torch.float32)


def _encode_padded_group(
    characters: list[Character],
    actor: Character,
    state: CombatState,
) -> list[float]:
    features: list[float] = []
    for index in range(MAX_NEARBY_CHARACTERS):
        if index < len(characters):
            features.extend(_encode_character(characters[index], actor, state, present=True))
        else:
            features.extend([0.0] * CHARACTER_FEATURE_SIZE)
    return features


def _encode_character(
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


def _nearest_characters(
    state: CombatState,
    actor: Character,
    characters: Iterable[Character],
) -> list[Character]:
    return sorted(
        characters,
        key=lambda character: (
            _distance(actor.position, character.position, state),
            character.position.x,
            character.position.y,
            character.name,
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
