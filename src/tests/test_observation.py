import pytest
import torch

from agents import (
    CHARACTER_FEATURE_SIZE,
    MAX_NEARBY_CHARACTERS,
    OBSERVATION_SIZE,
    encode_observation,
)
from combat import (
    CombatState,
    FighterArcher,
    FighterChampionGreatsword,
    Goblin,
    GridMap,
    Orc,
    Position,
)


PRESENT = 0
HP = 1
HP_RATIO = 3
X = 5
Y = 6
TEAM_PLAYERS = 7
TEAM_ENEMIES = 8
MELEE = 9
RANGED = 10
ALIVE = 11
DEAD = 12
ACTION_AVAILABLE = 13
MOVEMENT_REMAINING = 16
SPEED = 17
MOVEMENT_RATIO = 18
DISTANCE = 19


def block(observation: torch.Tensor, index: int) -> torch.Tensor:
    start = index * CHARACTER_FEATURE_SIZE
    return observation[start : start + CHARACTER_FEATURE_SIZE]


def test_encode_observation_returns_fixed_tensor_for_actor() -> None:
    actor = FighterArcher(Position(0, 0))
    ally = FighterChampionGreatsword(Position(0, 2))
    goblin = Goblin(Position(1, 0))
    orc = Orc(Position(5, 5))
    state = CombatState(
        characters=[actor, ally, goblin, orc],
        grid_map=GridMap(width=8, height=8),
    )

    observation = encode_observation(state, actor_id=0)
    actor_block = block(observation, 0)

    assert isinstance(observation, torch.Tensor)
    assert observation.dtype == torch.float32
    assert observation.shape == (OBSERVATION_SIZE,)
    assert OBSERVATION_SIZE == CHARACTER_FEATURE_SIZE * (1 + MAX_NEARBY_CHARACTERS * 2)
    assert actor_block[PRESENT] == 1
    assert actor_block[HP] == actor.hp
    assert actor_block[HP_RATIO] == 1
    assert actor_block[X] == 0
    assert actor_block[Y] == 0
    assert actor_block[TEAM_PLAYERS] == 1
    assert actor_block[TEAM_ENEMIES] == 0
    assert actor_block[MELEE] == 0
    assert actor_block[RANGED] == 1
    assert actor_block[ALIVE] == 1
    assert actor_block[DEAD] == 0
    assert actor_block[ACTION_AVAILABLE] == 1
    assert actor_block[MOVEMENT_REMAINING] == actor.speed
    assert actor_block[SPEED] == actor.speed
    assert actor_block[MOVEMENT_RATIO] == 1
    assert actor_block[DISTANCE] == 0


def test_encode_observation_nearest_groups_and_padding() -> None:
    actor = FighterChampionGreatsword(Position(0, 0))
    ally = FighterArcher(Position(0, 3))
    goblin = Goblin(Position(1, 0))
    orc = Orc(Position(7, 7))
    orc.hp = 0
    orc.action_economy.action_available = False
    state = CombatState(
        characters=[actor, ally, goblin, orc],
        grid_map=GridMap(width=8, height=8),
    )

    observation = encode_observation(state, actor_id=0)
    first_ally_block = block(observation, 1)
    second_ally_block = block(observation, 2)
    first_enemy_block = block(observation, 1 + MAX_NEARBY_CHARACTERS)
    second_enemy_block = block(observation, 2 + MAX_NEARBY_CHARACTERS)

    assert first_ally_block[PRESENT] == 1
    assert first_ally_block[RANGED] == 1
    assert first_ally_block[DISTANCE] == 3
    assert torch.count_nonzero(second_ally_block) == 0

    assert first_enemy_block[PRESENT] == 1
    assert first_enemy_block[TEAM_ENEMIES] == 1
    assert first_enemy_block[DISTANCE] == 1
    assert second_enemy_block[PRESENT] == 1
    assert second_enemy_block[ALIVE] == 0
    assert second_enemy_block[DEAD] == 1
    assert second_enemy_block[ACTION_AVAILABLE] == 0


def test_encode_observation_rejects_missing_actor() -> None:
    state = CombatState(characters=[FighterArcher()], grid_map=GridMap(width=8, height=8))

    with pytest.raises(ValueError, match="Actor 5 not found"):
        encode_observation(state, actor_id=5)
