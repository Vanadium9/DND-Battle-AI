import pytest
import torch

from agents import (
    ACTOR_FEATURE_SIZE,
    BASE_CHARACTER_FEATURE_SIZE,
    CHARACTER_FEATURE_SIZE,
    MAX_NEARBY_CHARACTERS,
    OBSERVATION_SIZE,
    OTHER_CHARACTER_FEATURE_SIZE,
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
    TerrainType,
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
ACTOR_FREE_OBJECT_INTERACTION = BASE_CHARACTER_FEATURE_SIZE
ACTOR_PRONE = BASE_CHARACTER_FEATURE_SIZE + 1
ACTOR_GRAPPLED = BASE_CHARACTER_FEATURE_SIZE + 2
ACTOR_HIDDEN = BASE_CHARACTER_FEATURE_SIZE + 3
ACTOR_DODGING = BASE_CHARACTER_FEATURE_SIZE + 4
ACTOR_DISENGAGED = BASE_CHARACTER_FEATURE_SIZE + 5
ACTOR_HAS_PREPARED_ACTION = BASE_CHARACTER_FEATURE_SIZE + 6
ACTOR_NUMBER_OF_WEAPONS = BASE_CHARACTER_FEATURE_SIZE + 7
ACTOR_HAS_SPELLS = BASE_CHARACTER_FEATURE_SIZE + 8
ACTOR_CAN_CAST_SPELL = BASE_CHARACTER_FEATURE_SIZE + 9
ACTOR_CAN_ATTACK = BASE_CHARACTER_FEATURE_SIZE + 10
ACTOR_CAN_DASH = BASE_CHARACTER_FEATURE_SIZE + 11
ACTOR_CAN_DISENGAGE = BASE_CHARACTER_FEATURE_SIZE + 12
ACTOR_CAN_DODGE = BASE_CHARACTER_FEATURE_SIZE + 13
ACTOR_CAN_HIDE = BASE_CHARACTER_FEATURE_SIZE + 14
ACTOR_CAN_HELP = BASE_CHARACTER_FEATURE_SIZE + 15
ACTOR_CAN_GRAPPLE = BASE_CHARACTER_FEATURE_SIZE + 16
ACTOR_CAN_SHOVE = BASE_CHARACTER_FEATURE_SIZE + 17
OTHER_PRONE = BASE_CHARACTER_FEATURE_SIZE
OTHER_GRAPPLED = BASE_CHARACTER_FEATURE_SIZE + 1
OTHER_HIDDEN = BASE_CHARACTER_FEATURE_SIZE + 2
OTHER_DODGING = BASE_CHARACTER_FEATURE_SIZE + 3
OTHER_IN_MELEE_REACH = BASE_CHARACTER_FEATURE_SIZE + 4
OTHER_CAN_BE_ATTACKED = BASE_CHARACTER_FEATURE_SIZE + 5
OTHER_CAN_BE_HELPED_AGAINST = BASE_CHARACTER_FEATURE_SIZE + 6
OTHER_CAN_BE_GRAPPLED = BASE_CHARACTER_FEATURE_SIZE + 7
OTHER_CAN_BE_SHOVED = BASE_CHARACTER_FEATURE_SIZE + 8


def actor_block(observation: torch.Tensor) -> torch.Tensor:
    return observation[:ACTOR_FEATURE_SIZE]


def other_block(observation: torch.Tensor, index: int) -> torch.Tensor:
    start = ACTOR_FEATURE_SIZE + index * OTHER_CHARACTER_FEATURE_SIZE
    return observation[start : start + OTHER_CHARACTER_FEATURE_SIZE]


def test_encode_observation_returns_fixed_tensor_for_actor() -> None:
    actor = FighterArcher(Position(0, 0))
    ally = FighterChampionGreatsword(Position(0, 2))
    goblin = Goblin(Position(1, 0))
    orc = Orc(Position(5, 5))
    state = CombatState(
        characters=[actor, ally, goblin, orc],
        grid_map=GridMap(
            width=8,
            height=8,
            terrain_grid=[
                [TerrainType.LOW_COVER, *([TerrainType.NORMAL] * 7)],
                *([[TerrainType.NORMAL] * 8] * 7),
            ],
        ),
    )

    observation = encode_observation(state, actor_id=0)
    actor_features = actor_block(observation)

    assert isinstance(observation, torch.Tensor)
    assert observation.dtype == torch.float32
    assert observation.shape == (OBSERVATION_SIZE,)
    assert CHARACTER_FEATURE_SIZE == OTHER_CHARACTER_FEATURE_SIZE
    assert OBSERVATION_SIZE == (
        ACTOR_FEATURE_SIZE + OTHER_CHARACTER_FEATURE_SIZE * MAX_NEARBY_CHARACTERS * 2
    )
    assert actor_features[PRESENT] == 1
    assert actor_features[HP] == actor.hp
    assert actor_features[HP_RATIO] == 1
    assert actor_features[X] == 0
    assert actor_features[Y] == 0
    assert actor_features[TEAM_PLAYERS] == 1
    assert actor_features[TEAM_ENEMIES] == 0
    assert actor_features[MELEE] == 0
    assert actor_features[RANGED] == 1
    assert actor_features[ALIVE] == 1
    assert actor_features[DEAD] == 0
    assert actor_features[ACTION_AVAILABLE] == 1
    assert actor_features[MOVEMENT_REMAINING] == actor.speed
    assert actor_features[SPEED] == actor.speed
    assert actor_features[MOVEMENT_RATIO] == 1
    assert actor_features[DISTANCE] == 0
    assert actor_features[ACTOR_FREE_OBJECT_INTERACTION] == 1
    assert actor_features[ACTOR_PRONE] == 0
    assert actor_features[ACTOR_GRAPPLED] == 0
    assert actor_features[ACTOR_HIDDEN] == 0
    assert actor_features[ACTOR_DODGING] == 0
    assert actor_features[ACTOR_DISENGAGED] == 0
    assert actor_features[ACTOR_HAS_PREPARED_ACTION] == 0
    assert actor_features[ACTOR_NUMBER_OF_WEAPONS] == len(actor.weapons)
    assert actor_features[ACTOR_HAS_SPELLS] == 0
    assert actor_features[ACTOR_CAN_CAST_SPELL] == 0
    assert actor_features[ACTOR_CAN_ATTACK] == 1
    assert actor_features[ACTOR_CAN_DASH] == 1
    assert actor_features[ACTOR_CAN_DISENGAGE] == 1
    assert actor_features[ACTOR_CAN_DODGE] == 1
    assert actor_features[ACTOR_CAN_HIDE] == 1
    assert actor_features[ACTOR_CAN_HELP] == 1
    assert actor_features[ACTOR_CAN_GRAPPLE] == 1
    assert actor_features[ACTOR_CAN_SHOVE] == 1


def test_encode_observation_nearest_groups_and_padding() -> None:
    actor = FighterChampionGreatsword(Position(0, 0))
    ally = FighterArcher(Position(0, 3))
    goblin = Goblin(Position(1, 0))
    goblin.prone = True
    goblin.hidden = True
    orc = Orc(Position(7, 7))
    orc.hp = 0
    orc.action_economy.action_available = False
    state = CombatState(
        characters=[actor, ally, goblin, orc],
        grid_map=GridMap(width=8, height=8),
    )

    observation = encode_observation(state, actor_id=0)
    first_ally_block = other_block(observation, 0)
    second_ally_block = other_block(observation, 1)
    first_enemy_block = other_block(observation, MAX_NEARBY_CHARACTERS)
    second_enemy_block = other_block(observation, MAX_NEARBY_CHARACTERS + 1)

    assert first_ally_block[PRESENT] == 1
    assert first_ally_block[RANGED] == 1
    assert first_ally_block[DISTANCE] == 3
    assert torch.count_nonzero(second_ally_block) == 0

    assert first_enemy_block[PRESENT] == 1
    assert first_enemy_block[TEAM_ENEMIES] == 1
    assert first_enemy_block[DISTANCE] == 1
    assert first_enemy_block[OTHER_PRONE] == 1
    assert first_enemy_block[OTHER_GRAPPLED] == 0
    assert first_enemy_block[OTHER_HIDDEN] == 1
    assert first_enemy_block[OTHER_DODGING] == 0
    assert first_enemy_block[OTHER_IN_MELEE_REACH] == 1
    assert first_enemy_block[OTHER_CAN_BE_ATTACKED] == 1
    assert first_enemy_block[OTHER_CAN_BE_HELPED_AGAINST] == 1
    assert first_enemy_block[OTHER_CAN_BE_GRAPPLED] == 1
    assert first_enemy_block[OTHER_CAN_BE_SHOVED] == 1
    assert second_enemy_block[PRESENT] == 1
    assert second_enemy_block[ALIVE] == 0
    assert second_enemy_block[DEAD] == 1
    assert second_enemy_block[ACTION_AVAILABLE] == 0
    assert second_enemy_block[OTHER_CAN_BE_ATTACKED] == 0
    assert second_enemy_block[OTHER_CAN_BE_HELPED_AGAINST] == 0
    assert second_enemy_block[OTHER_CAN_BE_GRAPPLED] == 0
    assert second_enemy_block[OTHER_CAN_BE_SHOVED] == 0


def test_encode_observation_rejects_missing_actor() -> None:
    state = CombatState(characters=[FighterArcher()], grid_map=GridMap(width=8, height=8))

    with pytest.raises(ValueError, match="Actor 5 not found"):
        encode_observation(state, actor_id=5)
