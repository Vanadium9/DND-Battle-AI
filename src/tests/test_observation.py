import pytest
import torch

from agents import (
    ACTOR_COMMON_ACTION_OFFSET,
    ACTOR_FEATURE_SIZE,
    ACTOR_REAL_GAME_OFFSET,
    BASE_CHARACTER_FEATURE_SIZE,
    CHARACTER_FEATURE_SIZE,
    GLOBAL_FEATURE_SIZE,
    MAX_NEARBY_CHARACTERS,
    OBSERVATION_SIZE,
    OTHER_COMMON_ACTION_OFFSET,
    OTHER_CHARACTER_FEATURE_SIZE,
    OTHER_ENTITY_PROFILE_OFFSET,
    OTHER_MAP_FEATURE_OFFSET,
    encode_observation,
)
from agents.entity_observation import (
    FEAT_FLAG_NAMES,
    INVENTORY_ITEM_FLAG_NAMES,
    PREPARED_SPELL_FLAG_NAMES,
    TERRAIN_FEATURE_TYPES,
)
from combat import (
    CombatState,
    FighterArcher,
    FighterChampionGreatsword,
    FireElementalSimple,
    Goblin,
    GridMap,
    Orc,
    Position,
    TerrainType,
    WizardEvoker,
    PotionOfHealing,
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
ACTOR_FREE_OBJECT_INTERACTION = ACTOR_COMMON_ACTION_OFFSET
ACTOR_PRONE = ACTOR_COMMON_ACTION_OFFSET + 1
ACTOR_GRAPPLED = ACTOR_COMMON_ACTION_OFFSET + 2
ACTOR_HIDDEN = ACTOR_COMMON_ACTION_OFFSET + 3
ACTOR_DODGING = ACTOR_COMMON_ACTION_OFFSET + 4
ACTOR_DISENGAGED = ACTOR_COMMON_ACTION_OFFSET + 5
ACTOR_HAS_PREPARED_ACTION = ACTOR_COMMON_ACTION_OFFSET + 6
ACTOR_NUMBER_OF_WEAPONS = ACTOR_COMMON_ACTION_OFFSET + 7
ACTOR_HAS_SPELLS = ACTOR_COMMON_ACTION_OFFSET + 8
ACTOR_CAN_CAST_SPELL = ACTOR_COMMON_ACTION_OFFSET + 9
ACTOR_CAN_ATTACK = ACTOR_COMMON_ACTION_OFFSET + 10
ACTOR_CAN_DASH = ACTOR_COMMON_ACTION_OFFSET + 11
ACTOR_CAN_DISENGAGE = ACTOR_COMMON_ACTION_OFFSET + 12
ACTOR_CAN_DODGE = ACTOR_COMMON_ACTION_OFFSET + 13
ACTOR_CAN_HIDE = ACTOR_COMMON_ACTION_OFFSET + 14
ACTOR_CAN_HELP = ACTOR_COMMON_ACTION_OFFSET + 15
ACTOR_CAN_GRAPPLE = ACTOR_COMMON_ACTION_OFFSET + 16
ACTOR_CAN_SHOVE = ACTOR_COMMON_ACTION_OFFSET + 17
ACTOR_LEVEL_NORMALIZED = ACTOR_REAL_GAME_OFFSET
ACTOR_PROFICIENCY_NORMALIZED = ACTOR_REAL_GAME_OFFSET + 1
ACTOR_CLASS_ID = ACTOR_REAL_GAME_OFFSET + 2
ACTOR_SUBCLASS_ID = ACTOR_REAL_GAME_OFFSET + 3
ACTOR_RACE_ID = ACTOR_REAL_GAME_OFFSET + 4
ACTOR_FEAT_FLAGS = ACTOR_REAL_GAME_OFFSET + 5
ACTOR_ECONOMY_FLAGS = ACTOR_FEAT_FLAGS + len(FEAT_FLAG_NAMES)
ACTOR_CLASS_RESOURCE_FLAGS = ACTOR_ECONOMY_FLAGS + 4
ACTOR_SPELL_SLOT_FLAGS = ACTOR_CLASS_RESOURCE_FLAGS + 4
ACTOR_PREPARED_SPELL_FLAGS = ACTOR_SPELL_SLOT_FLAGS + 6
ACTOR_INVENTORY_FLAGS = ACTOR_PREPARED_SPELL_FLAGS + len(PREPARED_SPELL_FLAG_NAMES)
ACTOR_CURRENT_COVER_STATUS = ACTOR_INVENTORY_FLAGS + len(INVENTORY_ITEM_FLAG_NAMES)
ACTOR_TERRAIN_AROUND = ACTOR_CURRENT_COVER_STATUS + 1
ACTOR_VISIBLE_ENEMIES_COUNT = ACTOR_TERRAIN_AROUND + 4 * len(TERRAIN_FEATURE_TYPES)
OTHER_PRONE = OTHER_COMMON_ACTION_OFFSET
OTHER_GRAPPLED = OTHER_COMMON_ACTION_OFFSET + 1
OTHER_HIDDEN = OTHER_COMMON_ACTION_OFFSET + 2
OTHER_DODGING = OTHER_COMMON_ACTION_OFFSET + 3
OTHER_IN_MELEE_REACH = OTHER_COMMON_ACTION_OFFSET + 4
OTHER_CAN_BE_ATTACKED = OTHER_COMMON_ACTION_OFFSET + 5
OTHER_CAN_BE_HELPED_AGAINST = OTHER_COMMON_ACTION_OFFSET + 6
OTHER_CAN_BE_GRAPPLED = OTHER_COMMON_ACTION_OFFSET + 7
OTHER_CAN_BE_SHOVED = OTHER_COMMON_ACTION_OFFSET + 8
OTHER_PROFILE_CLASS_ID = OTHER_ENTITY_PROFILE_OFFSET
OTHER_PROFILE_SUBCLASS_ID = OTHER_ENTITY_PROFILE_OFFSET + 1
OTHER_PROFILE_ROLE_ID = OTHER_ENTITY_PROFILE_OFFSET + 2
OTHER_PROFILE_CR = OTHER_ENTITY_PROFILE_OFFSET + 3
OTHER_PROFILE_XP = OTHER_ENTITY_PROFILE_OFFSET + 4
OTHER_PROFILE_CONDITIONS = OTHER_ENTITY_PROFILE_OFFSET + 5
OTHER_PROFILE_ACTIVE_CONCENTRATION = OTHER_PROFILE_CONDITIONS + 8
OTHER_PROFILE_CURRENT_AC = OTHER_PROFILE_ACTIVE_CONCENTRATION + 1
OTHER_PROFILE_CURRENT_HP_RATIO = OTHER_PROFILE_CURRENT_AC + 1
OTHER_PROFILE_THREAT_ESTIMATE = OTHER_PROFILE_CURRENT_HP_RATIO + 1
OTHER_LINE_OF_SIGHT_FROM_ACTOR = OTHER_MAP_FEATURE_OFFSET
OTHER_COVER_FROM_ACTOR = OTHER_MAP_FEATURE_OFFSET + 1
OTHER_DISTANCE_TO_ACTOR = OTHER_MAP_FEATURE_OFFSET + 2
OTHER_REACHABLE_BY_ACTOR = OTHER_MAP_FEATURE_OFFSET + 3
GLOBAL_START = ACTOR_FEATURE_SIZE + OTHER_CHARACTER_FEATURE_SIZE * MAX_NEARBY_CHARACTERS * 2
GLOBAL_ROUND_NUMBER = GLOBAL_START
GLOBAL_INITIATIVE_POSITION = GLOBAL_START + 1
GLOBAL_ALLIES_ALIVE = GLOBAL_START + 2
GLOBAL_ENEMIES_ALIVE = GLOBAL_START + 3
GLOBAL_ENCOUNTER_DIFFICULTY = GLOBAL_START + 4
GLOBAL_MAP_WIDTH = GLOBAL_START + 5
GLOBAL_MAP_HEIGHT = GLOBAL_START + 6


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
        ACTOR_FEATURE_SIZE
        + OTHER_CHARACTER_FEATURE_SIZE * MAX_NEARBY_CHARACTERS * 2
        + GLOBAL_FEATURE_SIZE
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
    assert actor_features[ACTOR_LEVEL_NORMALIZED] == 1
    assert actor_features[ACTOR_PROFICIENCY_NORMALIZED] == pytest.approx(0.5)
    assert actor_features[ACTOR_CLASS_ID] == 1
    assert actor_features[ACTOR_SUBCLASS_ID] == 1
    assert actor_features[ACTOR_RACE_ID] == 0
    assert torch.count_nonzero(
        actor_features[ACTOR_FEAT_FLAGS : ACTOR_FEAT_FLAGS + len(FEAT_FLAG_NAMES)]
    ) == 0
    assert actor_features[ACTOR_ECONOMY_FLAGS] == 1
    assert actor_features[ACTOR_ECONOMY_FLAGS + 1] == 1
    assert actor_features[ACTOR_ECONOMY_FLAGS + 2] == 1
    assert actor_features[ACTOR_ECONOMY_FLAGS + 3] == actor.speed
    assert actor_features[ACTOR_CLASS_RESOURCE_FLAGS] == 1
    assert actor_features[ACTOR_CLASS_RESOURCE_FLAGS + 1] == 1
    assert actor_features[ACTOR_CLASS_RESOURCE_FLAGS + 2] == 0
    assert actor_features[ACTOR_CLASS_RESOURCE_FLAGS + 3] == 0
    assert actor_features[ACTOR_CURRENT_COVER_STATUS] >= 1
    assert actor_features[ACTOR_VISIBLE_ENEMIES_COUNT] == 2


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
    assert first_enemy_block[OTHER_PROFILE_ROLE_ID] > 0
    assert first_enemy_block[OTHER_PROFILE_CR] > 0
    assert first_enemy_block[OTHER_PROFILE_XP] > 0
    assert first_enemy_block[OTHER_PROFILE_CONDITIONS] == 1
    assert first_enemy_block[OTHER_PROFILE_CURRENT_AC] == goblin.ac
    assert first_enemy_block[OTHER_PROFILE_CURRENT_HP_RATIO] == 1
    assert first_enemy_block[OTHER_PROFILE_THREAT_ESTIMATE] > 0
    assert first_enemy_block[OTHER_LINE_OF_SIGHT_FROM_ACTOR] == 1
    assert first_enemy_block[OTHER_DISTANCE_TO_ACTOR] == 1
    assert first_enemy_block[OTHER_REACHABLE_BY_ACTOR] == 1
    assert second_enemy_block[PRESENT] == 1
    assert second_enemy_block[ALIVE] == 0
    assert second_enemy_block[DEAD] == 1
    assert second_enemy_block[ACTION_AVAILABLE] == 0
    assert second_enemy_block[OTHER_CAN_BE_ATTACKED] == 0
    assert second_enemy_block[OTHER_CAN_BE_HELPED_AGAINST] == 0
    assert second_enemy_block[OTHER_CAN_BE_GRAPPLED] == 0
    assert second_enemy_block[OTHER_CAN_BE_SHOVED] == 0


def test_encode_observation_real_game_actor_and_global_features() -> None:
    actor = WizardEvoker(Position(0, 0))
    actor.inventory = [PotionOfHealing(quantity=1)]
    enemy = FireElementalSimple(Position(3, 0))
    state = CombatState(
        characters=[actor, enemy],
        grid_map=GridMap(width=10, height=6),
        round_number=3,
        initiative_order=[1, 0],
        current_turn_index=1,
    )

    observation = encode_observation(state, actor_id=0)
    actor_features = actor_block(observation)

    assert actor_features[ACTOR_CLASS_ID] == 3
    assert actor_features[ACTOR_SUBCLASS_ID] == 3
    assert actor_features[ACTOR_CLASS_RESOURCE_FLAGS + 3] == 1
    assert actor_features[ACTOR_SPELL_SLOT_FLAGS : ACTOR_SPELL_SLOT_FLAGS + 6].tolist() == [
        4,
        4,
        3,
        3,
        2,
        2,
    ]

    fireball_index = PREPARED_SPELL_FLAG_NAMES.index("Fireball")
    potion_index = INVENTORY_ITEM_FLAG_NAMES.index("Potion of Healing")
    assert actor_features[ACTOR_PREPARED_SPELL_FLAGS + fireball_index] == 1
    assert actor_features[ACTOR_INVENTORY_FLAGS + potion_index] == 1
    assert observation[GLOBAL_ROUND_NUMBER] == 3
    assert observation[GLOBAL_INITIATIVE_POSITION] == 1
    assert observation[GLOBAL_ALLIES_ALIVE] == 1
    assert observation[GLOBAL_ENEMIES_ALIVE] == 1
    assert observation[GLOBAL_ENCOUNTER_DIFFICULTY] > 0
    assert observation[GLOBAL_MAP_WIDTH] == pytest.approx(0.5)
    assert observation[GLOBAL_MAP_HEIGHT] == pytest.approx(0.3)


def test_encode_observation_rejects_missing_actor() -> None:
    state = CombatState(characters=[FighterArcher()], grid_map=GridMap(width=8, height=8))

    with pytest.raises(ValueError, match="Actor 5 not found"):
        encode_observation(state, actor_id=5)
