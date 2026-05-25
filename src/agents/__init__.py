"""Agent placeholders."""

from agents.action_space import (
    ACTION_CATEGORY_COUNT,
    MAIN_ACTION_TYPE_COUNT,
    MIN_OPTION_COUNT,
    ActionCategory,
    MainActionType,
    build_action_masks,
    decode_action,
)
from agents.base import BaseAgent
from agents.observation import (
    ACTOR_CLASS_FEATURE_SIZE,
    ACTOR_FEATURE_SIZE,
    ACTOR_DAMAGE_ACTION_FEATURE_SIZE,
    ACTOR_MAP_FEATURE_SIZE,
    BASE_CHARACTER_FEATURE_SIZE,
    CHARACTER_FEATURE_SIZE,
    DAMAGE_TYPE_FEATURE_SIZE,
    MAX_NEARBY_CHARACTERS,
    OBSERVATION_SIZE,
    OTHER_DAMAGE_PROFILE_FEATURE_SIZE,
    OTHER_CHARACTER_FEATURE_SIZE,
    OTHER_MAP_FEATURE_SIZE,
    encode_observation,
)
from agents.ppo_model import (
    DEFAULT_MOVE_COUNT,
    DEFAULT_OPTION_COUNT,
    DEFAULT_TARGET_COUNT,
    PPOActorCritic,
)
from agents.random_agent import RandomAgent

__all__ = [
    "ACTION_CATEGORY_COUNT",
    "ACTOR_CLASS_FEATURE_SIZE",
    "ACTOR_DAMAGE_ACTION_FEATURE_SIZE",
    "ACTOR_FEATURE_SIZE",
    "ACTOR_MAP_FEATURE_SIZE",
    "ActionCategory",
    "BaseAgent",
    "BASE_CHARACTER_FEATURE_SIZE",
    "CHARACTER_FEATURE_SIZE",
    "DAMAGE_TYPE_FEATURE_SIZE",
    "DEFAULT_MOVE_COUNT",
    "DEFAULT_OPTION_COUNT",
    "DEFAULT_TARGET_COUNT",
    "MAIN_ACTION_TYPE_COUNT",
    "MAX_NEARBY_CHARACTERS",
    "MIN_OPTION_COUNT",
    "MainActionType",
    "OBSERVATION_SIZE",
    "OTHER_CHARACTER_FEATURE_SIZE",
    "OTHER_DAMAGE_PROFILE_FEATURE_SIZE",
    "OTHER_MAP_FEATURE_SIZE",
    "PPOActorCritic",
    "RandomAgent",
    "build_action_masks",
    "decode_action",
    "encode_observation",
]
