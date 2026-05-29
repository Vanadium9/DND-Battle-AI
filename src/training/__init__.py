"""Training placeholders."""

from training.multi_agent import (
    CombatRole,
    MultiAgentPolicyRouter,
    RandomPolicy,
    RuleBasedEnemyPolicy,
    random_policy,
    role_embedding_for_actor,
    role_id_for_actor,
    rule_based_enemy_policy,
)
from training.ppo_trainer import (
    CurriculumConfig,
    EpisodeStats,
    PPOTrainer,
    RolloutBuffer,
    load_curriculum_config,
)
from training.self_play import (
    OpponentCheckpoint,
    OpponentPool,
    OpponentStats,
    SelfPlayConfig,
    SelfPlayManager,
    load_self_play_config,
)
from training.trainer import Trainer

__all__ = [
    "CombatRole",
    "CurriculumConfig",
    "EpisodeStats",
    "MultiAgentPolicyRouter",
    "OpponentCheckpoint",
    "OpponentPool",
    "OpponentStats",
    "PPOTrainer",
    "RandomPolicy",
    "RolloutBuffer",
    "RuleBasedEnemyPolicy",
    "SelfPlayConfig",
    "SelfPlayManager",
    "Trainer",
    "load_curriculum_config",
    "load_self_play_config",
    "random_policy",
    "role_embedding_for_actor",
    "role_id_for_actor",
    "rule_based_enemy_policy",
]
