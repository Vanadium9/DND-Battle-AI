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
from training.trainer import Trainer

__all__ = [
    "CombatRole",
    "CurriculumConfig",
    "EpisodeStats",
    "MultiAgentPolicyRouter",
    "PPOTrainer",
    "RandomPolicy",
    "RolloutBuffer",
    "RuleBasedEnemyPolicy",
    "Trainer",
    "load_curriculum_config",
    "random_policy",
    "role_embedding_for_actor",
    "role_id_for_actor",
    "rule_based_enemy_policy",
]
