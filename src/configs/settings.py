"""Configuration classes."""

from dataclasses import dataclass


class CombatConfig:
    """Placeholder combat configuration."""

    pass


class TrainingConfig:
    """Placeholder training configuration."""

    pass


@dataclass
class PPOConfig:
    """Hyperparameters for PPO training."""

    rollout_steps: int = 128
    update_epochs: int = 4
    minibatch_size: int = 32
    learning_rate: float = 3.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    damage_reward_scale: float = 0.1
    terminal_reward: float = 1.0
    invalid_action_penalty: float = -0.25
    step_penalty: float = -0.001
    checkpoint_dir: str = "checkpoints"
