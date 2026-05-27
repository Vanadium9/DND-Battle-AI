"""PPO trainer for the tactical combat environment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch.nn import functional as F

from agents import (
    GNNPPOActorCritic,
    PPOActorCritic,
    ActionCategory,
    MainActionType,
    build_action_masks,
    decode_action,
    encode_entity_observation,
    encode_observation,
)
from agents.entity_observation import EntityObservation
from configs import PPOConfig
from combat import CombatEnvironment, EndTurnAction, Team
from combat.encounter_generator import (
    EncounterGenerator,
    MAX_CURRICULUM_LEVEL,
    clamp_curriculum_level,
    get_curriculum_stage,
)
from training.multi_agent import (
    MultiAgentPolicyRouter,
    RandomPolicy,
    RuleBasedEnemyPolicy,
)


LOGGER = logging.getLogger(__name__)


@dataclass
class RolloutBuffer:
    """Collected PPO rollout data."""

    observations: list[Any] = field(default_factory=list)
    actor_observations: list[Any] = field(default_factory=list)
    critic_observations: list[Any] = field(default_factory=list)
    actions: dict[str, list[torch.Tensor]] = field(
        default_factory=lambda: {
            "action_category": [],
            "main_action_type": [],
            "target_index": [],
            "move_index": [],
            "option_index": [],
        }
    )
    log_probs: list[torch.Tensor] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    values: list[torch.Tensor] = field(default_factory=list)
    masks: dict[str, list[torch.Tensor]] = field(
        default_factory=lambda: {
            "action_category": [],
            "main_action_type": [],
            "target_index": [],
            "move_index": [],
            "option_index": [],
        }
    )
    last_value: float = 0.0

    def append(
        self,
        observation: Any,
        action: dict[str, torch.Tensor],
        reward: float,
        done: bool,
        masks: dict[str, torch.Tensor],
        critic_observation: Any | None = None,
    ) -> None:
        actor_observation = _detach_observation(observation)
        critic_observation = _detach_observation(
            observation if critic_observation is None else critic_observation
        )
        self.observations.append(actor_observation)
        self.actor_observations.append(actor_observation)
        self.critic_observations.append(critic_observation)
        self.log_probs.append(action["log_prob"].detach().cpu().reshape(()))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(action["value"].detach().cpu().reshape(()))

        for key, value in action.items():
            if key in {"log_prob", "entropy", "value"}:
                continue
            if key not in self.actions:
                self.actions[key] = []
            self.actions[key].append(value.detach().cpu().long().reshape(()))
        for key in self.actions:
            if len(self.actions[key]) < len(self.rewards):
                self.actions[key].append(torch.tensor(0, dtype=torch.long))
        for key, value in masks.items():
            if key not in self.masks:
                self.masks[key] = []
            self.masks[key].append(value.detach().cpu().bool())
        for key in self.masks:
            if len(self.masks[key]) < len(self.rewards):
                self.masks[key].append(torch.ones(1, dtype=torch.bool))

    def __len__(self) -> int:
        return len(self.rewards)

    def to_tensors(self, device: torch.device) -> dict[str, Any]:
        if not self.rewards:
            raise ValueError("rollout is empty")

        return {
            "observations": _stack_observations(self.observations, device),
            "actor_observations": _stack_observations(
                self.actor_observations,
                device,
            ),
            "critic_observations": _stack_observations(
                self.critic_observations,
                device,
            ),
            "actions": {
                key: torch.stack(values).to(device=device)
                for key, values in self.actions.items()
            },
            "log_probs": torch.stack(self.log_probs).to(device=device),
            "rewards": torch.tensor(self.rewards, dtype=torch.float32, device=device),
            "dones": torch.tensor(self.dones, dtype=torch.float32, device=device),
            "values": torch.stack(self.values).to(device=device),
            "masks": {
                key: torch.stack(values).to(device=device)
                for key, values in self.masks.items()
            },
            "last_value": torch.tensor(
                self.last_value,
                dtype=torch.float32,
                device=device,
            ),
        }


@dataclass(frozen=True)
class EpisodeStats:
    """Metrics collected during one environment episode."""

    total_reward: float
    length: int
    winner: Team | None
    action_counts: dict[str, int]


@dataclass(frozen=True)
class CurriculumConfig:
    """Runtime curriculum settings for PPO training."""

    enabled: bool = False
    initial_level: int = 1
    max_level: int = MAX_CURRICULUM_LEVEL
    win_rate_threshold: float = 0.75
    window_size: int = 20


def load_curriculum_config(path: str | Path) -> CurriculumConfig:
    """Load curriculum training settings from a YAML file."""

    import yaml

    with Path(path).open("r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}
    return CurriculumConfig(
        enabled=bool(data.get("enabled", False)),
        initial_level=int(data.get("initial_level", 1)),
        max_level=int(data.get("max_level", MAX_CURRICULUM_LEVEL)),
        win_rate_threshold=float(data.get("win_rate_threshold", 0.75)),
        window_size=int(data.get("window_size", 20)),
    )


class PPOTrainer:
    """Collect rollouts and optimize a PPO actor-critic model."""

    def __init__(
        self,
        environment: CombatEnvironment,
        model: PPOActorCritic | GNNPPOActorCritic | None = None,
        config: PPOConfig | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        device: torch.device | str | None = None,
        encounter_generator: EncounterGenerator | None = None,
        curriculum_config: CurriculumConfig | None = None,
        curriculum_enabled: bool | None = None,
        curriculum_win_rate_threshold: float | None = None,
        curriculum_window_size: int | None = None,
        curriculum_level: int | None = None,
        shared_policy: Any | None = None,
        player_policy: Any | None = None,
        enemy_policy: Any | None = None,
        role_policies: Mapping[object, Any] | None = None,
        policy_router: MultiAgentPolicyRouter | None = None,
        rule_based_enemy_policy: bool | Any = False,
        random_policy: bool | Any = False,
    ) -> None:
        self.environment = environment
        self.config = config or PPOConfig()
        self.device = torch.device(device or "cpu")
        self.model = model or (
            GNNPPOActorCritic()
            if self.config.model_type == "gnn"
            else PPOActorCritic()
        )
        self.model.to(self.device)
        self.policy_router = self._resolve_policy_router(
            policy_router=policy_router,
            shared_policy=shared_policy,
            player_policy=player_policy,
            enemy_policy=enemy_policy,
            role_policies=role_policies,
            rule_based_enemy_policy=rule_based_enemy_policy,
            random_policy=random_policy,
        )
        self._move_policy_modules_to_device()
        self.optimizer = optimizer or torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )
        self.encounter_generator = encounter_generator
        self.curriculum_config = self._resolve_curriculum_config(
            curriculum_config,
            curriculum_enabled=curriculum_enabled,
            curriculum_win_rate_threshold=curriculum_win_rate_threshold,
            curriculum_window_size=curriculum_window_size,
            curriculum_level=curriculum_level,
        )
        self.current_curriculum_level = (
            clamp_curriculum_level(self.curriculum_config.initial_level)
            if self.curriculum_config.enabled
            else None
        )
        if self.current_curriculum_level is not None:
            self.current_curriculum_level = min(
                self.current_curriculum_level,
                self.curriculum_config.max_level,
            )
        self.curriculum_recent_wins: list[bool] = []
        self.curriculum_transition_log: list[str] = []
        if self.curriculum_config.enabled and self.encounter_generator is not None:
            self.encounter_generator.set_curriculum_level(self.current_curriculum_level)

    def collect_episode(
        self,
        max_steps: int | None = None,
        reset: bool = True,
    ) -> tuple[RolloutBuffer, EpisodeStats]:
        """Collect one episode or a max-step truncated episode."""

        steps = max_steps or self.config.rollout_steps
        if steps <= 0:
            raise ValueError("max_steps must be greater than zero")
        if reset:
            self._reset_environment_for_episode()

        rollout = RolloutBuffer()
        action_counts = _empty_action_counts()
        self._set_policies_eval()

        for _ in range(steps):
            if self.environment.is_done():
                break

            state = self.environment.combat_state
            actor_id = self._active_actor_id()
            policy = self._policy_for_actor(state, actor_id)
            actor_observation = self._actor_observation(state, actor_id, policy)
            critic_observation = self._critic_observation(state, actor_id, policy)
            masks = self._padded_masks(build_action_masks(state, actor_id), policy)

            with torch.no_grad():
                action_output = self._policy_act(
                    policy,
                    actor_observation,
                    masks,
                    critic_observation,
                    state,
                    actor_id,
                )

            action_counts[_action_count_name(action_output)] += 1
            action = self._decode_model_action(action_output, state, actor_id)
            result = self.environment.step(action)
            done = self.environment.is_done()
            rollout.append(
                actor_observation,
                action_output,
                result.reward,
                done,
                masks,
                critic_observation=critic_observation,
            )

            if done:
                break

        rollout.last_value = 0.0 if self.environment.is_done() else self._bootstrap_value()
        stats = EpisodeStats(
            total_reward=sum(rollout.rewards),
            length=len(rollout),
            winner=self.environment.get_winner(),
            action_counts=action_counts,
        )
        self.record_curriculum_result(stats.winner)
        return rollout, stats

    def collect_rollout(self, rollout_steps: int | None = None) -> RolloutBuffer:
        """Collect one fixed-length rollout from the environment."""

        steps = rollout_steps or self.config.rollout_steps
        if steps <= 0:
            raise ValueError("rollout_steps must be greater than zero")

        rollout = RolloutBuffer()
        self._set_policies_eval()

        for _ in range(steps):
            if self.environment.is_done():
                self._reset_environment_for_episode()

            state = self.environment.combat_state
            actor_id = self._active_actor_id()
            actor = state.character_at(actor_id)
            if actor is None:
                raise ValueError("environment has no active actor")

            policy = self._policy_for_actor(state, actor_id)
            actor_observation = self._actor_observation(state, actor_id, policy)
            critic_observation = self._critic_observation(state, actor_id, policy)
            masks = self._padded_masks(build_action_masks(state, actor_id), policy)

            with torch.no_grad():
                action_output = self._policy_act(
                    policy,
                    actor_observation,
                    masks,
                    critic_observation,
                    state,
                    actor_id,
                )

            action = self._decode_model_action(action_output, state, actor_id)
            result = self.environment.step(action)
            done = self.environment.is_done()

            rollout.append(
                actor_observation,
                action_output,
                result.reward,
                done,
                masks,
                critic_observation=critic_observation,
            )
            if done:
                self.record_curriculum_result(self.environment.get_winner())
                self._reset_environment_for_episode()

        rollout.last_value = self._bootstrap_value()
        return rollout

    def compute_returns_and_advantages(
        self,
        rollout: RolloutBuffer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and discounted returns."""

        if len(rollout) == 0:
            raise ValueError("rollout is empty")

        rewards = torch.tensor(rollout.rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(rollout.dones, dtype=torch.float32, device=self.device)
        values = torch.stack(rollout.values).to(device=self.device)
        last_value = torch.tensor(
            rollout.last_value,
            dtype=torch.float32,
            device=self.device,
        )
        advantages = torch.zeros_like(rewards)
        gae = torch.tensor(0.0, device=self.device)

        for step in reversed(range(len(rollout))):
            next_value = last_value if step == len(rollout) - 1 else values[step + 1]
            next_non_terminal = 1.0 - dones[step]
            delta = (
                rewards[step]
                + self.config.gamma * next_value * next_non_terminal
                - values[step]
            )
            gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * gae
            advantages[step] = gae

        returns = advantages + values
        return returns.detach(), advantages.detach()

    def update(self, rollout: RolloutBuffer) -> dict[str, float]:
        """Run PPO optimization over a collected rollout."""

        if len(rollout) == 0:
            raise ValueError("rollout is empty")

        self.model.train()
        data = rollout.to_tensors(self.device)
        returns, advantages = self.compute_returns_and_advantages(rollout)
        advantages = _normalize_advantages(advantages)

        batch_size = len(rollout)
        minibatch_size = min(self.config.minibatch_size, batch_size)
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "loss": 0.0,
        }
        updates = 0

        for _ in range(self.config.update_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                batch_indices = indices[start : start + minibatch_size]
                batch_actions = {
                    key: value[batch_indices]
                    for key, value in data["actions"].items()
                }
                batch_masks = {
                    key: value[batch_indices]
                    for key, value in data["masks"].items()
                }

                evaluation = self._model_evaluate_actions(
                    _select_observation_batch(data["actor_observations"], batch_indices),
                    batch_actions,
                    batch_masks,
                    _select_observation_batch(
                        data["critic_observations"],
                        batch_indices,
                    ),
                )
                old_log_probs = data["log_probs"][batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                ratio = torch.exp(evaluation["log_prob"] - old_log_probs)
                unclipped = ratio * batch_advantages
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon,
                ) * batch_advantages
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(evaluation["value"], batch_returns)
                entropy = evaluation["entropy"].mean()
                loss = (
                    policy_loss
                    + self.config.value_loss_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                metrics["policy_loss"] += float(policy_loss.detach().cpu())
                metrics["value_loss"] += float(value_loss.detach().cpu())
                metrics["entropy"] += float(entropy.detach().cpu())
                metrics["loss"] += float(loss.detach().cpu())
                updates += 1

        for key in metrics:
            metrics[key] /= max(1, updates)
        return metrics

    def train_iteration(
        self,
        save_checkpoint: bool = True,
        checkpoint_name: str = "ppo_actor_critic.pt",
    ) -> dict[str, float | str]:
        """Collect rollout, update model, and optionally save a checkpoint."""

        rollout = self.collect_rollout()
        metrics: dict[str, float | str] = self.update(rollout)
        if save_checkpoint:
            metrics["checkpoint_path"] = str(self.save_checkpoint(checkpoint_name))
        return metrics

    def save_checkpoint(self, filename: str = "ppo_actor_critic.pt") -> Path:
        """Save model and optimizer state into the configured checkpoint directory."""

        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "curriculum_level": self.current_curriculum_level,
                "curriculum_state": self.curriculum_state_dict(),
            },
            checkpoint_path,
        )
        return checkpoint_path

    @property
    def curriculum_level(self) -> int | None:
        """Return the active curriculum level, if curriculum is enabled."""

        return self.current_curriculum_level

    @property
    def current_curriculum_win_rate(self) -> float:
        """Return win rate over the current curriculum window."""

        if not self.curriculum_recent_wins:
            return 0.0
        return sum(1 for won in self.curriculum_recent_wins if won) / len(
            self.curriculum_recent_wins
        )

    def record_curriculum_result(self, winner: Team | None) -> None:
        """Track an episode result and promote curriculum difficulty when ready."""

        if not self.curriculum_config.enabled or self.current_curriculum_level is None:
            return

        self.curriculum_recent_wins.append(winner is Team.PLAYERS)
        window_size = max(1, self.curriculum_config.window_size)
        if len(self.curriculum_recent_wins) > window_size:
            self.curriculum_recent_wins = self.curriculum_recent_wins[-window_size:]
        if len(self.curriculum_recent_wins) < window_size:
            return
        if self.current_curriculum_level >= self.curriculum_config.max_level:
            return
        if self.current_curriculum_win_rate < self.curriculum_config.win_rate_threshold:
            return

        previous_level = self.current_curriculum_level
        self.current_curriculum_level = min(
            self.curriculum_config.max_level,
            previous_level + 1,
        )
        self.curriculum_recent_wins = []
        if self.encounter_generator is not None:
            self.encounter_generator.set_curriculum_level(self.current_curriculum_level)
        previous_stage = get_curriculum_stage(previous_level)
        next_stage = get_curriculum_stage(self.current_curriculum_level)
        message = (
            f"Curriculum advanced from level {previous_level} "
            f"({previous_stage.name}) to level {self.current_curriculum_level} "
            f"({next_stage.name})."
        )
        self.curriculum_transition_log.append(message)
        LOGGER.info(message)

    def curriculum_state_dict(self) -> dict[str, object]:
        """Return serializable curriculum training state."""

        stage_name = (
            get_curriculum_stage(self.current_curriculum_level).name
            if self.current_curriculum_level is not None
            else None
        )
        return {
            "enabled": self.curriculum_config.enabled,
            "current_level": self.current_curriculum_level,
            "current_stage": stage_name,
            "max_level": self.curriculum_config.max_level,
            "win_rate_threshold": self.curriculum_config.win_rate_threshold,
            "window_size": self.curriculum_config.window_size,
            "recent_wins": list(self.curriculum_recent_wins),
            "current_win_rate": self.current_curriculum_win_rate,
            "transition_log": list(self.curriculum_transition_log),
        }

    def _bootstrap_value(self) -> float:
        if self.environment.is_done():
            return 0.0

        state = self.environment.combat_state
        actor_id = self._active_actor_id()
        with torch.no_grad():
            policy = self._policy_for_actor(state, actor_id)
            actor_observation = self._actor_observation(state, actor_id, policy)
            critic_observation = self._critic_observation(state, actor_id, policy)
            value = self._policy_value(policy, actor_observation, critic_observation).squeeze(0)
        return float(value.detach().cpu())

    def _active_actor_id(self) -> int:
        state = self.environment.combat_state
        actor_id = state.active_actor_id
        if actor_id is None:
            raise ValueError("environment has no characters")
        return actor_id

    def policy_for_actor(self, actor_id: int) -> Any:
        """Return the policy configured for an actor in the current environment."""

        return self._policy_for_actor(self.environment.combat_state, actor_id)

    def _policy_for_actor(self, state: Any, actor_id: int) -> Any:
        actor = state.character_at(actor_id)
        if actor is None:
            raise ValueError(f"Actor {actor_id} not found")
        return self.policy_router.policy_for(actor)

    def _actor_observation(self, state: Any, actor_id: int, policy: Any | None = None) -> Any:
        if self._uses_entity_observation(policy):
            return encode_entity_observation(state, actor_id)
        return encode_observation(state, actor_id).to(self.device)

    def _critic_observation(self, state: Any, actor_id: int, policy: Any | None = None) -> Any:
        if self._uses_entity_observation(policy):
            return encode_entity_observation(state, actor_id)
        return encode_observation(state, actor_id).to(self.device)

    def _uses_entity_observation(self, policy: Any | None = None) -> bool:
        selected_policy = self.model if policy is None else policy
        if isinstance(selected_policy, GNNPPOActorCritic):
            return True
        if isinstance(selected_policy, PPOActorCritic):
            return False
        return self.config.model_type == "gnn" or isinstance(self.model, GNNPPOActorCritic)

    def _policy_act(
        self,
        policy: Any,
        actor_observation: Any,
        masks: dict[str, torch.Tensor],
        critic_observation: Any,
        state: Any,
        actor_id: int,
    ) -> dict[str, torch.Tensor]:
        if isinstance(policy, GNNPPOActorCritic):
            return policy.act(
                actor_observation,
                masks,
                critic_observation=(
                    critic_observation if self.config.centralized_critic else None
                ),
            )
        if isinstance(policy, PPOActorCritic):
            return policy.act(actor_observation, masks)
        actor = state.character_at(actor_id)
        return policy.act(
            actor_observation,
            masks,
            state=state,
            actor_id=actor_id,
            actor=actor,
            critic_observation=critic_observation,
        )

    def _model_evaluate_actions(
        self,
        actor_observation: Any,
        actions: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
        critic_observation: Any,
    ) -> dict[str, torch.Tensor]:
        if isinstance(self.model, GNNPPOActorCritic):
            return self.model.evaluate_actions(
                actor_observation,
                actions,
                masks,
                critic_observation=(
                    critic_observation if self.config.centralized_critic else None
                ),
            )
        return self.model.evaluate_actions(actor_observation, actions, masks)

    def _model_value(
        self,
        actor_observation: Any,
        critic_observation: Any,
    ) -> torch.Tensor:
        if isinstance(self.model, GNNPPOActorCritic):
            return self.model(
                actor_observation,
                critic_observation=(
                    critic_observation if self.config.centralized_critic else None
                ),
            )["value"]
        return self.model(actor_observation)["value"]

    def _policy_value(
        self,
        policy: Any,
        actor_observation: Any,
        critic_observation: Any,
    ) -> torch.Tensor:
        if isinstance(policy, GNNPPOActorCritic):
            return policy(
                actor_observation,
                critic_observation=(
                    critic_observation if self.config.centralized_critic else None
                ),
            )["value"]
        if isinstance(policy, PPOActorCritic):
            return policy(actor_observation)["value"]
        return torch.zeros(1, dtype=torch.float32, device=self.device)

    def _set_policies_eval(self) -> None:
        for policy in self.policy_router.policies():
            if isinstance(policy, torch.nn.Module):
                policy.eval()

    def _move_policy_modules_to_device(self) -> None:
        for policy in self.policy_router.policies():
            if isinstance(policy, torch.nn.Module):
                policy.to(self.device)

    def _resolve_policy_router(
        self,
        *,
        policy_router: MultiAgentPolicyRouter | None,
        shared_policy: Any | None,
        player_policy: Any | None,
        enemy_policy: Any | None,
        role_policies: Mapping[object, Any] | None,
        rule_based_enemy_policy: bool | Any,
        random_policy: bool | Any,
    ) -> MultiAgentPolicyRouter:
        if policy_router is not None:
            if policy_router.shared_policy is None:
                policy_router.shared_policy = shared_policy or self.model
            return policy_router
        return MultiAgentPolicyRouter(
            shared_policy=shared_policy or self.model,
            player_policy=player_policy,
            enemy_policy=enemy_policy,
            role_policies=role_policies,
            rule_based_enemy_policy=_coerce_baseline_policy(
                rule_based_enemy_policy,
                RuleBasedEnemyPolicy,
            ),
            random_policy=_coerce_baseline_policy(random_policy, RandomPolicy),
        )

    def _decode_model_action(
        self,
        action_output: dict[str, torch.Tensor],
        state: Any,
        actor_id: int,
    ) -> Any:
        try:
            return decode_action(
                int(action_output["action_category"].item()),
                int(action_output["main_action_type"].item()),
                int(action_output["target_index"].item()),
                int(action_output["move_index"].item()),
                int(action_output["option_index"].item()),
                state,
                actor_id,
            )
        except ValueError:
            return EndTurnAction(actor_id=actor_id)

    def _reset_environment_for_episode(self) -> None:
        if self.curriculum_config.enabled and self.encounter_generator is not None:
            self.encounter_generator.set_curriculum_level(self.current_curriculum_level)
            self.environment = self.encounter_generator.generate_curriculum_environment(
                self.current_curriculum_level,
                use_initiative=getattr(self.environment, "use_initiative", True),
                log_to_console=getattr(self.environment, "log_to_console", True),
            )
            return
        self.environment.reset()

    def _resolve_curriculum_config(
        self,
        curriculum_config: CurriculumConfig | None,
        *,
        curriculum_enabled: bool | None,
        curriculum_win_rate_threshold: float | None,
        curriculum_window_size: int | None,
        curriculum_level: int | None,
    ) -> CurriculumConfig:
        base = curriculum_config or CurriculumConfig()
        generator_level = (
            self.encounter_generator.curriculum_level
            if self.encounter_generator is not None
            else None
        )
        enabled = (
            bool(curriculum_enabled)
            if curriculum_enabled is not None
            else base.enabled or generator_level is not None
        )
        initial_level = curriculum_level or generator_level or base.initial_level
        max_level = max(1, min(MAX_CURRICULUM_LEVEL, int(base.max_level)))
        return CurriculumConfig(
            enabled=enabled,
            initial_level=min(clamp_curriculum_level(initial_level), max_level),
            max_level=max_level,
            win_rate_threshold=(
                float(curriculum_win_rate_threshold)
                if curriculum_win_rate_threshold is not None
                else float(base.win_rate_threshold)
            ),
            window_size=(
                max(1, int(curriculum_window_size))
                if curriculum_window_size is not None
                else max(1, int(base.window_size))
            ),
        )

    def _padded_masks(
        self,
        masks: dict[str, torch.Tensor],
        policy: Any | None = None,
    ) -> dict[str, torch.Tensor]:
        action_space = _policy_action_space(policy or self.model, self.model)
        return {
            "action_category": _pad_mask(
                masks["action_category"],
                action_space.action_category_count,
                "action_category",
                self.device,
            ),
            "main_action_type": _pad_mask(
                masks["main_action_type"],
                action_space.main_action_type_count,
                "main_action_type",
                self.device,
            ),
            "target_index": _pad_mask(
                masks["target_index"],
                action_space.target_count,
                "target_index",
                self.device,
            ),
            "move_index": _pad_mask(
                masks["move_index"],
                action_space.move_count,
                "move_index",
                self.device,
            ),
            "option_index": _pad_mask(
                masks["option_index"],
                action_space.option_count,
                "option_index",
                self.device,
            ),
        }


def _pad_mask(
    mask: torch.Tensor,
    target_size: int,
    name: str,
    device: torch.device,
) -> torch.Tensor:
    prepared = mask.to(device=device, dtype=torch.bool)
    if prepared.ndim != 1:
        raise ValueError(f"{name} mask must be a 1D tensor")
    if prepared.shape[0] > target_size:
        raise ValueError(f"{name} mask is larger than the model head")
    if prepared.shape[0] == target_size:
        return prepared

    padding = torch.zeros(
        target_size - prepared.shape[0],
        dtype=torch.bool,
        device=device,
    )
    return torch.cat((prepared, padding), dim=0)


def _coerce_baseline_policy(value: bool | Any, policy_type: type) -> Any | None:
    if value is True:
        return policy_type()
    if value is False or value is None:
        return None
    return value


def _policy_action_space(policy: Any, fallback: Any) -> SimpleNamespace:
    source = policy
    required = (
        "action_category_count",
        "main_action_type_count",
        "target_count",
        "move_count",
        "option_count",
    )
    if not all(hasattr(source, name) for name in required):
        source = fallback
    return SimpleNamespace(
        action_category_count=int(getattr(source, "action_category_count")),
        main_action_type_count=int(getattr(source, "main_action_type_count")),
        target_count=int(getattr(source, "target_count")),
        move_count=int(getattr(source, "move_count")),
        option_count=int(getattr(source, "option_count")),
    )


def _detach_observation(observation: Any) -> Any:
    if isinstance(observation, EntityObservation):
        return EntityObservation(
            actor_features=observation.actor_features.detach().cpu(),
            entities_features=observation.entities_features.detach().cpu(),
            map_features=observation.map_features.detach().cpu(),
            global_features=observation.global_features.detach().cpu(),
            entity_mask=observation.entity_mask.detach().cpu(),
        )
    if isinstance(observation, Mapping):
        return {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in observation.items()
        }
    if torch.is_tensor(observation):
        return observation.detach().cpu()
    raise TypeError(f"unsupported observation type: {type(observation)!r}")


def _stack_observations(
    observations: list[Any],
    device: torch.device,
) -> torch.Tensor | dict[str, torch.Tensor]:
    if not observations:
        raise ValueError("observations are empty")

    first = observations[0]
    if isinstance(first, EntityObservation):
        return {
            "actor_features": torch.stack(
                [observation.actor_features for observation in observations]
            ).to(device=device),
            "entities_features": torch.stack(
                [observation.entities_features for observation in observations]
            ).to(device=device),
            "map_features": torch.stack(
                [observation.map_features for observation in observations]
            ).to(device=device),
            "global_features": torch.stack(
                [observation.global_features for observation in observations]
            ).to(device=device),
            "entity_mask": torch.stack(
                [observation.entity_mask for observation in observations]
            ).to(device=device),
        }
    if isinstance(first, Mapping):
        return {
            key: torch.stack([observation[key] for observation in observations]).to(
                device=device,
            )
            for key in first
        }
    if torch.is_tensor(first):
        return torch.stack(observations).to(device=device)
    raise TypeError(f"unsupported observation type: {type(first)!r}")


def _select_observation_batch(
    observation: torch.Tensor | Mapping[str, torch.Tensor],
    indices: torch.Tensor,
) -> torch.Tensor | dict[str, torch.Tensor]:
    if isinstance(observation, Mapping):
        return {key: value[indices] for key, value in observation.items()}
    return observation[indices]


def _normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    if advantages.numel() <= 1:
        return advantages
    return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1.0e-8)


def _empty_action_counts() -> dict[str, int]:
    names = [category.name for category in ActionCategory if category is not ActionCategory.MAIN_ACTION]
    names.extend(main_action.name for main_action in MainActionType)
    return {name: 0 for name in names}


def _action_count_name(action_output: dict[str, torch.Tensor]) -> str:
    category = ActionCategory(int(action_output["action_category"].item()))
    if category is ActionCategory.MAIN_ACTION:
        return MainActionType(int(action_output["main_action_type"].item())).name
    return category.name
