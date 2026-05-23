"""PPO trainer for the tactical combat environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from agents import (
    PPOActorCritic,
    ActionCategory,
    MainActionType,
    build_action_masks,
    decode_action,
    encode_observation,
)
from configs import PPOConfig
from combat import CombatEnvironment, Team


@dataclass
class RolloutBuffer:
    """Collected PPO rollout data."""

    observations: list[torch.Tensor] = field(default_factory=list)
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
        observation: torch.Tensor,
        action: dict[str, torch.Tensor],
        reward: float,
        done: bool,
        masks: dict[str, torch.Tensor],
    ) -> None:
        self.observations.append(observation.detach().cpu())
        self.log_probs.append(action["log_prob"].detach().cpu().reshape(()))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(action["value"].detach().cpu().reshape(()))

        for key in self.actions:
            self.actions[key].append(action[key].detach().cpu().long().reshape(()))
            self.masks[key].append(masks[key].detach().cpu().bool())

    def __len__(self) -> int:
        return len(self.rewards)

    def to_tensors(self, device: torch.device) -> dict[str, Any]:
        if not self.rewards:
            raise ValueError("rollout is empty")

        return {
            "observations": torch.stack(self.observations).to(device=device),
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


class PPOTrainer:
    """Collect rollouts and optimize a PPO actor-critic model."""

    def __init__(
        self,
        environment: CombatEnvironment,
        model: PPOActorCritic | None = None,
        config: PPOConfig | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.environment = environment
        self.config = config or PPOConfig()
        self.device = torch.device(device or "cpu")
        self.model = model or PPOActorCritic()
        self.model.to(self.device)
        self.optimizer = optimizer or torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

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
            self.environment.reset()

        rollout = RolloutBuffer()
        action_counts = _empty_action_counts()
        self.model.eval()

        for _ in range(steps):
            if self.environment.is_done():
                break

            state = self.environment.combat_state
            actor_id = self._active_actor_id()
            observation = encode_observation(state, actor_id).to(self.device)
            masks = self._padded_masks(build_action_masks(state, actor_id))

            with torch.no_grad():
                action_output = self.model.act(observation, masks)

            action_counts[_action_count_name(action_output)] += 1
            action = decode_action(
                int(action_output["action_category"].item()),
                int(action_output["main_action_type"].item()),
                int(action_output["target_index"].item()),
                int(action_output["move_index"].item()),
                int(action_output["option_index"].item()),
                state,
                actor_id,
            )
            result = self.environment.step(action)
            done = self.environment.is_done()
            rollout.append(observation, action_output, result.reward, done, masks)

            if done:
                break

        rollout.last_value = 0.0 if self.environment.is_done() else self._bootstrap_value()
        stats = EpisodeStats(
            total_reward=sum(rollout.rewards),
            length=len(rollout),
            winner=self.environment.get_winner(),
            action_counts=action_counts,
        )
        return rollout, stats

    def collect_rollout(self, rollout_steps: int | None = None) -> RolloutBuffer:
        """Collect one fixed-length rollout from the environment."""

        steps = rollout_steps or self.config.rollout_steps
        if steps <= 0:
            raise ValueError("rollout_steps must be greater than zero")

        rollout = RolloutBuffer()
        self.model.eval()

        for _ in range(steps):
            if self.environment.is_done():
                self.environment.reset()

            state = self.environment.combat_state
            actor_id = self._active_actor_id()
            actor = state.character_at(actor_id)
            if actor is None:
                raise ValueError("environment has no active actor")

            observation = encode_observation(state, actor_id).to(self.device)
            masks = self._padded_masks(build_action_masks(state, actor_id))

            with torch.no_grad():
                action_output = self.model.act(observation, masks)

            action = decode_action(
                int(action_output["action_category"].item()),
                int(action_output["main_action_type"].item()),
                int(action_output["target_index"].item()),
                int(action_output["move_index"].item()),
                int(action_output["option_index"].item()),
                state,
                actor_id,
            )
            result = self.environment.step(action)
            done = self.environment.is_done()

            rollout.append(observation, action_output, result.reward, done, masks)
            if done:
                self.environment.reset()

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

                evaluation = self.model.evaluate_actions(
                    data["observations"][batch_indices],
                    batch_actions,
                    batch_masks,
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
            },
            checkpoint_path,
        )
        return checkpoint_path

    def _bootstrap_value(self) -> float:
        if self.environment.is_done():
            return 0.0

        state = self.environment.combat_state
        actor_id = self._active_actor_id()
        with torch.no_grad():
            observation = encode_observation(state, actor_id).to(self.device)
            value = self.model(observation)["value"].squeeze(0)
        return float(value.detach().cpu())

    def _active_actor_id(self) -> int:
        state = self.environment.combat_state
        actor_id = state.active_actor_id
        if actor_id is None:
            raise ValueError("environment has no characters")
        return actor_id

    def _padded_masks(self, masks: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            "action_category": _pad_mask(
                masks["action_category"],
                self.model.action_category_count,
                "action_category",
                self.device,
            ),
            "main_action_type": _pad_mask(
                masks["main_action_type"],
                self.model.main_action_type_count,
                "main_action_type",
                self.device,
            ),
            "target_index": _pad_mask(
                masks["target_index"],
                self.model.target_count,
                "target_index",
                self.device,
            ),
            "move_index": _pad_mask(
                masks["move_index"],
                self.model.move_count,
                "move_index",
                self.device,
            ),
            "option_index": _pad_mask(
                masks["option_index"],
                self.model.option_count,
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
