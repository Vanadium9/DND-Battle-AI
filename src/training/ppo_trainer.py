"""PPO trainer for the tactical combat environment."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
import logging
from pathlib import Path
import time
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
    build_fast_training_action_masks,
    decode_action,
    decode_fast_training_action,
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
from training.self_play import SelfPlayConfig, SelfPlayManager


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
    next_values: list[torch.Tensor] = field(default_factory=list)
    actor_ids: list[int] = field(default_factory=list)
    team_ids: list[int] = field(default_factory=list)
    class_ids: list[int] = field(default_factory=list)
    trainable_flags: list[bool] = field(default_factory=list)
    env_ids: list[int] = field(default_factory=list)
    episode_winners: list[Team | None] = field(default_factory=list)
    episode_timeouts: int = 0
    profile_times: dict[str, float] = field(default_factory=dict)
    last_value: float = 0.0
    last_values_by_env: dict[int, float] = field(default_factory=dict)

    def append(
        self,
        observation: Any,
        action: dict[str, torch.Tensor],
        reward: float,
        done: bool,
        masks: dict[str, torch.Tensor],
        critic_observation: Any | None = None,
        actor_id: int | None = None,
        team: Team | None = None,
        actor_class: str | None = None,
        trainable_transition: bool = True,
        env_id: int = 0,
        next_value: torch.Tensor | float | None = None,
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
        if next_value is not None:
            self.next_values.append(torch.as_tensor(next_value).detach().cpu().reshape(()))
        self.actor_ids.append(-1 if actor_id is None else int(actor_id))
        self.team_ids.append(_team_id(team))
        self.class_ids.append(_class_id(actor_class))
        self.trainable_flags.append(bool(trainable_transition))
        self.env_ids.append(int(env_id))

        sanitized_action = _sanitize_action_for_masks(action, masks)
        for key, value in action.items():
            if key in {"log_prob", "entropy", "value"}:
                continue
            if key not in self.actions:
                self.actions[key] = []
            selected_value = sanitized_action.get(key, value)
            self.actions[key].append(selected_value.detach().cpu().long().reshape(()))
        for key in self.actions:
            if len(self.actions[key]) < len(self.rewards):
                self.actions[key].append(torch.tensor(0, dtype=torch.long))
        for key, value in masks.items():
            if key not in self.masks:
                self.masks[key] = []
            self.masks[key].append(value.detach().cpu().bool().clone())
        for key in self.masks:
            if len(self.masks[key]) < len(self.rewards):
                self.masks[key].append(torch.ones(1, dtype=torch.bool))

    def __len__(self) -> int:
        return len(self.rewards)

    def to_tensors(self, device: torch.device) -> dict[str, Any]:
        if not self.rewards:
            raise ValueError("rollout is empty")
        actor_ids = (
            self.actor_ids
            if len(self.actor_ids) == len(self.rewards)
            else [-1] * len(self.rewards)
        )
        team_ids = (
            self.team_ids
            if len(self.team_ids) == len(self.rewards)
            else [-1] * len(self.rewards)
        )
        env_ids = (
            self.env_ids
            if len(self.env_ids) == len(self.rewards)
            else [0] * len(self.rewards)
        )
        class_ids = (
            self.class_ids
            if len(self.class_ids) == len(self.rewards)
            else [0] * len(self.rewards)
        )
        trainable_flags = (
            self.trainable_flags
            if len(self.trainable_flags) == len(self.rewards)
            else [True] * len(self.rewards)
        )

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
            "actor_ids": torch.tensor(
                actor_ids,
                dtype=torch.long,
                device=device,
            ),
            "team_ids": torch.tensor(
                team_ids,
                dtype=torch.long,
                device=device,
            ),
            "class_ids": torch.tensor(
                class_ids,
                dtype=torch.long,
                device=device,
            ),
            "trainable_flags": torch.tensor(
                trainable_flags,
                dtype=torch.bool,
                device=device,
            ),
            "env_ids": torch.tensor(
                env_ids,
                dtype=torch.long,
                device=device,
            ),
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
    min_updates_per_level: int = 0
    kind: str = "general"
    rehearsal_probability: float = 0.0
    terminal_defeat_penalty: float = 0.0
    timeout_penalty: float = 0.0


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
        min_updates_per_level=max(0, int(data.get("min_updates_per_level", 0))),
        kind=str(data.get("kind", "general")).strip().lower(),
        rehearsal_probability=max(
            0.0,
            min(1.0, float(data.get("rehearsal_probability", 0.0))),
        ),
        terminal_defeat_penalty=max(
            0.0,
            float(data.get("terminal_defeat_penalty", 0.0)),
        ),
        timeout_penalty=max(0.0, float(data.get("timeout_penalty", 0.0))),
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
        ally_policy: Any | None = None,
        freeze_completed_class_allies: bool = False,
        policy_router: MultiAgentPolicyRouter | None = None,
        rule_based_enemy_policy: bool | Any = False,
        random_policy: bool | Any = False,
        trainable_teams: set[Team] | list[Team] | tuple[Team, ...] | None = None,
        self_play_config: SelfPlayConfig | None = None,
        self_play_manager: SelfPlayManager | None = None,
        fast_action_masks: bool = False,
        fast_observation: bool = False,
        profile_rollout: bool = False,
        num_envs: int = 1,
    ) -> None:
        self.environment = environment
        self.num_envs = max(1, int(num_envs))
        self.environments: list[CombatEnvironment] = [environment]
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
        self.trainable_teams = (
            None if trainable_teams is None else {Team(team) for team in trainable_teams}
        )
        self.ally_policy = ally_policy
        self.freeze_completed_class_allies = bool(freeze_completed_class_allies)
        self.frozen_ally_policy: Any | None = None
        self.self_play_manager = self_play_manager or (
            SelfPlayManager(self_play_config) if self_play_config is not None else None
        )
        self.fast_action_masks = bool(fast_action_masks)
        self.fast_observation = bool(fast_observation)
        self.profile_rollout = bool(profile_rollout)
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
        self.current_curriculum_level = None
        if self.curriculum_config.enabled:
            if self.encounter_generator is not None:
                self.current_curriculum_level = (
                    self.encounter_generator.clamp_curriculum_level(
                        self.curriculum_config.initial_level
                    )
                )
            else:
                self.current_curriculum_level = clamp_curriculum_level(
                    self.curriculum_config.initial_level
                )
        if self.current_curriculum_level is not None:
            self.current_curriculum_level = min(
                self.current_curriculum_level,
                self.curriculum_config.max_level,
            )
        self.curriculum_recent_wins: list[bool] = []
        self.curriculum_transition_log: list[str] = []
        self.curriculum_updates_on_level = 0
        self.curriculum_promoted_since_update = False
        self.curriculum_stage_changed = False
        self.curriculum_success_team = self._resolve_curriculum_success_team()
        if self.curriculum_config.enabled and self.encounter_generator is not None:
            self.encounter_generator.curriculum_kind = self.curriculum_config.kind
            self.encounter_generator.rehearsal_probability = (
                self.curriculum_config.rehearsal_probability
            )
            self.encounter_generator.set_curriculum_level(self.current_curriculum_level)
        self.completed_training_classes = self._completed_classes_before_current_stage()
        self._refresh_frozen_ally_policy()
        self._initialize_parallel_environments()
        self.episode_steps_by_env = [0 for _ in self.environments]
        self.target_class_activity_by_env = [False for _ in self.environments]

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
        self.target_class_activity_by_env[0] = False

        rollout = RolloutBuffer()
        action_counts = _empty_action_counts()
        self._set_policies_eval()

        for _ in range(steps):
            if self.environment.is_done():
                break

            state = self.environment.combat_state
            actor_id = self._active_actor_id()
            actor = state.character_at(actor_id)
            policy = self._policy_for_actor(state, actor_id)
            actor_observation = self._actor_observation(state, actor_id, policy)
            critic_observation = self._critic_observation(state, actor_id, policy)
            raw_masks = self._build_action_masks(state, actor_id)
            masks = self._padded_masks(raw_masks, policy)
            training_masks = self._padded_masks(raw_masks, self.model)

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
            self._record_target_class_activity(0, state, actor, action, result)
            done = self.environment.is_done()
            rollout.append(
                actor_observation,
                action_output,
                result.reward,
                done,
                training_masks,
                critic_observation=critic_observation,
                actor_id=actor_id,
                team=actor.team if actor is not None else None,
                actor_class=getattr(actor, "class_name", None),
                trainable_transition=self._is_transition_trainable(actor, state),
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
        self._apply_terminal_training_reward(
            rollout,
            0,
            winner=stats.winner,
            timeout=stats.winner is None and not self.environment.is_done(),
        )
        stats = EpisodeStats(
            total_reward=sum(rollout.rewards),
            length=len(rollout),
            winner=stats.winner,
            action_counts=action_counts,
        )
        if stats.winner is not None:
            rollout.episode_winners.append(stats.winner)
        self.record_curriculum_result(
            stats.winner,
            include=not state.curriculum_is_rehearsal,
            target_class_active=self.target_class_activity_by_env[0],
        )
        return rollout, stats

    def collect_rollout(
        self,
        rollout_steps: int | None = None,
        max_episode_steps: int | None = None,
        max_episode_steps_per_creature: int | None = None,
    ) -> RolloutBuffer:
        """Collect one fixed-length rollout from one or more environments."""

        steps = rollout_steps or self.config.rollout_steps
        if steps <= 0:
            raise ValueError("rollout_steps must be greater than zero")
        if max_episode_steps is not None and max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be greater than zero")
        if (
            max_episode_steps_per_creature is not None
            and max_episode_steps_per_creature <= 0
        ):
            raise ValueError("max_episode_steps_per_creature must be greater than zero")

        rollout = RolloutBuffer()
        if self.profile_rollout:
            rollout.profile_times = {
                "observation": 0.0,
                "mask": 0.0,
                "model_act": 0.0,
                "decode": 0.0,
                "env_step": 0.0,
                "update": 0.0,
            }
        self._ensure_episode_step_counters()
        self._set_policies_eval()

        while len(rollout) < steps:
            active_entries = self._prepare_rollout_entries(
                rollout,
                steps - len(rollout),
                self.episode_steps_by_env,
            )
            if not active_entries:
                break

            started = time.perf_counter()
            with torch.no_grad():
                action_outputs = self._act_for_rollout_entries(active_entries)
            self._add_profile_time(rollout, "model_act", started)

            for entry, action_output in zip(active_entries, action_outputs):
                env_index = entry["env_index"]
                environment = self.environments[env_index]
                state = entry["state"]
                actor_id = entry["actor_id"]
                actor = entry["actor"]

                started = time.perf_counter()
                action = self._decode_model_action(
                    action_output,
                    state,
                    actor_id,
                    entry["raw_masks"],
                )
                self._add_profile_time(rollout, "decode", started)

                started = time.perf_counter()
                result = environment.step(action)
                self._record_target_class_activity(
                    env_index,
                    state,
                    actor,
                    action,
                    result,
                )
                self._add_profile_time(rollout, "env_step", started)
                self.episode_steps_by_env[env_index] += 1
                environment_done = environment.is_done()
                episode_step_limit = _effective_episode_step_limit(
                    environment,
                    max_episode_steps,
                    max_episode_steps_per_creature,
                )
                timeout = (
                    episode_step_limit is not None
                    and self.episode_steps_by_env[env_index] >= episode_step_limit
                    and not environment_done
                )
                done = environment_done or timeout

                rollout.append(
                    entry["actor_observation"],
                    action_output,
                    result.reward,
                    done,
                    entry["training_masks"],
                    critic_observation=entry["critic_observation"],
                    actor_id=actor_id,
                    team=actor.team,
                    actor_class=actor.class_name,
                    trainable_transition=self._is_transition_trainable(actor, state),
                    env_id=env_index,
                )
                if environment_done:
                    winner = environment.get_winner()
                    self._apply_terminal_training_reward(
                        rollout,
                        env_index,
                        winner=winner,
                    )
                    rollout.episode_winners.append(winner)
                    self.record_curriculum_result(
                        winner,
                        include=not state.curriculum_is_rehearsal,
                        target_class_active=self.target_class_activity_by_env[env_index],
                    )
                    self._reset_environment_for_episode(env_index)
                elif timeout:
                    self._apply_terminal_training_reward(
                        rollout,
                        env_index,
                        winner=None,
                        timeout=True,
                    )
                    rollout.episode_winners.append(None)
                    rollout.episode_timeouts += 1
                    self.record_curriculum_result(
                        None,
                        include=not state.curriculum_is_rehearsal,
                        target_class_active=self.target_class_activity_by_env[env_index],
                    )
                    self._reset_environment_for_episode(env_index)

        rollout.last_value = self._bootstrap_value()
        rollout.last_values_by_env = {
            env_index: float(self._bootstrap_value_for_environment(environment))
            for env_index, environment in enumerate(self.environments)
        }
        return rollout

    def _prepare_rollout_entries(
        self,
        rollout: RolloutBuffer,
        remaining_steps: int,
        episode_steps: list[int],
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for env_index, environment in enumerate(self.environments):
            if len(entries) >= remaining_steps:
                break
            if environment.is_done():
                self._reset_environment_for_episode(env_index)
                episode_steps[env_index] = 0
                environment = self.environments[env_index]

            state = environment.combat_state
            actor_id = self._active_actor_id_for_environment(environment)
            actor = state.character_at(actor_id)
            if actor is None:
                raise ValueError("environment has no active actor")

            policy = self._policy_for_actor(state, actor_id)
            started = time.perf_counter()
            actor_observation = self._actor_observation(state, actor_id, policy)
            critic_observation = self._critic_observation(state, actor_id, policy)
            self._add_profile_time(rollout, "observation", started)

            started = time.perf_counter()
            raw_masks = self._build_action_masks(state, actor_id)
            masks = self._padded_masks(raw_masks, policy)
            training_masks = self._padded_masks(raw_masks, self.model)
            self._add_profile_time(rollout, "mask", started)

            entries.append(
                {
                    "env_index": env_index,
                    "state": state,
                    "actor_id": actor_id,
                    "actor": actor,
                    "policy": policy,
                    "actor_observation": actor_observation,
                    "critic_observation": critic_observation,
                    "raw_masks": raw_masks,
                    "masks": masks,
                    "training_masks": training_masks,
                }
            )
        return entries

    def _act_for_rollout_entries(
        self,
        entries: list[dict[str, Any]],
    ) -> list[dict[str, torch.Tensor]]:
        if not entries:
            return []
        if self._can_batch_rollout_entries(entries):
            policy = entries[0]["policy"]
            actor_observation = _stack_observations(
                [entry["actor_observation"] for entry in entries],
                self.device,
            )
            critic_observation = _stack_observations(
                [entry["critic_observation"] for entry in entries],
                self.device,
            )
            masks = _stack_mask_batch([entry["masks"] for entry in entries], self.device)
            output = self._policy_act(
                policy,
                actor_observation,
                masks,
                critic_observation,
                entries[0]["state"],
                entries[0]["actor_id"],
            )
            return [_select_action_output(output, index) for index in range(len(entries))]

        outputs = []
        for entry in entries:
            outputs.append(
                self._policy_act(
                    entry["policy"],
                    entry["actor_observation"],
                    entry["masks"],
                    entry["critic_observation"],
                    entry["state"],
                    entry["actor_id"],
                )
            )
        return outputs

    def _can_batch_rollout_entries(self, entries: list[dict[str, Any]]) -> bool:
        if len(entries) <= 1:
            return False
        policy = entries[0]["policy"]
        if not isinstance(policy, (GNNPPOActorCritic, PPOActorCritic)):
            return False
        return all(entry["policy"] is policy for entry in entries)

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
        advantages = torch.zeros_like(rewards)
        if len(rollout.next_values) == len(rollout):
            next_values = torch.stack(rollout.next_values).to(device=self.device)
            env_ids = (
                rollout.env_ids
                if len(rollout.env_ids) == len(rollout)
                else [0 for _ in rollout.rewards]
            )
            gae_by_env: dict[int, torch.Tensor] = {}
            zero = torch.tensor(0.0, device=self.device)
            for step in reversed(range(len(rollout))):
                env_id = int(env_ids[step])
                next_non_terminal = 1.0 - dones[step]
                delta = (
                    rewards[step]
                    + self.config.gamma * next_values[step] * next_non_terminal
                    - values[step]
                )
                gae = gae_by_env.get(env_id, zero)
                gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * gae
                advantages[step] = gae
                gae_by_env[env_id] = gae
            returns = advantages + values
            return returns.detach(), advantages.detach()

        if len(rollout.env_ids) == len(rollout) and rollout.last_values_by_env:
            next_value_by_env = {
                int(env_id): torch.tensor(value, dtype=torch.float32, device=self.device)
                for env_id, value in rollout.last_values_by_env.items()
            }
            gae_by_env: dict[int, torch.Tensor] = {}
            zero = torch.tensor(0.0, device=self.device)
            for step in reversed(range(len(rollout))):
                env_id = int(rollout.env_ids[step])
                next_value = next_value_by_env.get(env_id, zero)
                next_non_terminal = 1.0 - dones[step]
                delta = (
                    rewards[step]
                    + self.config.gamma * next_value * next_non_terminal
                    - values[step]
                )
                gae = gae_by_env.get(env_id, zero)
                gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * gae
                advantages[step] = gae
                gae_by_env[env_id] = gae
                next_value_by_env[env_id] = values[step]
            returns = advantages + values
            return returns.detach(), advantages.detach()

        last_value = torch.tensor(
            rollout.last_value,
            dtype=torch.float32,
            device=self.device,
        )
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

        update_started = time.perf_counter()
        self.model.train()
        data = rollout.to_tensors(self.device)
        _sanitize_action_batches_for_masks(data["actions"], data["masks"])
        returns, advantages = self.compute_returns_and_advantages(rollout)
        advantages = _normalize_advantages(advantages)

        train_indices = self._trainable_indices(
            data["team_ids"],
            data["trainable_flags"],
        )
        batch_size = int(train_indices.numel())
        minibatch_size = min(self.config.minibatch_size, batch_size)
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "loss": 0.0,
        }
        if batch_size == 0:
            self.record_curriculum_update()
            return metrics
        updates = 0

        for _ in range(self.config.update_epochs):
            indices = train_indices[torch.randperm(batch_size, device=self.device)]
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
        if self.profile_rollout:
            rollout.profile_times["update"] = time.perf_counter() - update_started
        self.record_curriculum_update()
        return metrics

    def train_iteration(
        self,
        save_checkpoint: bool = True,
        checkpoint_name: str = "ppo_actor_critic.pt",
    ) -> dict[str, float | str]:
        """Collect rollout, update model, and optionally save a checkpoint."""

        if self.self_play_manager is not None:
            self.self_play_manager.before_rollout(self)
        rollout = self.collect_rollout()
        metrics: dict[str, float | str] = self.update(rollout)
        if self.self_play_manager is not None:
            metrics.update(self.self_play_manager.after_update(self, rollout, metrics))
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

    def record_curriculum_result(
        self,
        winner: Team | None,
        *,
        include: bool = True,
        target_class_active: bool = True,
    ) -> None:
        """Track an episode result and promote curriculum difficulty when ready."""

        if (
            not include
            or not self.curriculum_config.enabled
            or self.current_curriculum_level is None
            or self.curriculum_promoted_since_update
        ):
            return

        qualified_win = (
            winner is self.curriculum_success_team
            and (
                not self._current_stage_requires_class_activity()
                or target_class_active
            )
        )
        self.curriculum_recent_wins.append(qualified_win)
        window_size = max(1, self.curriculum_config.window_size)
        if len(self.curriculum_recent_wins) > window_size:
            self.curriculum_recent_wins = self.curriculum_recent_wins[-window_size:]
        if len(self.curriculum_recent_wins) < window_size:
            return
        if self.curriculum_updates_on_level < self.curriculum_config.min_updates_per_level:
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
        self.curriculum_updates_on_level = 0
        self.curriculum_promoted_since_update = True
        self.curriculum_stage_changed = True
        if self.encounter_generator is not None:
            self.encounter_generator.set_curriculum_level(self.current_curriculum_level)
        previous_stage = self._get_curriculum_stage(previous_level)
        next_stage = self._get_curriculum_stage(self.current_curriculum_level)
        if previous_stage.phase != next_stage.phase:
            self.completed_training_classes.update(previous_stage.trainable_classes)
            self._refresh_frozen_ally_policy()
        message = (
            f"Curriculum advanced from level {previous_level} "
            f"({previous_stage.name}) to level {self.current_curriculum_level} "
            f"({next_stage.name})."
        )
        self.curriculum_transition_log.append(message)
        LOGGER.info(message)

    def record_curriculum_update(self) -> None:
        """Count one optimizer update spent on the current curriculum level."""

        if self.curriculum_config.enabled and self.current_curriculum_level is not None:
            self.curriculum_updates_on_level += 1
            self.curriculum_promoted_since_update = False
            if self.curriculum_stage_changed:
                self._refresh_frozen_ally_policy()
                for env_index in range(len(self.environments)):
                    self._reset_environment_for_episode(env_index)
                self.curriculum_stage_changed = False

    def curriculum_state_dict(self) -> dict[str, object]:
        """Return serializable curriculum training state."""

        stage_name = (
            self._get_curriculum_stage(self.current_curriculum_level).name
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
            "min_updates_per_level": self.curriculum_config.min_updates_per_level,
            "updates_on_level": self.curriculum_updates_on_level,
            "success_team": self.curriculum_success_team.value,
            "recent_wins": list(self.curriculum_recent_wins),
            "current_win_rate": self.current_curriculum_win_rate,
            "transition_log": list(self.curriculum_transition_log),
            "kind": self.curriculum_config.kind,
            "rehearsal_probability": self.curriculum_config.rehearsal_probability,
            "terminal_defeat_penalty": self.curriculum_config.terminal_defeat_penalty,
            "timeout_penalty": self.curriculum_config.timeout_penalty,
            "completed_training_classes": sorted(self.completed_training_classes),
        }

    def _bootstrap_value(self) -> float:
        return float(self._bootstrap_value_for_environment(self.environment))

    def _bootstrap_value_for_environment(self, environment: CombatEnvironment) -> torch.Tensor:
        if environment.is_done():
            return torch.tensor(0.0)

        state = environment.combat_state
        actor_id = self._active_actor_id_for_environment(environment)
        with torch.no_grad():
            policy = self._policy_for_actor(state, actor_id)
            actor_observation = self._actor_observation(state, actor_id, policy)
            critic_observation = self._critic_observation(state, actor_id, policy)
            value = self._policy_value(policy, actor_observation, critic_observation).squeeze(0)
        return value.detach().cpu()

    def _active_actor_id_for_environment(self, environment: CombatEnvironment) -> int:
        state = environment.combat_state
        actor_id = state.active_actor_id
        if actor_id is None:
            raise ValueError("environment has no characters")
        return actor_id

    def _active_actor_id(self) -> int:
        return self._active_actor_id_for_environment(self.environment)

    def policy_for_actor(self, actor_id: int) -> Any:
        """Return the policy configured for an actor in the current environment."""

        return self._policy_for_actor(self.environment.combat_state, actor_id)

    def _policy_for_actor(self, state: Any, actor_id: int) -> Any:
        actor = state.character_at(actor_id)
        if actor is None:
            raise ValueError(f"Actor {actor_id} not found")
        training_classes = set(getattr(state, "training_classes", ()))
        if (
            actor.team is Team.PLAYERS
            and training_classes
            and actor.class_name not in training_classes
        ):
            if (
                actor.class_name in self.completed_training_classes
                and self.frozen_ally_policy is not None
            ):
                return self.frozen_ally_policy
            if self.ally_policy is not None:
                return self.ally_policy
        return self.policy_router.policy_for(actor)

    def _actor_observation(self, state: Any, actor_id: int, policy: Any | None = None) -> Any:
        fast = self._use_fast_training_mode(state)
        if self._uses_entity_observation(policy):
            return encode_entity_observation(
                state,
                actor_id,
                fast=self.fast_observation and fast,
            )
        return encode_observation(
            state,
            actor_id,
            fast=self.fast_observation and fast,
        ).to(self.device)

    def _critic_observation(self, state: Any, actor_id: int, policy: Any | None = None) -> Any:
        fast = self._use_fast_training_mode(state)
        if self._uses_entity_observation(policy):
            return encode_entity_observation(
                state,
                actor_id,
                fast=self.fast_observation and fast,
            )
        return encode_observation(
            state,
            actor_id,
            fast=self.fast_observation and fast,
        ).to(self.device)

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
        masks: dict[str, torch.Tensor] | None = None,
    ) -> Any:
        try:
            slot_level_output = action_output.get("slot_level")
            if slot_level_output is None:
                slot_level_output = torch.zeros_like(action_output["action_category"])
            action_category, main_action_type, target_index, move_index, option_index, slot_level = (
                torch.stack(
                    (
                        action_output["action_category"].reshape(()),
                        action_output["main_action_type"].reshape(()),
                        action_output["target_index"].reshape(()),
                        action_output["move_index"].reshape(()),
                        action_output["option_index"].reshape(()),
                        slot_level_output.reshape(()),
                    )
                )
                .detach()
                .cpu()
                .tolist()
            )
            use_fast = self._use_fast_training_mode(state)
            decoder = decode_fast_training_action if use_fast else decode_action
            if use_fast:
                return decoder(
                    int(action_category),
                    int(main_action_type),
                    int(target_index),
                    int(move_index),
                    int(option_index),
                    state,
                    actor_id,
                    masks=masks,
                )
            return decoder(
                int(action_category),
                int(main_action_type),
                int(target_index),
                int(move_index),
                int(option_index),
                state,
                actor_id,
                slot_level=int(slot_level) if not use_fast else None,
                masks=masks,
            )
        except ValueError:
            return EndTurnAction(actor_id=actor_id)

    def _build_action_masks(self, state: Any, actor_id: int) -> dict[str, torch.Tensor]:
        if self._use_fast_training_mode(state):
            return build_fast_training_action_masks(state, actor_id)
        return build_action_masks(state, actor_id)

    def _use_fast_training_mode(self, state: Any) -> bool:
        if not self.fast_action_masks and not self.fast_observation:
            return False
        if self.curriculum_config.kind != "class":
            return True
        level = getattr(state, "curriculum_source_level", None)
        if level is None:
            level = self.current_curriculum_level
        if level is None:
            return False
        return self._get_curriculum_stage(level).phase == "fighter"

    def _add_profile_time(
        self,
        rollout: RolloutBuffer,
        key: str,
        started: float,
    ) -> None:
        if not self.profile_rollout:
            return
        rollout.profile_times[key] = rollout.profile_times.get(key, 0.0) + (
            time.perf_counter() - started
        )

    def _initialize_parallel_environments(self) -> None:
        if self.num_envs <= 1:
            return
        if self.encounter_generator is None:
            self.num_envs = 1
            return
        use_initiative = getattr(self.environment, "use_initiative", True)
        log_to_console = getattr(self.environment, "log_to_console", True)
        self.environments = [self.environment]
        for _ in range(self.num_envs - 1):
            self.environments.append(
                self._generate_environment_for_episode(
                    use_initiative=use_initiative,
                    log_to_console=log_to_console,
                )
            )

    def _ensure_episode_step_counters(self) -> None:
        if not hasattr(self, "episode_steps_by_env"):
            self.episode_steps_by_env = []
        if len(self.episode_steps_by_env) < len(self.environments):
            self.episode_steps_by_env.extend(
                0 for _ in range(len(self.environments) - len(self.episode_steps_by_env))
            )
        elif len(self.episode_steps_by_env) > len(self.environments):
            self.episode_steps_by_env = self.episode_steps_by_env[: len(self.environments)]
        if not hasattr(self, "target_class_activity_by_env"):
            self.target_class_activity_by_env = []
        if len(self.target_class_activity_by_env) < len(self.environments):
            self.target_class_activity_by_env.extend(
                False
                for _ in range(
                    len(self.environments) - len(self.target_class_activity_by_env)
                )
            )
        elif len(self.target_class_activity_by_env) > len(self.environments):
            self.target_class_activity_by_env = self.target_class_activity_by_env[
                : len(self.environments)
            ]

    def _generate_environment_for_episode(
        self,
        *,
        use_initiative: bool,
        log_to_console: bool,
    ) -> CombatEnvironment:
        if self.encounter_generator is None:
            raise ValueError("encounter_generator is required for multiple environments")
        if self.curriculum_config.enabled:
            self.encounter_generator.set_curriculum_level(self.current_curriculum_level)
            return self.encounter_generator.generate_curriculum_environment(
                self.current_curriculum_level,
                use_initiative=use_initiative,
                log_to_console=log_to_console,
            )
        return self.encounter_generator.generate_environment(
            use_initiative=use_initiative,
            log_to_console=log_to_console,
        )

    def _reset_environment_for_episode(self, env_index: int = 0) -> None:
        self._ensure_episode_step_counters()
        environment = self.environments[env_index]
        if self.curriculum_config.enabled and self.encounter_generator is not None:
            self.encounter_generator.set_curriculum_level(self.current_curriculum_level)
            self.environments[env_index] = self.encounter_generator.generate_curriculum_environment(
                self.current_curriculum_level,
                use_initiative=getattr(environment, "use_initiative", True),
                log_to_console=getattr(environment, "log_to_console", True),
            )
            if env_index == 0:
                self.environment = self.environments[0]
            self.episode_steps_by_env[env_index] = 0
            self.target_class_activity_by_env[env_index] = False
            return
        if self.encounter_generator is not None and self.num_envs > 1:
            self.environments[env_index] = self.encounter_generator.generate_environment(
                use_initiative=getattr(environment, "use_initiative", True),
                log_to_console=getattr(environment, "log_to_console", True),
            )
            if env_index == 0:
                self.environment = self.environments[0]
            self.episode_steps_by_env[env_index] = 0
            self.target_class_activity_by_env[env_index] = False
            return
        environment.reset()
        if env_index == 0:
            self.environment = environment
        self.episode_steps_by_env[env_index] = 0
        self.target_class_activity_by_env[env_index] = False

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
        if self.encounter_generator is not None:
            self.encounter_generator.curriculum_kind = base.kind
            max_level = max(
                1,
                min(self.encounter_generator.max_curriculum_level, int(base.max_level)),
            )
        return CurriculumConfig(
            enabled=enabled,
            initial_level=min(
                (
                    self.encounter_generator.clamp_curriculum_level(initial_level)
                    if self.encounter_generator is not None
                    else clamp_curriculum_level(initial_level)
                ),
                max_level,
            ),
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
            min_updates_per_level=max(0, int(base.min_updates_per_level)),
            kind=base.kind,
            rehearsal_probability=max(
                0.0,
                min(1.0, float(base.rehearsal_probability)),
            ),
            terminal_defeat_penalty=max(0.0, float(base.terminal_defeat_penalty)),
            timeout_penalty=max(0.0, float(base.timeout_penalty)),
        )

    def _resolve_curriculum_success_team(self) -> Team:
        if self.trainable_teams == {Team.ENEMIES}:
            return Team.ENEMIES
        return Team.PLAYERS

    def _trainable_indices(
        self,
        team_ids: torch.Tensor,
        trainable_flags: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.trainable_teams is None:
            mask = torch.ones_like(team_ids, dtype=torch.bool)
        else:
            allowed_team_ids = {
                _team_id(team)
                for team in self.trainable_teams
            }
            if not allowed_team_ids:
                return torch.empty(0, dtype=torch.long, device=self.device)
            mask = torch.zeros_like(team_ids, dtype=torch.bool)
            for team_id in allowed_team_ids:
                mask = mask | (team_ids == int(team_id))
        if trainable_flags is not None:
            mask = mask & trainable_flags.to(device=self.device, dtype=torch.bool)
        return torch.nonzero(mask, as_tuple=False).reshape(-1)

    def _is_transition_trainable(self, actor: Any, state: Any) -> bool:
        if actor is None:
            return False
        if self.trainable_teams is not None and actor.team not in self.trainable_teams:
            return False
        training_classes = set(getattr(state, "training_classes", ()))
        return not training_classes or actor.class_name in training_classes

    def _record_target_class_activity(
        self,
        env_index: int,
        state: Any,
        actor: Any,
        action: Any,
        result: Any,
    ) -> None:
        if not getattr(result, "success", False):
            return
        training_classes = set(getattr(state, "training_classes", ()))
        if actor is None or actor.class_name not in training_classes:
            return
        if action.__class__.__name__ in {
            "CastSpellAction",
            "ChannelDivinityPreserveLifeAction",
        }:
            self.target_class_activity_by_env[env_index] = True

    def _current_stage_requires_class_activity(self) -> bool:
        if (
            self.curriculum_config.kind != "class"
            or self.current_curriculum_level is None
        ):
            return False
        return self._get_curriculum_stage(
            self.current_curriculum_level
        ).phase in {"cleric", "wizard"}

    def _apply_terminal_training_reward(
        self,
        rollout: RolloutBuffer,
        env_index: int,
        *,
        winner: Team | None,
        timeout: bool = False,
    ) -> None:
        penalty = (
            self.curriculum_config.timeout_penalty
            if timeout
            else (
                self.curriculum_config.terminal_defeat_penalty
                if winner is not None and winner is not self.curriculum_success_team
                else 0.0
            )
        )
        if penalty <= 0.0:
            return
        success_team_id = _team_id(self.curriculum_success_team)
        for index in range(len(rollout.rewards) - 1, -1, -1):
            if rollout.env_ids[index] != env_index:
                continue
            if not rollout.trainable_flags[index]:
                continue
            if rollout.team_ids[index] != success_team_id:
                continue
            rollout.rewards[index] -= penalty
            return

    def _get_curriculum_stage(self, level: int) -> Any:
        if self.encounter_generator is not None:
            return self.encounter_generator.get_curriculum_stage(level)
        return get_curriculum_stage(level)

    def _completed_classes_before_current_stage(self) -> set[str]:
        if (
            not self.curriculum_config.enabled
            or self.current_curriculum_level is None
            or self.encounter_generator is None
            or self.curriculum_config.kind != "class"
        ):
            return set()
        current_stage = self._get_curriculum_stage(self.current_curriculum_level)
        completed: set[str] = set()
        for stage in self.encounter_generator.curriculum_stages:
            if stage.level >= current_stage.level:
                break
            if stage.phase != current_stage.phase:
                completed.update(stage.trainable_classes)
        return completed

    def _refresh_frozen_ally_policy(self) -> None:
        if not self.freeze_completed_class_allies or not self.completed_training_classes:
            self.frozen_ally_policy = None
            return
        frozen = deepcopy(self.model)
        frozen.to(self.device)
        frozen.eval()
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)
        self.frozen_ally_policy = frozen

    def _padded_masks(
        self,
        masks: dict[str, torch.Tensor],
        policy: Any | None = None,
    ) -> dict[str, torch.Tensor]:
        action_space = _policy_action_space(policy or self.model, self.model)
        padded = {
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
        for name, size in action_space.extra_heads.items():
            padded[name] = _pad_mask(
                masks.get(name, torch.zeros(0, dtype=torch.bool)),
                size,
                name,
                self.device,
                default_first=True,
            )
        return padded


def _pad_mask(
    mask: torch.Tensor,
    target_size: int,
    name: str,
    device: torch.device,
    *,
    default_first: bool = False,
) -> torch.Tensor:
    prepared = mask.to(device=device, dtype=torch.bool)
    if prepared.ndim != 1:
        raise ValueError(f"{name} mask must be a 1D tensor")
    if prepared.shape[0] > target_size:
        raise ValueError(f"{name} mask is larger than the model head")
    if prepared.shape[0] == target_size:
        if prepared.any() or not default_first or target_size <= 0:
            return prepared
        prepared = prepared.clone()
        prepared[0] = True
        return prepared

    padding = torch.zeros(
        target_size - prepared.shape[0],
        dtype=torch.bool,
        device=device,
    )
    padded = torch.cat((prepared, padding), dim=0)
    if default_first and target_size > 0 and not padded.any():
        padded[0] = True
    return padded


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
    extra_heads = {}
    for attribute, mask_name in (
        ("bonus_action_type_count", "bonus_action_type"),
        ("reaction_type_count", "reaction_type"),
        ("class_feature_count", "class_feature"),
        ("spell_count", "spell_index"),
        ("slot_level_count", "slot_level"),
        ("item_count", "item_index"),
    ):
        if hasattr(source, attribute):
            extra_heads[mask_name] = int(getattr(source, attribute))
    return SimpleNamespace(
        action_category_count=int(getattr(source, "action_category_count")),
        main_action_type_count=int(getattr(source, "main_action_type_count")),
        target_count=int(getattr(source, "target_count")),
        move_count=int(getattr(source, "move_count")),
        option_count=int(getattr(source, "option_count")),
        extra_heads=extra_heads,
    )


def _team_id(team: Team | None) -> int:
    if team is Team.PLAYERS:
        return 0
    if team is Team.ENEMIES:
        return 1
    return -1


def _class_id(class_name: str | None) -> int:
    normalized = str(class_name or "").strip().lower()
    return {
        "fighter": 1,
        "cleric": 2,
        "wizard": 3,
    }.get(normalized, 0)


def _sanitize_action_for_masks(
    action: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Replace impossible auxiliary head choices before storing a rollout row."""

    sanitized: dict[str, torch.Tensor] = {}
    for key, value in action.items():
        if key in {"log_prob", "entropy", "value"} or key not in masks:
            continue
        selected = value.detach().long().reshape(())
        mask = masks[key].detach().bool()
        if mask.ndim != 1:
            sanitized[key] = selected
            continue
        index = int(selected.item())
        if 0 <= index < int(mask.shape[0]) and bool(mask[index]):
            sanitized[key] = selected
            continue
        valid_indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
        fallback = int(valid_indices[0].item()) if valid_indices.numel() else 0
        sanitized[key] = torch.tensor(fallback, dtype=torch.long)
    return sanitized


def _sanitize_action_batches_for_masks(
    actions: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
) -> None:
    """Repair stale auxiliary choices before PPO validation."""

    for key, action in actions.items():
        mask = masks.get(key)
        if mask is None or action.ndim != 1 or mask.ndim != 2:
            continue
        if action.shape[0] != mask.shape[0]:
            continue
        safe_action = action.clamp(min=0, max=max(0, mask.shape[1] - 1))
        rows = torch.arange(action.shape[0], device=action.device)
        valid = (action >= 0) & (action < mask.shape[1])
        if mask.shape[1] > 0:
            valid = valid & mask[rows, safe_action]
        invalid_rows = torch.nonzero(~valid, as_tuple=False).reshape(-1)
        for row in invalid_rows.tolist():
            allowed = torch.nonzero(mask[row], as_tuple=False).reshape(-1)
            action[row] = int(allowed[0].item()) if allowed.numel() else 0


def _effective_episode_step_limit(
    environment: CombatEnvironment,
    max_episode_steps: int | None,
    max_episode_steps_per_creature: int | None,
) -> int | None:
    """Return a timeout that grows with encounter size."""

    limits: list[int] = []
    if max_episode_steps is not None:
        limits.append(int(max_episode_steps))
    if max_episode_steps_per_creature is not None:
        creature_count = len(getattr(environment.combat_state, "characters", []) or [])
        limits.append(max(1, creature_count) * int(max_episode_steps_per_creature))
    if not limits:
        return None
    return max(limits)


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


def _stack_mask_batch(
    masks: list[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if not masks:
        raise ValueError("masks are empty")
    keys = masks[0].keys()
    return {
        key: torch.stack([item[key] for item in masks]).to(device=device)
        for key in keys
    }


def _select_action_output(
    action_output: dict[str, torch.Tensor],
    index: int,
) -> dict[str, torch.Tensor]:
    selected: dict[str, torch.Tensor] = {}
    for key, value in action_output.items():
        selected[key] = value[index] if torch.is_tensor(value) and value.ndim > 0 else value
    return selected


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
