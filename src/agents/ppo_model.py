"""PPO actor-critic model for hierarchical combat actions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.distributions import Categorical

from agents.action_space import ACTION_TYPE_COUNT, ActionType
from agents.observation import OBSERVATION_SIZE


DEFAULT_TARGET_COUNT = 8
DEFAULT_MOVE_COUNT = 64
MASKED_LOGIT_VALUE = -1.0e9


class PPOActorCritic(nn.Module):
    """Shared-encoder actor-critic network for PPO."""

    def __init__(
        self,
        observation_size: int = OBSERVATION_SIZE,
        target_count: int = DEFAULT_TARGET_COUNT,
        move_count: int = DEFAULT_MOVE_COUNT,
        action_type_count: int = ACTION_TYPE_COUNT,
        hidden_sizes: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        if observation_size <= 0:
            raise ValueError("observation_size must be greater than zero")
        if target_count <= 0:
            raise ValueError("target_count must be greater than zero")
        if move_count <= 0:
            raise ValueError("move_count must be greater than zero")
        if action_type_count <= 0:
            raise ValueError("action_type_count must be greater than zero")

        self.observation_size = observation_size
        self.target_count = target_count
        self.move_count = move_count
        self.action_type_count = action_type_count

        self.encoder = _build_mlp(observation_size, hidden_sizes)
        encoder_size = hidden_sizes[-1] if hidden_sizes else observation_size

        self.action_type_head = nn.Linear(encoder_size, action_type_count)
        self.target_head = nn.Linear(encoder_size, target_count)
        self.move_head = nn.Linear(encoder_size, move_count)
        self.value_head = nn.Linear(encoder_size, 1)

    def forward(self, observations: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return raw policy logits and value estimates."""

        batched_observations, _ = _ensure_batched_observations(
            observations,
            self.observation_size,
        )
        encoded = self.encoder(batched_observations)
        return {
            "action_type_logits": self.action_type_head(encoded),
            "target_logits": self.target_head(encoded),
            "move_logits": self.move_head(encoded),
            "value": self.value_head(encoded).squeeze(-1),
        }

    def act(
        self,
        observation: torch.Tensor,
        masks: Mapping[str, torch.Tensor],
        deterministic: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Sample or greedily select a hierarchical action under masks."""

        batched_observation, single_observation = _ensure_batched_observations(
            observation,
            self.observation_size,
        )
        outputs = self.forward(batched_observation)
        batch_size = batched_observation.shape[0]
        distributions = self._build_distributions(outputs, masks, batch_size)

        action_type = _select_from_distribution(
            distributions["action_type"],
            deterministic,
        )
        target_index = _select_from_distribution(
            distributions["target_index"],
            deterministic,
        )
        move_index = _select_from_distribution(
            distributions["move_index"],
            deterministic,
        )

        log_prob, entropy = self._combine_action_log_probs(
            distributions,
            action_type,
            target_index,
            move_index,
        )

        result = {
            "action_type": action_type,
            "target_index": target_index,
            "move_index": move_index,
            "log_prob": log_prob,
            "entropy": entropy,
            "value": outputs["value"],
        }
        if single_observation:
            return {key: value.squeeze(0) for key, value in result.items()}
        return result

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: Mapping[str, torch.Tensor | int],
        masks: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Evaluate sampled actions for PPO loss calculation."""

        batched_observations, _ = _ensure_batched_observations(
            observations,
            self.observation_size,
        )
        outputs = self.forward(batched_observations)
        batch_size = batched_observations.shape[0]
        distributions = self._build_distributions(outputs, masks, batch_size)

        action_type = _action_tensor(actions, "action_type", batch_size, outputs["value"].device)
        target_index = _action_tensor(
            actions,
            "target_index",
            batch_size,
            outputs["value"].device,
            default=0,
        )
        move_index = _action_tensor(
            actions,
            "move_index",
            batch_size,
            outputs["value"].device,
            default=0,
        )
        self._validate_selected_actions(
            distributions["masks"],
            action_type,
            target_index,
            move_index,
        )

        log_prob, entropy = self._combine_action_log_probs(
            distributions,
            action_type,
            target_index,
            move_index,
        )
        return {
            "log_prob": log_prob,
            "entropy": entropy,
            "value": outputs["value"],
        }

    def _build_distributions(
        self,
        outputs: Mapping[str, torch.Tensor],
        masks: Mapping[str, torch.Tensor],
        batch_size: int,
    ) -> dict[str, Categorical | dict[str, torch.Tensor]]:
        action_type_mask = _prepare_mask(
            masks.get("action_type"),
            batch_size,
            self.action_type_count,
            outputs["value"].device,
            "action_type",
            allow_empty=False,
        )
        target_mask = _prepare_mask(
            masks.get("target_index"),
            batch_size,
            self.target_count,
            outputs["value"].device,
            "target_index",
            allow_empty=True,
        )
        move_mask = _prepare_mask(
            masks.get("move_index"),
            batch_size,
            self.move_count,
            outputs["value"].device,
            "move_index",
            allow_empty=True,
        )

        return {
            "action_type": Categorical(
                logits=_mask_logits(outputs["action_type_logits"], action_type_mask)
            ),
            "target_index": Categorical(
                logits=_mask_logits(outputs["target_logits"], target_mask)
            ),
            "move_index": Categorical(
                logits=_mask_logits(outputs["move_logits"], move_mask)
            ),
            "masks": {
                "action_type": action_type_mask,
                "target_index": target_mask,
                "move_index": move_mask,
            },
        }

    def _combine_action_log_probs(
        self,
        distributions: Mapping[str, Categorical | dict[str, torch.Tensor]],
        action_type: torch.Tensor,
        target_index: torch.Tensor,
        move_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_type_distribution = distributions["action_type"]
        target_distribution = distributions["target_index"]
        move_distribution = distributions["move_index"]
        if not isinstance(action_type_distribution, Categorical):
            raise TypeError("action_type distribution is missing")
        if not isinstance(target_distribution, Categorical):
            raise TypeError("target_index distribution is missing")
        if not isinstance(move_distribution, Categorical):
            raise TypeError("move_index distribution is missing")

        action_type_log_prob = action_type_distribution.log_prob(action_type)
        target_log_prob = target_distribution.log_prob(target_index)
        move_log_prob = move_distribution.log_prob(move_index)

        action_type_entropy = action_type_distribution.entropy()
        target_entropy = target_distribution.entropy()
        move_entropy = move_distribution.entropy()

        attack_selected = action_type == int(ActionType.MAIN_ACTION_ATTACK)
        move_selected = action_type == int(ActionType.MOVE)

        zero = torch.zeros_like(action_type_log_prob)
        log_prob = (
            action_type_log_prob
            + torch.where(attack_selected, target_log_prob, zero)
            + torch.where(move_selected, move_log_prob, zero)
        )
        entropy = (
            action_type_entropy
            + torch.where(attack_selected, target_entropy, zero)
            + torch.where(move_selected, move_entropy, zero)
        )
        return log_prob, entropy

    @staticmethod
    def _validate_selected_actions(
        masks: Mapping[str, torch.Tensor],
        action_type: torch.Tensor,
        target_index: torch.Tensor,
        move_index: torch.Tensor,
    ) -> None:
        if (action_type >= masks["action_type"].shape[1]).any():
            raise ValueError("actions contain an out-of-range action_type")

        batch_indices = torch.arange(action_type.shape[0], device=action_type.device)
        if not masks["action_type"][batch_indices, action_type].all():
            raise ValueError("actions contain a masked action_type")

        attack_selected = action_type == int(ActionType.MAIN_ACTION_ATTACK)
        if attack_selected.any() and (
            target_index[attack_selected] >= masks["target_index"].shape[1]
        ).any():
            raise ValueError("actions contain an out-of-range target_index")
        if attack_selected.any() and not masks["target_index"][
            batch_indices[attack_selected],
            target_index[attack_selected],
        ].all():
            raise ValueError("actions contain a masked target_index")

        move_selected = action_type == int(ActionType.MOVE)
        if move_selected.any() and (
            move_index[move_selected] >= masks["move_index"].shape[1]
        ).any():
            raise ValueError("actions contain an out-of-range move_index")
        if move_selected.any() and not masks["move_index"][
            batch_indices[move_selected],
            move_index[move_selected],
        ].all():
            raise ValueError("actions contain a masked move_index")


def _build_mlp(input_size: int, hidden_sizes: Sequence[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_size = input_size
    for hidden_size in hidden_sizes:
        if hidden_size <= 0:
            raise ValueError("hidden sizes must be greater than zero")
        layers.append(nn.Linear(current_size, hidden_size))
        layers.append(nn.Tanh())
        current_size = hidden_size
    return nn.Sequential(*layers)


def _ensure_batched_observations(
    observations: torch.Tensor,
    observation_size: int,
) -> tuple[torch.Tensor, bool]:
    if observations.ndim == 1:
        if observations.shape[0] != observation_size:
            raise ValueError(
                f"observation has size {observations.shape[0]}, expected {observation_size}"
            )
        return observations.unsqueeze(0), True
    if observations.ndim == 2:
        if observations.shape[1] != observation_size:
            raise ValueError(
                f"observations have size {observations.shape[1]}, expected {observation_size}"
            )
        return observations, False
    raise ValueError("observations must be a 1D or 2D tensor")


def _prepare_mask(
    mask: torch.Tensor | None,
    batch_size: int,
    action_count: int,
    device: torch.device,
    name: str,
    allow_empty: bool,
) -> torch.Tensor:
    if mask is None:
        prepared = torch.ones((batch_size, action_count), dtype=torch.bool, device=device)
    else:
        prepared = mask.to(device=device, dtype=torch.bool)
        if prepared.ndim == 1:
            prepared = prepared.unsqueeze(0)
        if prepared.ndim != 2:
            raise ValueError(f"{name} mask must be a 1D or 2D tensor")
        if prepared.shape[0] == 1 and batch_size > 1:
            prepared = prepared.expand(batch_size, prepared.shape[1])
        if prepared.shape[0] != batch_size:
            raise ValueError(
                f"{name} mask batch size {prepared.shape[0]} does not match {batch_size}"
            )
        if prepared.shape[1] > action_count:
            raise ValueError(
                f"{name} mask size {prepared.shape[1]} exceeds head size {action_count}"
            )
        if prepared.shape[1] < action_count:
            padding = torch.zeros(
                (batch_size, action_count - prepared.shape[1]),
                dtype=torch.bool,
                device=device,
            )
            prepared = torch.cat((prepared, padding), dim=1)

    empty_rows = ~prepared.any(dim=1)
    if empty_rows.any():
        if not allow_empty:
            raise ValueError(f"{name} mask has no valid actions")
        prepared = prepared.clone()
        prepared[empty_rows, 0] = True
    return prepared


def _mask_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask, MASKED_LOGIT_VALUE)


def _select_from_distribution(
    distribution: Categorical,
    deterministic: bool,
) -> torch.Tensor:
    if deterministic:
        return torch.argmax(distribution.logits, dim=-1)
    return distribution.sample()


def _action_tensor(
    actions: Mapping[str, torch.Tensor | int],
    key: str,
    batch_size: int,
    device: torch.device,
    default: int | None = None,
) -> torch.Tensor:
    if key not in actions:
        if default is None:
            raise ValueError(f"actions missing required key: {key}")
        value = torch.full((batch_size,), default, dtype=torch.long, device=device)
    else:
        raw_value = actions[key]
        value = torch.as_tensor(raw_value, dtype=torch.long, device=device)
        if value.ndim == 0:
            value = value.unsqueeze(0)
        if value.ndim != 1:
            raise ValueError(f"{key} action must be a scalar or 1D tensor")
        if value.shape[0] == 1 and batch_size > 1:
            value = value.expand(batch_size)
        if value.shape[0] != batch_size:
            raise ValueError(
                f"{key} action batch size {value.shape[0]} does not match {batch_size}"
            )
    if (value < 0).any():
        raise ValueError(f"{key} action contains a negative index")
    return value
