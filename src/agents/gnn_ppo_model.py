"""GNN-backed PPO actor-critic for entity-based combat observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical

from agents.action_space import ACTION_CATEGORY_COUNT, MAIN_ACTION_TYPE_COUNT
from agents.entity_observation import EntityObservation
from agents.gnn_encoder import CombatGNNEncoder
from agents.observation import (
    ACTOR_FEATURE_SIZE,
    ENTITY_FEATURE_SIZE,
    ENTITY_GLOBAL_FEATURE_SIZE,
    MAP_FEATURE_SIZE,
    PREPARED_SPELL_FEATURE_SIZE,
    INVENTORY_ITEM_FEATURE_SIZE,
)
from agents.ppo_model import (
    DEFAULT_MOVE_COUNT,
    DEFAULT_OPTION_COUNT,
    DEFAULT_TARGET_COUNT,
    MASKED_LOGIT_VALUE,
)


DEFAULT_BONUS_ACTION_TYPE_COUNT = 8
DEFAULT_REACTION_TYPE_COUNT = 8
DEFAULT_CLASS_FEATURE_COUNT = 8
DEFAULT_SPELL_COUNT = max(1, PREPARED_SPELL_FEATURE_SIZE)
DEFAULT_SLOT_LEVEL_COUNT = 4
DEFAULT_ITEM_COUNT = max(1, INVENTORY_ITEM_FEATURE_SIZE)
DEFAULT_GNN_POLICY_HIDDEN_SIZE = 128


HEAD_ORDER: tuple[str, ...] = (
    "action_category",
    "main_action_type",
    "bonus_action_type",
    "reaction_type",
    "class_feature",
    "target_index",
    "move_index",
    "spell_index",
    "slot_level",
    "item_index",
    "option_index",
)


class GNNPPOActorCritic(nn.Module):
    """Actor-critic model that encodes combat entities with message passing."""

    def __init__(
        self,
        *,
        actor_feature_size: int = ACTOR_FEATURE_SIZE,
        entity_feature_size: int = ENTITY_FEATURE_SIZE,
        map_feature_size: int = MAP_FEATURE_SIZE,
        global_feature_size: int = ENTITY_GLOBAL_FEATURE_SIZE,
        target_count: int = DEFAULT_TARGET_COUNT,
        move_count: int = DEFAULT_MOVE_COUNT,
        option_count: int = DEFAULT_OPTION_COUNT,
        action_category_count: int = ACTION_CATEGORY_COUNT,
        main_action_type_count: int = MAIN_ACTION_TYPE_COUNT,
        bonus_action_type_count: int = DEFAULT_BONUS_ACTION_TYPE_COUNT,
        reaction_type_count: int = DEFAULT_REACTION_TYPE_COUNT,
        class_feature_count: int = DEFAULT_CLASS_FEATURE_COUNT,
        spell_count: int = DEFAULT_SPELL_COUNT,
        slot_level_count: int = DEFAULT_SLOT_LEVEL_COUNT,
        item_count: int = DEFAULT_ITEM_COUNT,
        gnn_hidden_size: int = 64,
        policy_hidden_size: int = DEFAULT_GNN_POLICY_HIDDEN_SIZE,
        message_passing_steps: int = 2,
        context_hidden_sizes: Sequence[int] = (128,),
    ) -> None:
        super().__init__()
        _validate_positive(
            actor_feature_size=actor_feature_size,
            entity_feature_size=entity_feature_size,
            map_feature_size=map_feature_size,
            global_feature_size=global_feature_size,
            target_count=target_count,
            move_count=move_count,
            option_count=option_count,
            action_category_count=action_category_count,
            main_action_type_count=main_action_type_count,
            bonus_action_type_count=bonus_action_type_count,
            reaction_type_count=reaction_type_count,
            class_feature_count=class_feature_count,
            spell_count=spell_count,
            slot_level_count=slot_level_count,
            item_count=item_count,
            gnn_hidden_size=gnn_hidden_size,
            policy_hidden_size=policy_hidden_size,
        )

        self.actor_feature_size = int(actor_feature_size)
        self.entity_feature_size = int(entity_feature_size)
        self.map_feature_size = int(map_feature_size)
        self.global_feature_size = int(global_feature_size)
        self.target_count = int(target_count)
        self.move_count = int(move_count)
        self.option_count = int(option_count)
        self.action_category_count = int(action_category_count)
        self.main_action_type_count = int(main_action_type_count)
        self.bonus_action_type_count = int(bonus_action_type_count)
        self.reaction_type_count = int(reaction_type_count)
        self.class_feature_count = int(class_feature_count)
        self.spell_count = int(spell_count)
        self.slot_level_count = int(slot_level_count)
        self.item_count = int(item_count)
        self.gnn_hidden_size = int(gnn_hidden_size)
        self.policy_hidden_size = int(policy_hidden_size)

        self.gnn_encoder = CombatGNNEncoder(
            node_feature_size=self.entity_feature_size,
            actor_feature_size=self.actor_feature_size,
            hidden_size=self.gnn_hidden_size,
            message_passing_steps=message_passing_steps,
        )
        self.map_projection = nn.Sequential(
            nn.Linear(self.map_feature_size, self.policy_hidden_size),
            nn.ReLU(),
        )
        self.global_projection = nn.Sequential(
            nn.Linear(self.global_feature_size, self.policy_hidden_size),
            nn.ReLU(),
        )

        context_input_size = self.gnn_hidden_size * 4 + self.policy_hidden_size * 2
        self.context_encoder = _build_mlp(
            context_input_size,
            (*tuple(context_hidden_sizes), self.policy_hidden_size),
        )

        self.action_category_head = nn.Linear(
            self.policy_hidden_size,
            self.action_category_count,
        )
        self.main_action_type_head = nn.Linear(
            self.policy_hidden_size,
            self.main_action_type_count,
        )
        self.bonus_action_type_head = nn.Linear(
            self.policy_hidden_size,
            self.bonus_action_type_count,
        )
        self.reaction_type_head = nn.Linear(
            self.policy_hidden_size,
            self.reaction_type_count,
        )
        self.class_feature_head = nn.Linear(
            self.policy_hidden_size,
            self.class_feature_count,
        )
        self.target_head = nn.Linear(self.policy_hidden_size, self.target_count)
        self.move_head = nn.Linear(self.policy_hidden_size, self.move_count)
        self.spell_head = nn.Linear(self.policy_hidden_size, self.spell_count)
        self.slot_level_head = nn.Linear(self.policy_hidden_size, self.slot_level_count)
        self.item_head = nn.Linear(self.policy_hidden_size, self.item_count)
        self.option_head = nn.Linear(self.policy_hidden_size, self.option_count)
        self.value_head = nn.Linear(self.gnn_hidden_size, 1)

    def forward(
        self,
        observation: EntityObservation | Mapping[str, torch.Tensor],
        critic_observation: EntityObservation | Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return raw policy logits and value estimates."""

        prepared = _prepare_observation(
            observation,
            self.map_feature_size,
            self.global_feature_size,
        )
        device = next(self.parameters()).device
        prepared = {
            key: value.to(device=device, dtype=torch.float32)
            for key, value in prepared.items()
        }
        gnn_output = self.gnn_encoder(
            actor_features=prepared["actor_features"],
            entities_features=prepared["entities_features"],
            entity_mask=prepared["entity_mask"],
        )
        actor_embedding = _ensure_2d(gnn_output.actor_embedding)
        allies_embedding = _ensure_2d(gnn_output.pooled_allies_embedding)
        enemies_embedding = _ensure_2d(gnn_output.pooled_enemies_embedding)
        battle_embedding = _ensure_2d(gnn_output.pooled_battle_embedding)
        map_embedding = self.map_projection(prepared["map_features"])
        global_embedding = self.global_projection(prepared["global_features"])
        context = self.context_encoder(
            torch.cat(
                (
                    actor_embedding,
                    allies_embedding,
                    enemies_embedding,
                    battle_embedding,
                    map_embedding,
                    global_embedding,
                ),
                dim=-1,
            )
        )
        critic_battle_embedding = self._critic_battle_embedding(
            prepared if critic_observation is None else critic_observation,
        )
        return {
            "action_category_logits": self.action_category_head(context),
            "main_action_type_logits": self.main_action_type_head(context),
            "bonus_action_type_logits": self.bonus_action_type_head(context),
            "reaction_type_logits": self.reaction_type_head(context),
            "class_feature_logits": self.class_feature_head(context),
            "target_logits": self.target_head(context),
            "move_logits": self.move_head(context),
            "spell_logits": self.spell_head(context),
            "slot_level_logits": self.slot_level_head(context),
            "item_logits": self.item_head(context),
            "option_logits": self.option_head(context),
            "value": self.value_head(critic_battle_embedding).squeeze(-1),
        }

    def act(
        self,
        observation: EntityObservation | Mapping[str, torch.Tensor],
        masks: Mapping[str, torch.Tensor] | None = None,
        deterministic: bool = False,
        critic_observation: EntityObservation | Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample or greedily select all hierarchical action heads under masks."""

        single = _is_single_observation(observation)
        outputs = self.forward(observation, critic_observation=critic_observation)
        batch_size = int(outputs["value"].shape[0])
        distributions = self._build_distributions(outputs, masks or {}, batch_size)
        selected_actions = {
            action_key: _select_from_distribution(
                _distribution(distributions, action_key),
                deterministic,
            )
            for action_key in HEAD_ORDER
        }
        log_prob, entropy = _combine_log_probs(distributions, selected_actions)
        result = {
            **selected_actions,
            "log_prob": log_prob,
            "entropy": entropy,
            "value": outputs["value"],
        }
        if single:
            return {key: value.squeeze(0) for key, value in result.items()}
        return result

    def evaluate_actions(
        self,
        observation: EntityObservation | Mapping[str, torch.Tensor],
        actions: Mapping[str, torch.Tensor | int],
        masks: Mapping[str, torch.Tensor] | None = None,
        critic_observation: EntityObservation | Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Evaluate selected hierarchical actions for PPO loss calculation."""

        outputs = self.forward(observation, critic_observation=critic_observation)
        batch_size = int(outputs["value"].shape[0])
        distributions = self._build_distributions(outputs, masks or {}, batch_size)
        selected_actions = {
            action_key: _action_tensor(
                actions,
                action_key,
                batch_size,
                outputs["value"].device,
                default=0 if action_key != "action_category" else None,
            )
            for action_key in HEAD_ORDER
        }
        self._validate_selected_actions(distributions["masks"], selected_actions)
        log_prob, entropy = _combine_log_probs(distributions, selected_actions)
        return {
            "log_prob": log_prob,
            "entropy": entropy,
            "value": outputs["value"],
        }

    def _critic_battle_embedding(
        self,
        critic_observation: EntityObservation | Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        if _is_prepared_observation(critic_observation):
            prepared = dict(critic_observation)
        else:
            prepared = _prepare_observation(
                critic_observation,
                self.map_feature_size,
                self.global_feature_size,
            )
        device = next(self.parameters()).device
        prepared = {
            key: value.to(device=device, dtype=torch.float32)
            for key, value in prepared.items()
        }
        critic_output = self.gnn_encoder(
            actor_features=prepared["actor_features"],
            entities_features=prepared["entities_features"],
            entity_mask=prepared["entity_mask"],
        )
        return _ensure_2d(critic_output.pooled_battle_embedding)

    def _build_distributions(
        self,
        outputs: Mapping[str, torch.Tensor],
        masks: Mapping[str, torch.Tensor],
        batch_size: int,
    ) -> dict[str, Categorical | dict[str, torch.Tensor]]:
        device = outputs["value"].device
        prepared_masks = {
            "action_category": _prepare_mask(
                masks.get("action_category"),
                batch_size,
                self.action_category_count,
                device,
                "action_category",
                allow_empty=False,
            ),
            "main_action_type": _prepare_mask(
                masks.get("main_action_type"),
                batch_size,
                self.main_action_type_count,
                device,
                "main_action_type",
                allow_empty=True,
            ),
            "bonus_action_type": _prepare_mask(
                masks.get("bonus_action_type"),
                batch_size,
                self.bonus_action_type_count,
                device,
                "bonus_action_type",
                allow_empty=True,
            ),
            "reaction_type": _prepare_mask(
                masks.get("reaction_type"),
                batch_size,
                self.reaction_type_count,
                device,
                "reaction_type",
                allow_empty=True,
            ),
            "class_feature": _prepare_mask(
                masks.get("class_feature"),
                batch_size,
                self.class_feature_count,
                device,
                "class_feature",
                allow_empty=True,
            ),
            "target_index": _prepare_mask(
                masks.get("target_index"),
                batch_size,
                self.target_count,
                device,
                "target_index",
                allow_empty=True,
            ),
            "move_index": _prepare_mask(
                masks.get("move_index"),
                batch_size,
                self.move_count,
                device,
                "move_index",
                allow_empty=True,
            ),
            "spell_index": _prepare_mask(
                masks.get("spell_index"),
                batch_size,
                self.spell_count,
                device,
                "spell_index",
                allow_empty=True,
            ),
            "slot_level": _prepare_mask(
                masks.get("slot_level"),
                batch_size,
                self.slot_level_count,
                device,
                "slot_level",
                allow_empty=True,
            ),
            "item_index": _prepare_mask(
                masks.get("item_index"),
                batch_size,
                self.item_count,
                device,
                "item_index",
                allow_empty=True,
            ),
            "option_index": _prepare_mask(
                masks.get("option_index"),
                batch_size,
                self.option_count,
                device,
                "option_index",
                allow_empty=True,
            ),
        }
        return {
            "action_category": Categorical(
                logits=_mask_logits(
                    outputs["action_category_logits"],
                    prepared_masks["action_category"],
                ),
                validate_args=False,
            ),
            "main_action_type": Categorical(
                logits=_mask_logits(
                    outputs["main_action_type_logits"],
                    prepared_masks["main_action_type"],
                ),
                validate_args=False,
            ),
            "bonus_action_type": Categorical(
                logits=_mask_logits(
                    outputs["bonus_action_type_logits"],
                    prepared_masks["bonus_action_type"],
                ),
                validate_args=False,
            ),
            "reaction_type": Categorical(
                logits=_mask_logits(
                    outputs["reaction_type_logits"],
                    prepared_masks["reaction_type"],
                ),
                validate_args=False,
            ),
            "class_feature": Categorical(
                logits=_mask_logits(
                    outputs["class_feature_logits"],
                    prepared_masks["class_feature"],
                ),
                validate_args=False,
            ),
            "target_index": Categorical(
                logits=_mask_logits(outputs["target_logits"], prepared_masks["target_index"]),
                validate_args=False,
            ),
            "move_index": Categorical(
                logits=_mask_logits(outputs["move_logits"], prepared_masks["move_index"]),
                validate_args=False,
            ),
            "spell_index": Categorical(
                logits=_mask_logits(outputs["spell_logits"], prepared_masks["spell_index"]),
                validate_args=False,
            ),
            "slot_level": Categorical(
                logits=_mask_logits(
                    outputs["slot_level_logits"],
                    prepared_masks["slot_level"],
                ),
                validate_args=False,
            ),
            "item_index": Categorical(
                logits=_mask_logits(outputs["item_logits"], prepared_masks["item_index"]),
                validate_args=False,
            ),
            "option_index": Categorical(
                logits=_mask_logits(outputs["option_logits"], prepared_masks["option_index"]),
                validate_args=False,
            ),
            "masks": prepared_masks,
        }

    @staticmethod
    def _validate_selected_actions(
        masks: Mapping[str, torch.Tensor],
        actions: Mapping[str, torch.Tensor],
    ) -> None:
        batch_indices = torch.arange(
            next(iter(actions.values())).shape[0],
            device=next(iter(actions.values())).device,
        )
        for action_key in HEAD_ORDER:
            action = actions[action_key]
            mask = masks[action_key]
            if (action >= mask.shape[1]).any():
                raise ValueError(f"actions contain an out-of-range {action_key}")
            if not mask[batch_indices, action].all():
                raise ValueError(f"actions contain a masked {action_key}")


def _prepare_observation(
    observation: EntityObservation | Mapping[str, torch.Tensor],
    map_feature_size: int,
    global_feature_size: int,
) -> dict[str, torch.Tensor]:
    if isinstance(observation, EntityObservation):
        actor_features = observation.actor_features
        entities_features = observation.entities_features
        entity_mask = observation.entity_mask
        map_features = observation.map_features
        global_features = observation.global_features
    else:
        actor_features = observation["actor_features"]
        entities_features = observation["entities_features"]
        entity_mask = observation["entity_mask"]
        map_features = observation["map_features"]
        global_features = observation["global_features"]

    actor_features = _ensure_2d(actor_features)
    if entities_features.ndim == 2:
        entities_features = entities_features.unsqueeze(0)
    elif entities_features.ndim != 3:
        raise ValueError("entities_features must be a 2D or 3D tensor")
    entity_mask = _ensure_2d(entity_mask)
    batch_size = actor_features.shape[0]
    entities_features = _expand_batch(entities_features, batch_size, "entities_features")
    entity_mask = _expand_batch(entity_mask, batch_size, "entity_mask")
    map_features = _fit_feature_size(
        _flatten_map_features(map_features),
        map_feature_size,
        "map_features",
    )
    map_features = _expand_batch(map_features, batch_size, "map_features")
    global_features = _fit_feature_size(
        _ensure_2d(global_features),
        global_feature_size,
        "global_features",
    )
    global_features = _expand_batch(global_features, batch_size, "global_features")
    return {
        "actor_features": actor_features.to(dtype=torch.float32),
        "entities_features": entities_features.to(dtype=torch.float32),
        "entity_mask": entity_mask.to(dtype=torch.float32),
        "map_features": map_features.to(dtype=torch.float32),
        "global_features": global_features.to(dtype=torch.float32),
    }


def _is_prepared_observation(observation: object) -> bool:
    if not isinstance(observation, Mapping):
        return False
    required_keys = {
        "actor_features",
        "entities_features",
        "entity_mask",
        "map_features",
        "global_features",
    }
    if not required_keys.issubset(observation):
        return False
    map_features = observation["map_features"]
    global_features = observation["global_features"]
    return (
        torch.is_tensor(map_features)
        and torch.is_tensor(global_features)
        and map_features.ndim == 2
        and global_features.ndim == 2
    )


def _is_single_observation(
    observation: EntityObservation | Mapping[str, torch.Tensor],
) -> bool:
    actor_features = (
        observation.actor_features
        if isinstance(observation, EntityObservation)
        else observation["actor_features"]
    )
    return actor_features.ndim == 1


def _flatten_map_features(map_features: torch.Tensor) -> torch.Tensor:
    if map_features.ndim == 2:
        return map_features.reshape(1, -1)
    if map_features.ndim == 3:
        return map_features.reshape(map_features.shape[0], -1)
    if map_features.ndim == 1:
        return map_features.unsqueeze(0)
    raise ValueError("map_features must be a 1D, 2D or 3D tensor")


def _fit_feature_size(
    features: torch.Tensor,
    feature_size: int,
    name: str,
) -> torch.Tensor:
    if features.shape[-1] == feature_size:
        return features
    if features.shape[-1] > feature_size:
        return features[..., :feature_size]
    padding = torch.zeros(
        (*features.shape[:-1], feature_size - features.shape[-1]),
        dtype=features.dtype,
        device=features.device,
    )
    return torch.cat((features, padding), dim=-1)


def _expand_batch(tensor: torch.Tensor, batch_size: int, name: str) -> torch.Tensor:
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] == 1 and batch_size > 1:
        return tensor.expand(batch_size, *tensor.shape[1:])
    raise ValueError(f"{name} batch size {tensor.shape[0]} does not match {batch_size}")


def _ensure_2d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    if tensor.ndim == 2:
        return tensor
    raise ValueError("tensor must be 1D or 2D")


def _build_mlp(input_size: int, hidden_sizes: Sequence[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_size = input_size
    for hidden_size in hidden_sizes:
        if hidden_size <= 0:
            raise ValueError("hidden sizes must be greater than zero")
        layers.append(nn.Linear(current_size, int(hidden_size)))
        layers.append(nn.ReLU())
        current_size = int(hidden_size)
    return nn.Sequential(*layers)


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
            raise ValueError(f"{name} mask size {prepared.shape[1]} exceeds head size")
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


def _combine_log_probs(
    distributions: Mapping[str, Categorical | dict[str, torch.Tensor]],
    actions: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    first_action = next(iter(actions.values()))
    log_prob = torch.zeros_like(first_action, dtype=torch.float32)
    entropy = torch.zeros_like(first_action, dtype=torch.float32)
    for action_key in HEAD_ORDER:
        distribution = _distribution(distributions, action_key)
        log_prob = log_prob + distribution.log_prob(actions[action_key])
        entropy = entropy + distribution.entropy()
    return log_prob, entropy


def _distribution(
    distributions: Mapping[str, Categorical | dict[str, torch.Tensor]],
    action_key: str,
) -> Categorical:
    distribution = distributions[action_key]
    if not isinstance(distribution, Categorical):
        raise TypeError(f"{action_key} distribution is missing")
    return distribution


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
        value = torch.as_tensor(actions[key], dtype=torch.long, device=device)
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


def _validate_positive(**values: int) -> None:
    for name, value in values.items():
        if int(value) <= 0:
            raise ValueError(f"{name} must be greater than zero")


__all__ = [
    "DEFAULT_BONUS_ACTION_TYPE_COUNT",
    "DEFAULT_CLASS_FEATURE_COUNT",
    "DEFAULT_GNN_POLICY_HIDDEN_SIZE",
    "DEFAULT_ITEM_COUNT",
    "DEFAULT_REACTION_TYPE_COUNT",
    "DEFAULT_SLOT_LEVEL_COUNT",
    "DEFAULT_SPELL_COUNT",
    "GNNPPOActorCritic",
    "HEAD_ORDER",
]
