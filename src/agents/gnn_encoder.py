"""Simple mask-aware GNN encoder for entity-based combat observations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from agents.entity_observation import EntityObservation
from agents.observation import (
    ACTOR_COMMON_ACTION_OFFSET,
    ACTOR_DAMAGE_ACTION_OFFSET,
    ACTOR_FEATURE_SIZE,
    ACTOR_REAL_GAME_OFFSET,
    BASE_CHARACTER_FEATURE_SIZE,
    ENTITY_FEATURE_SIZE,
    ENTITY_MASK_SIZE,
    GNN_NODE_FEATURE_SIZE,
    OTHER_CHARACTER_FEATURE_SIZE,
    OTHER_COMMON_ACTION_OFFSET,
    OTHER_DAMAGE_PROFILE_OFFSET,
    OTHER_ENTITY_PROFILE_OFFSET,
    OTHER_MAP_FEATURE_OFFSET,
)


EDGE_FEATURE_SIZE = 7
DEFAULT_GNN_HIDDEN_SIZE = 64
DEFAULT_MESSAGE_PASSING_STEPS = 2

BASE_X_INDEX = 5
BASE_Y_INDEX = 6
TEAM_PLAYERS_INDEX = 7
TEAM_ENEMIES_INDEX = 8
HAS_MELEE_ATTACK_INDEX = 9
HAS_RANGED_ATTACK_INDEX = 10
ALIVE_INDEX = 11
HP_RATIO_INDEX = 3
AC_INDEX = 4

ACTOR_PRONE_INDEX = ACTOR_COMMON_ACTION_OFFSET + 1
ACTOR_GRAPPLED_INDEX = ACTOR_COMMON_ACTION_OFFSET + 2
ACTOR_HIDDEN_INDEX = ACTOR_COMMON_ACTION_OFFSET + 3
ACTOR_DODGING_INDEX = ACTOR_COMMON_ACTION_OFFSET + 4
ACTOR_HAS_SPELLS_INDEX = ACTOR_COMMON_ACTION_OFFSET + 8
ACTOR_CLASS_ID_INDEX = ACTOR_REAL_GAME_OFFSET + 2
ACTOR_SUBCLASS_ID_INDEX = ACTOR_REAL_GAME_OFFSET + 3
ACTOR_ROLE_ID_INDEX = ACTOR_REAL_GAME_OFFSET + 5

OTHER_PRONE_INDEX = OTHER_COMMON_ACTION_OFFSET
OTHER_GRAPPLED_INDEX = OTHER_COMMON_ACTION_OFFSET + 1
OTHER_HIDDEN_INDEX = OTHER_COMMON_ACTION_OFFSET + 2
OTHER_DODGING_INDEX = OTHER_COMMON_ACTION_OFFSET + 3
OTHER_CAN_BE_ATTACKED_INDEX = OTHER_COMMON_ACTION_OFFSET + 5
OTHER_CAN_BE_HELPED_AGAINST_INDEX = OTHER_COMMON_ACTION_OFFSET + 6
OTHER_LINE_OF_SIGHT_INDEX = OTHER_MAP_FEATURE_OFFSET
OTHER_COVER_INDEX = OTHER_MAP_FEATURE_OFFSET + 1
OTHER_DISTANCE_INDEX = OTHER_MAP_FEATURE_OFFSET + 2

ENTITY_EXTRA_IS_SPELLCASTER_INDEX = ENTITY_FEATURE_SIZE - 1


@dataclass(frozen=True)
class GNNEncoderOutput:
    """Embeddings produced by the combat GNN encoder."""

    actor_embedding: torch.Tensor
    pooled_allies_embedding: torch.Tensor
    pooled_enemies_embedding: torch.Tensor
    pooled_battle_embedding: torch.Tensor


class CombatGNNEncoder(nn.Module):
    """A small dense message-passing encoder without external GNN libraries."""

    def __init__(
        self,
        *,
        node_feature_size: int = GNN_NODE_FEATURE_SIZE,
        actor_feature_size: int = ACTOR_FEATURE_SIZE,
        edge_feature_size: int = EDGE_FEATURE_SIZE,
        hidden_size: int = DEFAULT_GNN_HIDDEN_SIZE,
        message_passing_steps: int = DEFAULT_MESSAGE_PASSING_STEPS,
    ) -> None:
        super().__init__()
        if node_feature_size <= 0:
            raise ValueError("node_feature_size must be greater than zero")
        if actor_feature_size <= 0:
            raise ValueError("actor_feature_size must be greater than zero")
        if edge_feature_size <= 0:
            raise ValueError("edge_feature_size must be greater than zero")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be greater than zero")
        if message_passing_steps < 0:
            raise ValueError("message_passing_steps cannot be negative")

        self.node_feature_size = int(node_feature_size)
        self.actor_feature_size = int(actor_feature_size)
        self.edge_feature_size = int(edge_feature_size)
        self.hidden_size = int(hidden_size)
        self.message_passing_steps = int(message_passing_steps)

        self.node_projection = nn.Sequential(
            nn.Linear(self.node_feature_size, self.hidden_size),
            nn.ReLU(),
        )
        self.edge_projection = nn.Sequential(
            nn.Linear(self.edge_feature_size, self.hidden_size),
            nn.ReLU(),
        )
        self.message_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
        )

    def forward(
        self,
        observation: EntityObservation | None = None,
        *,
        actor_features: torch.Tensor | None = None,
        entities_features: torch.Tensor | None = None,
        entity_mask: torch.Tensor | None = None,
    ) -> GNNEncoderOutput:
        """Encode one or more entity observations into pooled battle embeddings."""

        if observation is not None:
            actor_features = observation.actor_features
            entities_features = observation.entities_features
            entity_mask = observation.entity_mask
        if actor_features is None or entities_features is None or entity_mask is None:
            raise ValueError(
                "Pass either observation or actor_features/entities_features/entity_mask."
            )

        actor_features, entities_features, entity_mask, single = _prepare_inputs(
            actor_features,
            entities_features,
            entity_mask,
        )
        device = next(self.parameters()).device
        actor_features = actor_features.to(device=device, dtype=torch.float32)
        entities_features = entities_features.to(device=device, dtype=torch.float32)
        entity_mask = entity_mask.to(device=device, dtype=torch.float32)

        node_features = self._build_node_features(actor_features, entities_features)
        node_mask = torch.cat(
            (
                torch.ones(
                    (entity_mask.shape[0], 1),
                    dtype=torch.float32,
                    device=device,
                ),
                entity_mask,
            ),
            dim=1,
        )
        node_features = node_features * node_mask.unsqueeze(-1)

        node_states = self.node_projection(node_features) * node_mask.unsqueeze(-1)
        edge_features = build_edge_features(node_features, node_mask)
        edge_states = self.edge_projection(edge_features)
        pair_mask = _pair_mask(node_mask)

        for _ in range(self.message_passing_steps):
            node_states = self._message_passing_step(
                node_states,
                edge_states,
                pair_mask,
                node_mask,
            )

        actor_embedding = node_states[:, 0, :]
        entity_states = node_states[:, 1:, :]
        allies_mask, enemies_mask = _ally_enemy_masks(node_features, entity_mask)
        pooled_allies = _masked_mean(entity_states, allies_mask)
        pooled_enemies = _masked_mean(entity_states, enemies_mask)
        pooled_battle = _masked_mean(node_states, node_mask)

        if single:
            return GNNEncoderOutput(
                actor_embedding=actor_embedding.squeeze(0),
                pooled_allies_embedding=pooled_allies.squeeze(0),
                pooled_enemies_embedding=pooled_enemies.squeeze(0),
                pooled_battle_embedding=pooled_battle.squeeze(0),
            )
        return GNNEncoderOutput(
            actor_embedding=actor_embedding,
            pooled_allies_embedding=pooled_allies,
            pooled_enemies_embedding=pooled_enemies,
            pooled_battle_embedding=pooled_battle,
        )

    def _build_node_features(
        self,
        actor_features: torch.Tensor,
        entities_features: torch.Tensor,
    ) -> torch.Tensor:
        actor_node = actor_features_to_entity_node(actor_features)
        nodes = torch.cat((actor_node.unsqueeze(1), entities_features), dim=1)
        if nodes.shape[-1] == self.node_feature_size:
            return nodes
        if nodes.shape[-1] > self.node_feature_size:
            return nodes[..., : self.node_feature_size]
        padding = torch.zeros(
            (*nodes.shape[:-1], self.node_feature_size - nodes.shape[-1]),
            dtype=nodes.dtype,
            device=nodes.device,
        )
        return torch.cat((nodes, padding), dim=-1)

    def _message_passing_step(
        self,
        node_states: torch.Tensor,
        edge_states: torch.Tensor,
        pair_mask: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, node_count, _ = node_states.shape
        source_states = node_states.unsqueeze(2).expand(
            batch_size,
            node_count,
            node_count,
            self.hidden_size,
        )
        target_states = node_states.unsqueeze(1).expand(
            batch_size,
            node_count,
            node_count,
            self.hidden_size,
        )
        messages = self.message_mlp(
            torch.cat((source_states, target_states, edge_states), dim=-1)
        )
        messages = messages * pair_mask.unsqueeze(-1)
        message_counts = pair_mask.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
        aggregated = messages.sum(dim=1) / message_counts
        updated = self.update_mlp(torch.cat((node_states, aggregated), dim=-1))
        return updated * node_mask.unsqueeze(-1)


def actor_features_to_entity_node(actor_features: torch.Tensor) -> torch.Tensor:
    """Project actor-specific features into the entity node feature schema."""

    if actor_features.ndim == 1:
        actor_features = actor_features.unsqueeze(0)
        squeeze = True
    elif actor_features.ndim == 2:
        squeeze = False
    else:
        raise ValueError("actor_features must be a 1D or 2D tensor")

    batch_size = actor_features.shape[0]
    actor_node = torch.zeros(
        (batch_size, ENTITY_FEATURE_SIZE),
        dtype=actor_features.dtype,
        device=actor_features.device,
    )
    actor_node[:, :BASE_CHARACTER_FEATURE_SIZE] = actor_features[
        :,
        :BASE_CHARACTER_FEATURE_SIZE,
    ]
    actor_node[:, OTHER_PRONE_INDEX] = actor_features[:, ACTOR_PRONE_INDEX]
    actor_node[:, OTHER_GRAPPLED_INDEX] = actor_features[:, ACTOR_GRAPPLED_INDEX]
    actor_node[:, OTHER_HIDDEN_INDEX] = actor_features[:, ACTOR_HIDDEN_INDEX]
    actor_node[:, OTHER_DODGING_INDEX] = actor_features[:, ACTOR_DODGING_INDEX]
    damage_profile_width = OTHER_ENTITY_PROFILE_OFFSET - OTHER_DAMAGE_PROFILE_OFFSET
    actor_damage_source = actor_features[:, ACTOR_DAMAGE_ACTION_OFFSET:]
    copied_damage_width = min(damage_profile_width, actor_damage_source.shape[1])
    if copied_damage_width > 0:
        actor_node[
            :,
            OTHER_DAMAGE_PROFILE_OFFSET:OTHER_DAMAGE_PROFILE_OFFSET + copied_damage_width,
        ] = actor_damage_source[:, :copied_damage_width].clamp(0.0, 1.0)
    actor_node[:, OTHER_ENTITY_PROFILE_OFFSET] = actor_features[:, ACTOR_CLASS_ID_INDEX]
    actor_node[:, OTHER_ENTITY_PROFILE_OFFSET + 1] = actor_features[
        :,
        ACTOR_SUBCLASS_ID_INDEX,
    ]
    actor_node[:, OTHER_ENTITY_PROFILE_OFFSET + 2] = actor_features[:, ACTOR_ROLE_ID_INDEX]
    actor_node[:, OTHER_ENTITY_PROFILE_OFFSET + 13] = actor_features[:, AC_INDEX]
    actor_node[:, OTHER_ENTITY_PROFILE_OFFSET + 14] = actor_features[:, HP_RATIO_INDEX]
    actor_node[:, OTHER_MAP_FEATURE_OFFSET] = 1.0
    actor_node[:, OTHER_MAP_FEATURE_OFFSET + 1] = 0.0
    actor_node[:, OTHER_MAP_FEATURE_OFFSET + 2] = 0.0
    actor_node[:, OTHER_MAP_FEATURE_OFFSET + 3] = 1.0
    actor_node[:, ENTITY_EXTRA_IS_SPELLCASTER_INDEX] = actor_features[
        :,
        ACTOR_HAS_SPELLS_INDEX,
    ]
    if squeeze:
        return actor_node.squeeze(0)
    return actor_node


def build_edge_features(
    node_features: torch.Tensor,
    node_mask: torch.Tensor,
) -> torch.Tensor:
    """Build dense directed edge features from node features and masks."""

    if node_features.ndim != 3:
        raise ValueError("node_features must have shape [batch, nodes, features]")
    if node_mask.ndim != 2:
        raise ValueError("node_mask must have shape [batch, nodes]")

    source = node_features.unsqueeze(2)
    target = node_features.unsqueeze(1)
    distance = (
        (source[..., BASE_X_INDEX] - target[..., BASE_X_INDEX]).abs()
        + (source[..., BASE_Y_INDEX] - target[..., BASE_Y_INDEX]).abs()
    )
    same_team = _same_team(source, target)
    enemy_relation = _enemy_relation(source, target)
    can_attack = _can_attack_edge(source, target, distance, enemy_relation)
    can_help = _can_help_edge(source, target, same_team, enemy_relation)
    line_of_sight = _line_of_sight_edge(source, target)
    cover_between = _cover_edge(source, target)

    edges = torch.stack(
        (
            distance,
            same_team,
            enemy_relation,
            can_attack,
            can_help,
            line_of_sight,
            cover_between,
        ),
        dim=-1,
    )
    return edges * _pair_mask(node_mask, include_self=True).unsqueeze(-1)


def _prepare_inputs(
    actor_features: torch.Tensor,
    entities_features: torch.Tensor,
    entity_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    if actor_features.ndim == 1:
        actor_features = actor_features.unsqueeze(0)
        single = True
    elif actor_features.ndim == 2:
        single = False
    else:
        raise ValueError("actor_features must be a 1D or 2D tensor")

    if entities_features.ndim == 2:
        entities_features = entities_features.unsqueeze(0)
    elif entities_features.ndim != 3:
        raise ValueError("entities_features must be a 2D or 3D tensor")
    if entity_mask.ndim == 1:
        entity_mask = entity_mask.unsqueeze(0)
    elif entity_mask.ndim != 2:
        raise ValueError("entity_mask must be a 1D or 2D tensor")

    batch_size = actor_features.shape[0]
    if entities_features.shape[0] == 1 and batch_size > 1:
        entities_features = entities_features.expand(batch_size, -1, -1)
    if entity_mask.shape[0] == 1 and batch_size > 1:
        entity_mask = entity_mask.expand(batch_size, -1)
    if entities_features.shape[0] != batch_size or entity_mask.shape[0] != batch_size:
        raise ValueError("actor, entity and mask batch sizes must match")
    if entity_mask.shape[1] != entities_features.shape[1]:
        raise ValueError("entity_mask length must match entities_features node count")
    return actor_features, entities_features, entity_mask, single


def _pair_mask(node_mask: torch.Tensor, include_self: bool = False) -> torch.Tensor:
    pair_mask = node_mask.unsqueeze(2) * node_mask.unsqueeze(1)
    if include_self:
        return pair_mask
    node_count = node_mask.shape[1]
    eye = torch.eye(node_count, dtype=pair_mask.dtype, device=node_mask.device).unsqueeze(0)
    return pair_mask * (1.0 - eye)


def _same_team(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    both_players = source[..., TEAM_PLAYERS_INDEX] * target[..., TEAM_PLAYERS_INDEX]
    both_enemies = source[..., TEAM_ENEMIES_INDEX] * target[..., TEAM_ENEMIES_INDEX]
    return ((both_players + both_enemies) > 0).to(dtype=source.dtype)


def _enemy_relation(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    player_to_enemy = source[..., TEAM_PLAYERS_INDEX] * target[..., TEAM_ENEMIES_INDEX]
    enemy_to_player = source[..., TEAM_ENEMIES_INDEX] * target[..., TEAM_PLAYERS_INDEX]
    return ((player_to_enemy + enemy_to_player) > 0).to(dtype=source.dtype)


def _can_attack_edge(
    source: torch.Tensor,
    target: torch.Tensor,
    distance: torch.Tensor,
    enemy_relation: torch.Tensor,
) -> torch.Tensor:
    melee_reach = source[..., HAS_MELEE_ATTACK_INDEX] * (distance <= 1).to(source.dtype)
    ranged_reach = source[..., HAS_RANGED_ATTACK_INDEX] * (distance <= 6).to(source.dtype)
    target_alive = target[..., ALIVE_INDEX]
    return (
        enemy_relation
        * target_alive
        * ((melee_reach + ranged_reach) > 0).to(source.dtype)
        * _line_of_sight_edge(source, target)
    )


def _can_help_edge(
    source: torch.Tensor,
    target: torch.Tensor,
    same_team: torch.Tensor,
    enemy_relation: torch.Tensor,
) -> torch.Tensor:
    help_ally = same_team * target[..., ALIVE_INDEX]
    help_against_enemy = (
        enemy_relation
        * target[..., ALIVE_INDEX]
        * target[..., OTHER_CAN_BE_HELPED_AGAINST_INDEX].clamp(0.0, 1.0)
    )
    return ((help_ally + help_against_enemy) > 0).to(source.dtype)


def _line_of_sight_edge(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    source_to_actor = source[..., OTHER_LINE_OF_SIGHT_INDEX].clamp(0.0, 1.0)
    target_from_actor = target[..., OTHER_LINE_OF_SIGHT_INDEX].clamp(0.0, 1.0)
    actor_source = source[..., OTHER_DISTANCE_INDEX] == 0
    actor_target = target[..., OTHER_DISTANCE_INDEX] == 0
    return torch.where(
        actor_source,
        target_from_actor,
        torch.where(actor_target, source_to_actor, torch.ones_like(source_to_actor)),
    )


def _cover_edge(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    source_cover = source[..., OTHER_COVER_INDEX].clamp(0.0, 3.0)
    target_cover = target[..., OTHER_COVER_INDEX].clamp(0.0, 3.0)
    actor_source = source[..., OTHER_DISTANCE_INDEX] == 0
    actor_target = target[..., OTHER_DISTANCE_INDEX] == 0
    return torch.where(
        actor_source,
        target_cover,
        torch.where(actor_target, source_cover, torch.zeros_like(source_cover)),
    )


def _ally_enemy_masks(
    node_features: torch.Tensor,
    entity_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    actor = node_features[:, :1, :]
    entities = node_features[:, 1:, :]
    same_team = _same_team(actor, entities).squeeze(1)
    enemy_relation = _enemy_relation(actor, entities).squeeze(1)
    allies = entity_mask * same_team
    enemies = entity_mask * enemy_relation
    return allies, enemies


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype).unsqueeze(-1)
    total = (values * weights).sum(dim=1)
    count = weights.sum(dim=1).clamp_min(1.0)
    return total / count


__all__ = [
    "DEFAULT_GNN_HIDDEN_SIZE",
    "DEFAULT_MESSAGE_PASSING_STEPS",
    "EDGE_FEATURE_SIZE",
    "CombatGNNEncoder",
    "GNNEncoderOutput",
    "actor_features_to_entity_node",
    "build_edge_features",
]
