import torch

from agents import (
    CombatGNNEncoder,
    EDGE_FEATURE_SIZE,
    GNN_NODE_FEATURE_SIZE,
    encode_entity_observation,
)
from agents.gnn_encoder import build_edge_features
from combat import (
    Bandit,
    CombatState,
    FighterArcher,
    FighterChampionGreatsword,
    Goblin,
    GridMap,
    Orc,
    Position,
    WizardEvoker,
)


def _state_with_characters(*characters) -> CombatState:
    return CombatState(
        characters=list(characters),
        grid_map=GridMap(width=8, height=8),
    )


def test_gnn_encoder_works_with_different_entity_counts() -> None:
    encoder = CombatGNNEncoder(hidden_size=24, message_passing_steps=1)
    small = encode_entity_observation(
        _state_with_characters(
            FighterArcher(Position(0, 0)),
            Goblin(Position(1, 0)),
        ),
        actor_id=0,
    )
    large = encode_entity_observation(
        _state_with_characters(
            FighterChampionGreatsword(Position(0, 0)),
            WizardEvoker(Position(0, 2)),
            Goblin(Position(1, 0)),
            Orc(Position(5, 5)),
            Bandit(Position(4, 2)),
        ),
        actor_id=0,
    )

    small_output = encoder(small)
    large_output = encoder(large)

    assert small_output.actor_embedding.shape == (24,)
    assert large_output.actor_embedding.shape == (24,)
    assert small_output.pooled_enemies_embedding.shape == (24,)
    assert large_output.pooled_battle_embedding.shape == (24,)


def test_masked_entities_do_not_affect_pooled_embeddings() -> None:
    encoder = CombatGNNEncoder(hidden_size=16, message_passing_steps=2)
    observation = encode_entity_observation(
        _state_with_characters(
            FighterArcher(Position(0, 0)),
            Goblin(Position(1, 0)),
        ),
        actor_id=0,
    )
    polluted = type(observation)(
        actor_features=observation.actor_features.clone(),
        entities_features=observation.entities_features.clone(),
        map_features=observation.map_features.clone(),
        global_features=observation.global_features.clone(),
        entity_mask=observation.entity_mask.clone(),
    )
    polluted.entities_features[polluted.entity_mask == 0] = 1.0e6

    clean_output = encoder(observation)
    polluted_output = encoder(polluted)

    assert torch.allclose(
        clean_output.pooled_allies_embedding,
        polluted_output.pooled_allies_embedding,
        atol=1.0e-5,
    )
    assert torch.allclose(
        clean_output.pooled_enemies_embedding,
        polluted_output.pooled_enemies_embedding,
        atol=1.0e-5,
    )
    assert torch.allclose(
        clean_output.pooled_battle_embedding,
        polluted_output.pooled_battle_embedding,
        atol=1.0e-5,
    )


def test_gnn_output_and_edge_feature_dimensions_are_stable() -> None:
    encoder = CombatGNNEncoder(hidden_size=32)
    observation = encode_entity_observation(
        _state_with_characters(
            FighterArcher(Position(0, 0)),
            WizardEvoker(Position(0, 2)),
            Goblin(Position(1, 0)),
        ),
        actor_id=0,
    )
    node_count = 1 + observation.entities_features.shape[0]

    actor_node = encoder._build_node_features(  # noqa: SLF001 - tests stable tensor contract.
        observation.actor_features.unsqueeze(0),
        observation.entities_features.unsqueeze(0),
    )
    node_mask = torch.cat((torch.ones(1, 1), observation.entity_mask.unsqueeze(0)), dim=1)
    edge_features = build_edge_features(actor_node, node_mask)
    output = encoder(observation)

    assert actor_node.shape == (1, node_count, GNN_NODE_FEATURE_SIZE)
    assert edge_features.shape == (1, node_count, node_count, EDGE_FEATURE_SIZE)
    assert output.actor_embedding.shape == (32,)
    assert output.pooled_allies_embedding.shape == (32,)
    assert output.pooled_enemies_embedding.shape == (32,)
    assert output.pooled_battle_embedding.shape == (32,)
