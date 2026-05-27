import torch

from agents import (
    ACTION_CATEGORY_COUNT,
    MAIN_ACTION_TYPE_COUNT,
    GNNPPOActorCritic,
    HEAD_ORDER,
    ActionCategory,
    encode_entity_observation,
)
from agents.entity_observation import EntityObservation
from combat import (
    CombatState,
    FighterArcher,
    FighterChampionGreatsword,
    Goblin,
    GridMap,
    Orc,
    Position,
    WizardEvoker,
)
from configs import PPOConfig


def _state(*characters) -> CombatState:
    return CombatState(
        characters=list(characters),
        grid_map=GridMap(width=8, height=8),
    )


def _small_model() -> GNNPPOActorCritic:
    return GNNPPOActorCritic(
        target_count=8,
        move_count=64,
        option_count=8,
        bonus_action_type_count=4,
        reaction_type_count=4,
        class_feature_count=4,
        spell_count=4,
        slot_level_count=4,
        item_count=4,
        gnn_hidden_size=16,
        policy_hidden_size=32,
        context_hidden_sizes=(),
        message_passing_steps=1,
    )


def _single_choice_masks(model: GNNPPOActorCritic) -> dict[str, torch.Tensor]:
    return {
        "action_category": _one_hot(ACTION_CATEGORY_COUNT, int(ActionCategory.END_TURN)),
        "main_action_type": _one_hot(MAIN_ACTION_TYPE_COUNT, 2),
        "bonus_action_type": _one_hot(model.bonus_action_type_count, 1),
        "reaction_type": _one_hot(model.reaction_type_count, 2),
        "class_feature": _one_hot(model.class_feature_count, 3),
        "target_index": _one_hot(model.target_count, 1),
        "move_index": _one_hot(model.move_count, 5),
        "spell_index": _one_hot(model.spell_count, 2),
        "slot_level": _one_hot(model.slot_level_count, 1),
        "item_index": _one_hot(model.item_count, 0),
        "option_index": _one_hot(model.option_count, 4),
    }


def _one_hot(size: int, index: int) -> torch.Tensor:
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask


def test_gnn_ppo_model_returns_legal_action_with_masks() -> None:
    observation = encode_entity_observation(
        _state(
            FighterArcher(Position(0, 0)),
            Goblin(Position(1, 0)),
        ),
        actor_id=0,
    )
    model = _small_model()
    masks = _single_choice_masks(model)

    action = model.act(observation, masks, deterministic=True)
    evaluation = model.evaluate_actions(observation, action, masks)

    assert action["action_category"].item() == int(ActionCategory.END_TURN)
    assert action["main_action_type"].item() == 2
    assert action["bonus_action_type"].item() == 1
    assert action["reaction_type"].item() == 2
    assert action["class_feature"].item() == 3
    assert action["target_index"].item() == 1
    assert action["move_index"].item() == 5
    assert action["spell_index"].item() == 2
    assert action["slot_level"].item() == 1
    assert action["item_index"].item() == 0
    assert action["option_index"].item() == 4
    assert torch.isfinite(action["log_prob"])
    assert torch.isfinite(evaluation["log_prob"]).all()
    assert set(HEAD_ORDER).issubset(action)


def test_gnn_ppo_model_works_with_variable_entity_count() -> None:
    full_observation = encode_entity_observation(
        _state(
            FighterChampionGreatsword(Position(0, 0)),
            WizardEvoker(Position(0, 2)),
            Goblin(Position(1, 0)),
            Orc(Position(6, 6)),
        ),
        actor_id=0,
    )
    sliced_observation = EntityObservation(
        actor_features=full_observation.actor_features,
        entities_features=full_observation.entities_features[:2],
        map_features=full_observation.map_features,
        global_features=full_observation.global_features,
        entity_mask=full_observation.entity_mask[:2],
    )
    model = _small_model()

    full_output = model(full_observation)
    sliced_output = model(sliced_observation)

    assert full_output["action_category_logits"].shape == (1, ACTION_CATEGORY_COUNT)
    assert sliced_output["action_category_logits"].shape == (1, ACTION_CATEGORY_COUNT)
    assert full_output["target_logits"].shape == (1, model.target_count)
    assert sliced_output["target_logits"].shape == (1, model.target_count)


def test_gnn_ppo_model_works_with_batched_map_features() -> None:
    first = encode_entity_observation(
        _state(
            FighterArcher(Position(0, 0)),
            Goblin(Position(1, 0)),
        ),
        actor_id=0,
    )
    second = encode_entity_observation(
        _state(
            FighterArcher(Position(0, 0)),
            WizardEvoker(Position(0, 2)),
            Orc(Position(5, 5)),
        ),
        actor_id=0,
    )
    batched = {
        "actor_features": torch.stack((first.actor_features, second.actor_features)),
        "entities_features": torch.stack((first.entities_features, second.entities_features)),
        "map_features": torch.stack((first.map_features, second.map_features)),
        "global_features": torch.stack((first.global_features, second.global_features)),
        "entity_mask": torch.stack((first.entity_mask, second.entity_mask)),
    }
    model = _small_model()

    outputs = model(batched)
    action = model.act(batched, _single_choice_masks(model), deterministic=True)

    assert outputs["move_logits"].shape == (2, model.move_count)
    assert outputs["value"].shape == (2,)
    assert action["action_category"].shape == (2,)


def test_ppo_config_accepts_gnn_model_type() -> None:
    config = PPOConfig(model_type="gnn")

    assert config.model_type == "gnn"
