import pytest
import torch

from agents import (
    ACTION_CATEGORY_COUNT,
    MAIN_ACTION_TYPE_COUNT,
    OBSERVATION_SIZE,
    ActionCategory,
    MainActionType,
    PPOActorCritic,
    build_action_masks,
    encode_observation,
)
from combat import (
    CombatState,
    FighterArcher,
    FighterChampionGreatsword,
    Goblin,
    GridMap,
    Position,
)


def make_state() -> CombatState:
    return CombatState(
        characters=[
            FighterArcher(Position(0, 0)),
            FighterChampionGreatsword(Position(0, 2)),
            Goblin(Position(1, 0)),
        ],
        grid_map=GridMap(width=8, height=8),
    )


def test_ppo_actor_critic_forward_shapes() -> None:
    model = PPOActorCritic(
        target_count=6,
        move_count=64,
        option_count=8,
        hidden_sizes=(32,),
    )
    observations = torch.zeros((2, OBSERVATION_SIZE), dtype=torch.float32)

    outputs = model(observations)

    assert outputs["action_category_logits"].shape == (2, ACTION_CATEGORY_COUNT)
    assert outputs["main_action_type_logits"].shape == (2, MAIN_ACTION_TYPE_COUNT)
    assert outputs["target_logits"].shape == (2, 6)
    assert outputs["move_logits"].shape == (2, 64)
    assert outputs["option_logits"].shape == (2, 8)
    assert outputs["value"].shape == (2,)


def test_ppo_act_respects_action_category_mask() -> None:
    model = PPOActorCritic(target_count=6, move_count=64, option_count=8, hidden_sizes=(32,))
    observation = torch.zeros(OBSERVATION_SIZE, dtype=torch.float32)
    masks = {
        "action_category": torch.tensor(
            [
                False,
                False,
                False,
                False,
                True,
                False,
            ],
            dtype=torch.bool,
        ),
        "main_action_type": torch.zeros(MAIN_ACTION_TYPE_COUNT, dtype=torch.bool),
        "target_index": torch.zeros(3, dtype=torch.bool),
        "move_index": torch.zeros(64, dtype=torch.bool),
        "option_index": torch.zeros(8, dtype=torch.bool),
    }

    action = model.act(observation, masks, deterministic=True)

    assert action["action_category"].item() == int(ActionCategory.END_TURN)
    assert action["log_prob"].shape == ()
    assert action["entropy"].shape == ()
    assert action["value"].shape == ()
    assert torch.isfinite(action["log_prob"])
    assert torch.isfinite(action["value"])


def test_ppo_evaluate_actions_returns_ppo_terms() -> None:
    torch.manual_seed(7)
    state = make_state()
    observation = encode_observation(state, actor_id=0)
    masks = build_action_masks(state, actor_id=0)
    model = PPOActorCritic(target_count=6, move_count=64, option_count=8, hidden_sizes=(32,))

    action = model.act(observation, masks)
    evaluation = model.evaluate_actions(observation, action, masks)

    assert evaluation["log_prob"].shape == (1,)
    assert evaluation["entropy"].shape == (1,)
    assert evaluation["value"].shape == (1,)
    assert torch.isfinite(evaluation["log_prob"]).all()
    assert torch.isfinite(evaluation["entropy"]).all()
    assert torch.isfinite(evaluation["value"]).all()


def test_ppo_evaluate_actions_rejects_masked_selected_action() -> None:
    model = PPOActorCritic(target_count=6, move_count=64, option_count=8, hidden_sizes=(32,))
    observation = torch.zeros(OBSERVATION_SIZE, dtype=torch.float32)
    masks = {
        "action_category": torch.tensor(
            [
                False,
                False,
                False,
                False,
                True,
                False,
            ],
            dtype=torch.bool,
        ),
        "main_action_type": torch.zeros(MAIN_ACTION_TYPE_COUNT, dtype=torch.bool),
        "target_index": torch.zeros(3, dtype=torch.bool),
        "move_index": torch.zeros(64, dtype=torch.bool),
        "option_index": torch.zeros(8, dtype=torch.bool),
    }

    with pytest.raises(ValueError, match="masked action_category"):
        model.evaluate_actions(
            observation,
            {
                "action_category": int(ActionCategory.MOVEMENT),
                "main_action_type": int(MainActionType.ATTACK),
                "move_index": 0,
                "target_index": 0,
                "option_index": 0,
            },
            masks,
        )


def test_ppo_act_rejects_empty_action_category_mask() -> None:
    model = PPOActorCritic(target_count=6, move_count=64, option_count=8, hidden_sizes=(32,))
    observation = torch.zeros(OBSERVATION_SIZE, dtype=torch.float32)
    masks = {
        "action_category": torch.zeros(ACTION_CATEGORY_COUNT, dtype=torch.bool),
        "main_action_type": torch.zeros(MAIN_ACTION_TYPE_COUNT, dtype=torch.bool),
        "target_index": torch.zeros(3, dtype=torch.bool),
        "move_index": torch.zeros(64, dtype=torch.bool),
        "option_index": torch.zeros(8, dtype=torch.bool),
    }

    with pytest.raises(ValueError, match="action_category mask has no valid actions"):
        model.act(observation, masks)
