import pytest
import torch

from agents import ACTION_TYPE_COUNT, ActionType, build_action_masks, decode_action
from combat import (
    AttackAction,
    CombatState,
    EndTurnAction,
    FighterArcher,
    FighterChampionGreatsword,
    Goblin,
    GridMap,
    MoveAction,
    Orc,
    Position,
)


def move_index(position: Position, width: int = 8) -> int:
    return position.y * width + position.x


def make_state() -> CombatState:
    return CombatState(
        characters=[
            FighterArcher(Position(0, 0)),
            FighterChampionGreatsword(Position(0, 2)),
            Goblin(Position(1, 0)),
            Orc(Position(7, 7)),
        ],
        grid_map=GridMap(width=8, height=8),
    )


def test_build_action_masks_for_active_actor() -> None:
    state = make_state()

    masks = build_action_masks(state, actor_id=0)

    assert set(masks) == {"action_type", "target_index", "move_index"}
    assert masks["action_type"].dtype == torch.bool
    assert masks["action_type"].shape == (ACTION_TYPE_COUNT,)
    assert masks["target_index"].shape == (len(state.characters),)
    assert masks["move_index"].shape == (64,)

    assert masks["action_type"][ActionType.MOVE]
    assert masks["action_type"][ActionType.MAIN_ACTION_ATTACK]
    assert masks["action_type"][ActionType.END_TURN]
    assert not masks["action_type"][ActionType.BONUS_ACTION]
    assert not masks["action_type"][ActionType.REACTION]

    assert not masks["target_index"][0]
    assert not masks["target_index"][1]
    assert masks["target_index"][2]
    assert not masks["target_index"][3]

    assert not masks["move_index"][move_index(Position(0, 0))]
    assert masks["move_index"][move_index(Position(0, 1))]
    assert masks["move_index"][move_index(Position(3, 0))]
    assert not masks["move_index"][move_index(Position(7, 7))]


def test_build_action_masks_respect_action_economy() -> None:
    state = make_state()
    actor = state.characters[0]

    actor.action_economy.action_available = False
    masks = build_action_masks(state, actor_id=0)
    assert not masks["action_type"][ActionType.MAIN_ACTION_ATTACK]
    assert not masks["target_index"].any()

    actor.action_economy.movement_remaining = 0
    masks = build_action_masks(state, actor_id=0)
    assert not masks["action_type"][ActionType.MOVE]
    assert not masks["move_index"].any()
    assert masks["action_type"][ActionType.END_TURN]


def test_build_action_masks_mask_non_active_actor() -> None:
    state = make_state()

    masks = build_action_masks(state, actor_id=1)

    assert not masks["action_type"].any()
    assert not masks["target_index"].any()
    assert not masks["move_index"].any()


def test_decode_action_returns_concrete_combat_actions() -> None:
    state = make_state()

    move_action = decode_action(
        ActionType.MOVE,
        target_index=0,
        move_index=move_index(Position(0, 1)),
        state=state,
        actor_id=0,
    )
    attack_action = decode_action(
        ActionType.MAIN_ACTION_ATTACK,
        target_index=2,
        move_index=0,
        state=state,
        actor_id=0,
    )
    end_turn_action = decode_action(
        ActionType.END_TURN,
        target_index=0,
        move_index=0,
        state=state,
        actor_id=0,
    )

    assert isinstance(move_action, MoveAction)
    assert move_action.destination == Position(0, 1)
    assert isinstance(attack_action, AttackAction)
    assert attack_action.target_id == 2
    assert attack_action.weapon is not None
    assert attack_action.weapon.name == "Longbow"
    assert isinstance(end_turn_action, EndTurnAction)


def test_decode_action_rejects_masked_components() -> None:
    state = make_state()

    with pytest.raises(ValueError, match="move_index"):
        decode_action(
            ActionType.MOVE,
            target_index=0,
            move_index=move_index(Position(7, 7)),
            state=state,
            actor_id=0,
        )

    with pytest.raises(ValueError, match="target_index"):
        decode_action(
            ActionType.MAIN_ACTION_ATTACK,
            target_index=3,
            move_index=0,
            state=state,
            actor_id=0,
        )

    with pytest.raises(ValueError, match="BONUS_ACTION"):
        decode_action(
            ActionType.BONUS_ACTION,
            target_index=0,
            move_index=0,
            state=state,
            actor_id=0,
        )
