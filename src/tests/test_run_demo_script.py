import json
from pathlib import Path

import torch

from agents import PPOActorCritic
from combat import MoveAction, Position
from scripts.run_demo import (
    create_demo_environment,
    describe_action,
    format_hp,
    load_ppo_checkpoint,
    run_battle_demo,
)


CHECKPOINT_PATH = Path("checkpoints") / "test_run_demo.pt"


def test_load_ppo_checkpoint_infers_model_shape() -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model = PPOActorCritic(target_count=6, move_count=64, hidden_sizes=(32,))
    torch.save({"model_state_dict": model.state_dict()}, CHECKPOINT_PATH)

    loaded_model = load_ppo_checkpoint(CHECKPOINT_PATH)

    assert loaded_model.target_count == 6
    assert loaded_model.move_count == 64
    assert (
        loaded_model.action_category_head.weight.shape
        == model.action_category_head.weight.shape
    )
    assert (
        loaded_model.main_action_type_head.weight.shape
        == model.main_action_type_head.weight.shape
    )


def test_format_hp_and_describe_action() -> None:
    environment = create_demo_environment()
    action = MoveAction(actor_id=0, destination=Position(1, 1))

    hp_text = format_hp(environment.combat_state.characters)
    action_text = describe_action(action, environment)

    assert "Fighter Champion Greatsword 49/49" in hp_text
    assert "Fighter Archer 44/44" in hp_text
    assert "Orc 18/18" in hp_text
    assert action_text == "MOVE to (1, 1)"


def test_run_battle_demo_prints_turn_fields(capsys) -> None:
    model = PPOActorCritic(target_count=6, move_count=64, hidden_sizes=(32,))
    environment = create_demo_environment()

    run_battle_demo(model, environment, max_steps=1)

    output = capsys.readouterr().out
    assert "Round:" in output
    assert "Actor:" in output
    assert "HP:" in output
    assert "Action:" in output
    assert "Result:" in output
    assert "Winner:" in output


def test_run_battle_demo_saves_replay(capsys) -> None:
    model = PPOActorCritic(target_count=6, move_count=64, hidden_sizes=(32,))
    environment = create_demo_environment()
    replay_dir = Path("checkpoints") / "test_run_demo_replays"

    replay_path = run_battle_demo(
        model,
        environment,
        max_steps=1,
        save_replay=True,
        replay_dir=replay_dir,
    )

    output = capsys.readouterr().out
    assert replay_path is not None
    assert replay_path.exists()
    data = json.loads(replay_path.read_text(encoding="utf-8"))
    assert data["format"] == "BattleReplay"
    assert len(data["steps"]) == 1
    assert "Replay saved:" in output
