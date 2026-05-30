from pathlib import Path
from uuid import uuid4

import pytest
import torch

from agents import ActionCategory, GNNPPOActorCritic, PPOActorCritic, build_action_masks
from combat import CombatState, EndTurnAction, FighterArcher, Goblin, GridMap, Position
from inference import BattleAIService, CheckpointLoadError
from ui.services import ModelService


def test_fallback_agent_selects_legal_action() -> None:
    state = _state()
    service = BattleAIService(seed=7)

    action = service.select_action(state, actor_id=0)

    assert action.is_valid(state)
    masks = build_action_masks(state, actor_id=0)
    assert masks["action_category"].any()
    assert service.get_policy_name() == "Fallback: random_legal"


def test_missing_checkpoint_reports_clear_error() -> None:
    service = BattleAIService()
    missing_path = _checkpoint_path("missing_checkpoint")

    with pytest.raises(CheckpointLoadError, match="Checkpoint not found"):
        service.load_checkpoint(missing_path, "mlp")


def test_loaded_mlp_model_is_used_for_action_selection() -> None:
    checkpoint_path = _checkpoint_path("mlp_policy")
    model = _end_turn_mlp_model()
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
    service = BattleAIService(seed=7)

    service.load_checkpoint(checkpoint_path, "mlp")
    action = service.select_action(_state(), actor_id=0)

    assert service.is_model_loaded()
    assert "MLP PPO" in service.get_policy_name()
    assert isinstance(action, EndTurnAction)


def test_loaded_gnn_model_is_used_for_action_selection() -> None:
    checkpoint_path = _checkpoint_path("gnn_policy")
    model = _end_turn_gnn_model()
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
    service = BattleAIService(seed=7)

    service.load_checkpoint(checkpoint_path, "gnn")
    action = service.select_action(_state(), actor_id=0)

    assert service.is_model_loaded()
    assert "GNN PPO" in service.get_policy_name()
    assert isinstance(action, EndTurnAction)


def test_model_service_loads_checkpoint_for_gui_settings() -> None:
    checkpoint_path = _checkpoint_path("model_service_policy")
    settings_path = _checkpoint_path("model_service_settings").with_suffix(".json")
    model = _end_turn_mlp_model()
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
    service = ModelService(settings_path=settings_path)

    service.load_checkpoint(str(checkpoint_path), "mlp")
    action = service.select_action(_state(), actor_id=0)

    assert service.is_model_loaded()
    assert service.settings.checkpoint_path == str(checkpoint_path)
    assert settings_path.exists()
    assert isinstance(action, EndTurnAction)


def _state() -> CombatState:
    return CombatState(
        characters=[
            FighterArcher(Position(0, 0)),
            Goblin(Position(3, 0)),
        ],
        grid_map=GridMap(width=8, height=8),
    )


def _checkpoint_path(prefix: str) -> Path:
    directory = Path("checkpoints") / "test_inference_services"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{prefix}_{uuid4().hex}.pt"


def _end_turn_mlp_model() -> PPOActorCritic:
    model = PPOActorCritic()
    _force_end_turn(model)
    return model


def _end_turn_gnn_model() -> GNNPPOActorCritic:
    model = GNNPPOActorCritic()
    _force_end_turn(model)
    return model


def _force_end_turn(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.action_category_head.bias[int(ActionCategory.END_TURN)] = 10.0
    model.eval()
