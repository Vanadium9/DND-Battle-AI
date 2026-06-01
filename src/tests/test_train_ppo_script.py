from pathlib import Path

from combat import Team
from agents import GNNPPOActorCritic, PPOActorCritic
from scripts.train_ppo import (
    aggregate_action_distribution,
    build_model,
    default_checkpoint_for_model,
    format_report,
    resolve_device,
)
from training import EpisodeStats


def test_aggregate_action_distribution_formats_nonzero_counts() -> None:
    stats = [
        EpisodeStats(
            total_reward=1.0,
            length=2,
            winner=Team.PLAYERS,
            action_counts={
                "MOVEMENT": 1,
                "ATTACK": 1,
                "END_TURN": 0,
                "BONUS_ACTION": 0,
                "REACTION": 0,
            },
        )
    ]

    assert aggregate_action_distribution(stats) == "{ATTACK:0.500, MOVEMENT:0.500}"


def test_format_report_includes_training_metrics() -> None:
    stats = [
        EpisodeStats(
            total_reward=3.0,
            length=5,
            winner=Team.PLAYERS,
            action_counts={
                "MOVEMENT": 2,
                "ATTACK": 2,
                "END_TURN": 1,
                "BONUS_ACTION": 0,
                "REACTION": 0,
            },
        ),
        EpisodeStats(
            total_reward=-1.0,
            length=3,
            winner=Team.ENEMIES,
            action_counts={
                "MOVEMENT": 1,
                "ATTACK": 1,
                "END_TURN": 1,
                "BONUS_ACTION": 0,
                "REACTION": 0,
            },
        ),
    ]

    report = format_report(episode=2, stats=stats, checkpoint_path=Path("model.pt"))

    assert "episode=2" in report
    assert "win_rate=0.500" in report
    assert "average_reward=1.000" in report
    assert "average_episode_length=4.00" in report
    assert "action_distribution=" in report
    assert "checkpoint=model.pt" in report


def test_default_checkpoint_matches_selected_model_type() -> None:
    assert default_checkpoint_for_model("gnn").name == "gnn_ppo_actor_critic.pt"
    assert default_checkpoint_for_model("mlp").name == "ppo_actor_critic.pt"


def test_build_model_uses_selected_architecture() -> None:
    assert isinstance(build_model("gnn"), GNNPPOActorCritic)
    assert isinstance(build_model("mlp"), PPOActorCritic)


def test_resolve_device_auto_returns_available_torch_device() -> None:
    assert resolve_device("auto").type in {"cpu", "cuda"}
    assert resolve_device("cpu").type == "cpu"
