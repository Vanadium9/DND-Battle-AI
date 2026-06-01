from pathlib import Path

from combat import Team
from agents import GNNPPOActorCritic, PPOActorCritic
from scripts.train_ppo import (
    aggregate_action_distribution,
    build_curriculum_config,
    build_model,
    default_checkpoint_for_model,
    format_report,
    format_profile_times,
    format_update_report,
    resolve_device,
)
from training import EpisodeStats, RolloutBuffer


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


def test_build_curriculum_config_is_disabled_by_default() -> None:
    args = type(
        "Args",
        (),
        {
            "curriculum": False,
            "curriculum_config": "configs/train_curriculum.yaml",
            "curriculum_level": None,
            "curriculum_max_level": None,
            "curriculum_threshold": None,
            "curriculum_window_size": None,
        },
    )()

    config = build_curriculum_config(args)

    assert config.enabled is False


def test_build_curriculum_config_loads_yaml_when_enabled() -> None:
    args = type(
        "Args",
        (),
        {
            "curriculum": True,
            "curriculum_config": "configs/train_curriculum.yaml",
            "curriculum_level": None,
            "curriculum_max_level": None,
            "curriculum_threshold": None,
            "curriculum_window_size": None,
        },
    )()

    config = build_curriculum_config(args)

    assert config.enabled is True
    assert config.initial_level == 1
    assert config.max_level == 13
    assert config.win_rate_threshold == 0.75


def test_format_update_report_includes_batched_metrics() -> None:
    rollout = RolloutBuffer()
    rollout.rewards.extend([1.0, -0.5, 0.25])
    rollout.episode_winners.extend([Team.PLAYERS, Team.ENEMIES, None])
    rollout.episode_timeouts = 1
    metrics = {
        "policy_loss": 0.1,
        "value_loss": 0.2,
        "entropy": 0.3,
        "loss": 0.4,
    }

    report = format_update_report(
        update_index=7,
        rollout=rollout,
        metrics=metrics,
        checkpoint_path=Path("model.pt"),
    )

    assert "update=7" in report
    assert "rollout_steps=3" in report
    assert "episodes_finished=3" in report
    assert "completed=2" in report
    assert "timeouts=1" in report
    assert "win_rate=0.500" in report
    assert "average_step_reward=0.250" in report
    assert "policy_loss=0.1000" in report
    assert "checkpoint=model.pt" in report


def test_format_profile_times_is_optional() -> None:
    rollout = RolloutBuffer()
    assert format_profile_times(rollout) == ""

    rollout.profile_times = {"mask": 0.001, "update": 0.25}

    formatted = format_profile_times(rollout)

    assert "mask_ms=1.0" in formatted
    assert "update_ms=250.0" in formatted
