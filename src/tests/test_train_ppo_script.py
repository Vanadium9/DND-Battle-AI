from pathlib import Path

from combat import Team
from scripts.train_ppo import aggregate_action_distribution, format_report
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
