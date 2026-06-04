from pathlib import Path

import torch

from combat import Team
from agents import GNNPPOActorCritic, PPOActorCritic
from agents.action_space import build_fast_training_action_masks
from combat import CombatEnvironment, EncounterGenerator, FighterArcher, Goblin, GridMap, MAX_CURRICULUM_LEVEL, Position
from scripts.train_ppo import (
    aggregate_action_distribution,
    build_curriculum_config,
    build_model,
    build_training_policy_kwargs,
    default_checkpoint_for_model,
    format_report,
    format_profile_times,
    format_update_report,
    initial_reported_curriculum_transitions,
    load_frozen_policy,
    load_checkpoint_for_training,
    resolve_device,
)
from configs import PPOConfig
from training import CurriculumConfig, EpisodeStats, PPOTrainer, RolloutBuffer


def make_script_trainer() -> PPOTrainer:
    environment = CombatEnvironment(
        characters=[FighterArcher(Position(0, 0)), Goblin(Position(1, 0))],
        grid_map=GridMap(width=8, height=8),
        use_initiative=False,
        log_to_console=False,
    )
    return PPOTrainer(
        environment=environment,
        model=PPOActorCritic(target_count=6, move_count=64, hidden_sizes=(16,)),
        config=PPOConfig(
            checkpoint_dir=str(Path("checkpoints") / "test_train_ppo_script"),
            update_epochs=1,
            minibatch_size=1,
        ),
    )


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


def test_load_checkpoint_for_training_loads_model_state() -> None:
    checkpoint_dir = Path("checkpoints") / "test_train_ppo_script"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    environment = CombatEnvironment(
        characters=[FighterArcher(Position(0, 0)), Goblin(Position(1, 0))],
        grid_map=GridMap(width=8, height=8),
        use_initiative=False,
        log_to_console=False,
    )
    model = PPOActorCritic(target_count=6, move_count=64, hidden_sizes=(16,))
    trainer = PPOTrainer(
        environment=environment,
        model=model,
        config=PPOConfig(checkpoint_dir=str(checkpoint_dir), update_epochs=1, minibatch_size=1),
    )
    checkpoint_path = checkpoint_dir / "resume_model.pt"
    with torch.no_grad():
        trainer.model.value_head.bias.fill_(3.0)
    trainer.save_checkpoint(checkpoint_path.name)
    with torch.no_grad():
        trainer.model.value_head.bias.zero_()

    status = load_checkpoint_for_training(trainer, checkpoint_path)

    assert status.startswith("loaded:")
    assert trainer.model.value_head.bias.item() == 3.0


def test_resume_skips_already_logged_curriculum_transitions() -> None:
    trainer = make_script_trainer()
    trainer.curriculum_transition_log.extend(["old transition", "older transition"])

    assert initial_reported_curriculum_transitions(trainer, "loaded:optimizer_loaded") == 2
    assert initial_reported_curriculum_transitions(trainer, "fresh_missing") == 0


def test_default_gnn_model_covers_curriculum_option_masks() -> None:
    model = build_model("gnn")
    generator = EncounterGenerator(seed=0)
    max_option_size = 0
    for level in range(1, MAX_CURRICULUM_LEVEL + 1):
        state = generator.generate_curriculum_state(level)
        for actor_id, actor in enumerate(state.characters):
            if actor.team is Team.PLAYERS:
                masks = build_fast_training_action_masks(state, actor_id)
                max_option_size = max(max_option_size, masks["option_index"].shape[0])

    assert model.option_count >= max_option_size


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
    assert config.min_updates_per_level == 10


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
    assert "trained_win_rate=0.500" in report
    assert "opponent_win_rate=0.500" in report
    assert "average_step_reward=0.250" in report
    assert "policy_loss=0.1000" in report
    assert "checkpoint=model.pt" in report


def test_format_update_report_includes_curriculum_status() -> None:
    trainer = PPOTrainer(
        environment=CombatEnvironment(
            characters=[FighterArcher(Position(0, 0)), Goblin(Position(1, 0))],
            grid_map=GridMap(width=8, height=8),
            use_initiative=False,
            log_to_console=False,
        ),
        model=PPOActorCritic(target_count=6, move_count=64, hidden_sizes=(16,)),
        config=PPOConfig(
            checkpoint_dir=str(Path("checkpoints") / "test_train_ppo_script"),
            update_epochs=1,
            minibatch_size=1,
        ),
        encounter_generator=EncounterGenerator(seed=1, curriculum_level=1),
        curriculum_config=CurriculumConfig(enabled=True, initial_level=1, window_size=2),
    )
    trainer.curriculum_recent_wins = [True, False]
    trainer.curriculum_updates_on_level = 3
    rollout = RolloutBuffer()
    rollout.rewards.extend([1.0])
    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "loss": 0.0}

    report = format_update_report(
        update_index=1,
        rollout=rollout,
        metrics=metrics,
        checkpoint_path=Path("model.pt"),
        trainer=trainer,
    )

    assert "curriculum_level=1" in report
    assert "curriculum_stage=1_Fighter_level_1_vs_1_Goblin" in report
    assert "curriculum_window=2/2" in report
    assert "curriculum_window_win_rate=0.500" in report
    assert "curriculum_updates_on_level=3/0" in report


def test_format_profile_times_is_optional() -> None:
    rollout = RolloutBuffer()
    assert format_profile_times(rollout) == ""

    rollout.profile_times = {"mask": 0.001, "update": 0.25}

    formatted = format_profile_times(rollout)

    assert "mask_ms=1.0" in formatted
    assert "update_ms=250.0" in formatted


def test_build_training_policy_kwargs_trains_players_against_opponent() -> None:
    args = type(
        "Args",
        (),
        {
            "train_side": "players",
            "opponent_policy": "aggressive",
            "opponent_checkpoint": None,
            "opponent_model_type": None,
            "model_type": "mlp",
            "seed": 7,
        },
    )()
    model = build_model("mlp")

    kwargs = build_training_policy_kwargs(args, model, torch.device("cpu"))

    assert kwargs["shared_policy"] is model
    assert kwargs["trainable_teams"] == {Team.PLAYERS}
    assert kwargs["enemy_policy"] is not model


def test_build_training_policy_kwargs_trains_enemies_against_opponent() -> None:
    args = type(
        "Args",
        (),
        {
            "train_side": "enemies",
            "opponent_policy": "aggressive",
            "opponent_checkpoint": None,
            "opponent_model_type": None,
            "model_type": "mlp",
            "seed": 7,
        },
    )()
    model = build_model("mlp")

    kwargs = build_training_policy_kwargs(args, model, torch.device("cpu"))

    assert kwargs["enemy_policy"] is model
    assert kwargs["trainable_teams"] == {Team.ENEMIES}
    assert kwargs["player_policy"] is not model


def test_load_frozen_policy_checkpoint_disables_gradients() -> None:
    checkpoint_dir = Path("checkpoints") / "test_frozen_opponent"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model = build_model("mlp")
    checkpoint_path = checkpoint_dir / "opponent.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    policy = load_frozen_policy(checkpoint_path, "mlp", torch.device("cpu"))

    assert isinstance(policy, PPOActorCritic)
    assert all(not parameter.requires_grad for parameter in policy.parameters())
