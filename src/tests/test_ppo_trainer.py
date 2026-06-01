from pathlib import Path

import torch

from agents import (
    ACTION_CATEGORY_COUNT,
    MAIN_ACTION_TYPE_COUNT,
    GNNPPOActorCritic,
    HEAD_ORDER,
    PPOActorCritic,
)
from combat import CombatEnvironment, EncounterGenerator, FighterArcher, Goblin, GridMap, Position, Team
from configs import PPOConfig
from training import CurriculumConfig, PPOTrainer, RolloutBuffer, load_curriculum_config


CHECKPOINT_DIR = Path("checkpoints") / "test_ppo_trainer"


def make_trainer(rollout_steps: int = 4) -> PPOTrainer:
    environment = CombatEnvironment(
        characters=[
            FighterArcher(Position(0, 0)),
            Goblin(Position(1, 0)),
        ],
        grid_map=GridMap(width=8, height=8),
        use_initiative=False,
        log_to_console=False,
    )
    model = PPOActorCritic(target_count=6, move_count=64, hidden_sizes=(32,))
    config = PPOConfig(
        rollout_steps=rollout_steps,
        update_epochs=1,
        minibatch_size=2,
        learning_rate=1.0e-3,
        checkpoint_dir=str(CHECKPOINT_DIR),
    )
    return PPOTrainer(environment=environment, model=model, config=config)


def make_gnn_trainer(
    *,
    centralized_critic: bool,
    rollout_steps: int = 2,
) -> PPOTrainer:
    environment = CombatEnvironment(
        characters=[
            FighterArcher(Position(0, 0)),
            Goblin(Position(1, 0)),
        ],
        grid_map=GridMap(width=8, height=8),
        use_initiative=False,
        log_to_console=False,
    )
    model = GNNPPOActorCritic(
        target_count=6,
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
    config = PPOConfig(
        rollout_steps=rollout_steps,
        update_epochs=1,
        minibatch_size=2,
        learning_rate=1.0e-3,
        checkpoint_dir=str(CHECKPOINT_DIR),
        model_type="gnn",
        centralized_critic=centralized_critic,
    )
    return PPOTrainer(environment=environment, model=model, config=config)


def test_collect_rollout_stores_ppo_fields() -> None:
    trainer = make_trainer(rollout_steps=4)

    rollout = trainer.collect_rollout()

    assert len(rollout) == 4
    assert len(rollout.observations) == 4
    assert len(rollout.log_probs) == 4
    assert len(rollout.rewards) == 4
    assert len(rollout.dones) == 4
    assert len(rollout.values) == 4
    assert set(rollout.actions) == {
        "action_category",
        "main_action_type",
        "target_index",
        "move_index",
        "option_index",
    }
    assert set(rollout.masks) == {
        "action_category",
        "main_action_type",
        "target_index",
        "move_index",
        "option_index",
    }
    assert rollout.masks["action_category"][0].shape == (ACTION_CATEGORY_COUNT,)
    assert rollout.masks["main_action_type"][0].shape == (MAIN_ACTION_TYPE_COUNT,)
    assert rollout.masks["target_index"][0].shape == (trainer.model.target_count,)
    assert rollout.masks["move_index"][0].shape == (trainer.model.move_count,)
    assert rollout.masks["option_index"][0].shape == (trainer.model.option_count,)


def test_collect_rollout_resets_episode_on_step_timeout() -> None:
    trainer = make_trainer(rollout_steps=4)

    rollout = trainer.collect_rollout(max_episode_steps=2)

    assert len(rollout) == 4
    assert rollout.episode_timeouts >= 1
    assert None in rollout.episode_winners
    assert any(rollout.dones)


def test_collect_rollout_supports_fast_masks_and_profile_timings() -> None:
    trainer = make_trainer(rollout_steps=4)
    trainer.fast_action_masks = True
    trainer.fast_observation = True
    trainer.profile_rollout = True

    rollout = trainer.collect_rollout(max_episode_steps=4)
    metrics = trainer.update(rollout)

    assert len(rollout) == 4
    assert rollout.profile_times["mask"] >= 0.0
    assert rollout.profile_times["observation"] >= 0.0
    assert rollout.profile_times["model_act"] >= 0.0
    assert rollout.profile_times["env_step"] >= 0.0
    assert rollout.profile_times["update"] >= 0.0
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


def test_collect_rollout_supports_multiple_environments() -> None:
    generator = EncounterGenerator(seed=33)
    environment = generator.generate_environment(log_to_console=False)
    model = PPOActorCritic(target_count=6, move_count=64, hidden_sizes=(32,))
    trainer = PPOTrainer(
        environment=environment,
        model=model,
        config=PPOConfig(
            rollout_steps=6,
            update_epochs=1,
            minibatch_size=3,
            checkpoint_dir=str(CHECKPOINT_DIR),
        ),
        encounter_generator=generator,
        num_envs=3,
        fast_action_masks=True,
    )

    rollout = trainer.collect_rollout(max_episode_steps=3)
    metrics = trainer.update(rollout)

    assert trainer.num_envs == 3
    assert len(trainer.environments) == 3
    assert len(rollout) == 6
    assert rollout.last_values_by_env
    assert len(set(rollout.env_ids)) > 1
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


def test_multi_environment_rollout_tracks_advantages_per_environment() -> None:
    trainer = make_trainer(rollout_steps=4)
    rollout = RolloutBuffer(
        rewards=[1.0, 10.0, 1.0, 10.0],
        dones=[False, False, True, True],
        values=[
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
        ],
        next_values=[
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
        ],
        env_ids=[0, 1, 0, 1],
    )
    trainer.config.gamma = 1.0
    trainer.config.gae_lambda = 1.0

    returns, advantages = trainer.compute_returns_and_advantages(rollout)

    assert torch.allclose(returns.cpu(), torch.tensor([2.0, 20.0, 1.0, 10.0]))
    assert torch.allclose(advantages.cpu(), returns.cpu())


def test_multi_environment_rollout_bootstraps_each_environment() -> None:
    trainer = make_trainer(rollout_steps=4)
    rollout = RolloutBuffer(
        rewards=[1.0, 10.0, 1.0, 10.0],
        dones=[False, False, True, True],
        values=[
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
        ],
        env_ids=[0, 1, 0, 1],
        last_values_by_env={0: 0.0, 1: 0.0},
    )
    trainer.config.gamma = 1.0
    trainer.config.gae_lambda = 1.0

    returns, advantages = trainer.compute_returns_and_advantages(rollout)

    assert torch.allclose(returns.cpu(), torch.tensor([2.0, 20.0, 1.0, 10.0]))
    assert torch.allclose(advantages.cpu(), returns.cpu())


def test_collect_episode_returns_rollout_and_episode_stats() -> None:
    trainer = make_trainer(rollout_steps=4)

    rollout, stats = trainer.collect_episode(max_steps=4)

    assert len(rollout) == stats.length
    assert 0 < stats.length <= 4
    assert stats.total_reward == sum(rollout.rewards)
    assert {"ATTACK", "MOVEMENT", "END_TURN"}.issubset(stats.action_counts)
    assert sum(stats.action_counts.values()) == stats.length


def test_compute_returns_and_advantages_uses_done_boundaries() -> None:
    trainer = make_trainer(rollout_steps=2)
    trainer.config.gamma = 1.0
    trainer.config.gae_lambda = 1.0
    rollout = RolloutBuffer(
        rewards=[1.0, 1.0],
        dones=[False, True],
        values=[torch.tensor(0.5), torch.tensor(0.25)],
        last_value=0.0,
    )

    returns, advantages = trainer.compute_returns_and_advantages(rollout)

    assert torch.allclose(returns.cpu(), torch.tensor([2.0, 1.0]))
    assert torch.allclose(advantages.cpu(), torch.tensor([1.5, 0.75]))


def test_update_runs_ppo_objective() -> None:
    trainer = make_trainer(rollout_steps=4)
    rollout = trainer.collect_rollout()

    metrics = trainer.update(rollout)

    assert set(metrics) == {"policy_loss", "value_loss", "entropy", "loss"}
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


def test_gnn_trainer_runs_with_centralized_critic_enabled() -> None:
    trainer = make_gnn_trainer(centralized_critic=True)
    rollout = trainer.collect_rollout()

    data = rollout.to_tensors(torch.device("cpu"))
    metrics = trainer.update(rollout)

    assert len(rollout.actor_observations) == len(rollout)
    assert len(rollout.critic_observations) == len(rollout)
    assert {"actor_features", "entities_features", "entity_mask"}.issubset(
        data["actor_observations"]
    )
    assert {"actor_features", "entities_features", "entity_mask"}.issubset(
        data["critic_observations"]
    )
    assert set(HEAD_ORDER).issubset(data["actions"])
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


def test_gnn_trainer_pads_all_policy_head_masks() -> None:
    trainer = make_gnn_trainer(centralized_critic=False)
    trainer.fast_action_masks = True

    rollout = trainer.collect_rollout(max_episode_steps=2)
    data = rollout.to_tensors(torch.device("cpu"))

    assert set(HEAD_ORDER).issubset(data["masks"])
    for key in (
        "bonus_action_type",
        "reaction_type",
        "class_feature",
        "spell_index",
        "slot_level",
        "item_index",
    ):
        assert data["masks"][key].sum(dim=1).eq(1).all()
        assert data["masks"][key][:, 0].all()


def test_gnn_trainer_runs_with_centralized_critic_disabled() -> None:
    trainer = make_gnn_trainer(centralized_critic=False)
    rollout = trainer.collect_rollout()

    metrics = trainer.update(rollout)

    assert len(rollout.actor_observations) == len(rollout)
    assert len(rollout.critic_observations) == len(rollout)
    assert set(HEAD_ORDER).issubset(rollout.actions)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


def test_train_iteration_saves_checkpoint() -> None:
    trainer = make_trainer(rollout_steps=4)

    metrics = trainer.train_iteration(checkpoint_name="test_model.pt")

    checkpoint_path = Path(str(metrics["checkpoint_path"]))
    assert checkpoint_path.exists()
    assert checkpoint_path.parent == CHECKPOINT_DIR
    assert checkpoint_path.name == "test_model.pt"


def test_curriculum_level_advances_at_win_rate_threshold() -> None:
    generator = EncounterGenerator(seed=21, curriculum_level=1)
    environment = generator.generate_curriculum_environment(log_to_console=False)
    trainer = PPOTrainer(
        environment=environment,
        model=PPOActorCritic(target_count=6, move_count=64, hidden_sizes=(32,)),
        config=PPOConfig(rollout_steps=2, checkpoint_dir=str(CHECKPOINT_DIR)),
        encounter_generator=generator,
        curriculum_config=CurriculumConfig(
            enabled=True,
            initial_level=1,
            max_level=3,
            win_rate_threshold=1.0,
            window_size=2,
        ),
    )

    trainer.record_curriculum_result(Team.PLAYERS)
    assert trainer.curriculum_level == 1

    trainer.record_curriculum_result(Team.PLAYERS)

    assert trainer.curriculum_level == 2
    assert generator.curriculum_level == 2
    assert trainer.curriculum_recent_wins == []
    assert "Curriculum advanced from level 1" in trainer.curriculum_transition_log[-1]


def test_curriculum_state_is_saved_in_checkpoint() -> None:
    generator = EncounterGenerator(seed=22, curriculum_level=1)
    environment = generator.generate_curriculum_environment(log_to_console=False)
    trainer = PPOTrainer(
        environment=environment,
        model=PPOActorCritic(target_count=6, move_count=64, hidden_sizes=(32,)),
        config=PPOConfig(rollout_steps=2, checkpoint_dir=str(CHECKPOINT_DIR)),
        encounter_generator=generator,
        curriculum_config=CurriculumConfig(
            enabled=True,
            initial_level=1,
            max_level=3,
            win_rate_threshold=1.0,
            window_size=1,
        ),
    )
    trainer.record_curriculum_result(Team.PLAYERS)

    checkpoint_path = trainer.save_checkpoint("curriculum_model.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert checkpoint["curriculum_level"] == 2
    assert checkpoint["curriculum_state"]["enabled"] is True
    assert checkpoint["curriculum_state"]["current_level"] == 2
    assert checkpoint["curriculum_state"]["transition_log"]


def test_train_curriculum_config_loads_from_yaml() -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "train_curriculum.yaml"

    config = load_curriculum_config(config_path)

    assert config.enabled is True
    assert config.initial_level == 1
    assert config.max_level == 13
    assert config.win_rate_threshold == 0.75
