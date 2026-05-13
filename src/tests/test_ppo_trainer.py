from pathlib import Path

import torch

from agents import ACTION_TYPE_COUNT, PPOActorCritic
from combat import CombatEnvironment, FighterArcher, Goblin, GridMap, Position
from configs import PPOConfig
from training import PPOTrainer, RolloutBuffer


CHECKPOINT_DIR = Path("checkpoints") / "test_ppo_trainer"


def make_trainer(rollout_steps: int = 4) -> PPOTrainer:
    environment = CombatEnvironment(
        characters=[
            FighterArcher(Position(0, 0)),
            Goblin(Position(1, 0)),
        ],
        grid_map=GridMap(width=8, height=8),
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


def test_collect_rollout_stores_ppo_fields() -> None:
    trainer = make_trainer(rollout_steps=4)

    rollout = trainer.collect_rollout()

    assert len(rollout) == 4
    assert len(rollout.observations) == 4
    assert len(rollout.log_probs) == 4
    assert len(rollout.rewards) == 4
    assert len(rollout.dones) == 4
    assert len(rollout.values) == 4
    assert set(rollout.actions) == {"action_type", "target_index", "move_index"}
    assert set(rollout.masks) == {"action_type", "target_index", "move_index"}
    assert rollout.masks["action_type"][0].shape == (ACTION_TYPE_COUNT,)
    assert rollout.masks["target_index"][0].shape == (trainer.model.target_count,)
    assert rollout.masks["move_index"][0].shape == (trainer.model.move_count,)


def test_collect_episode_returns_rollout_and_episode_stats() -> None:
    trainer = make_trainer(rollout_steps=4)

    rollout, stats = trainer.collect_episode(max_steps=4)

    assert len(rollout) == stats.length
    assert 0 < stats.length <= 4
    assert stats.total_reward == sum(rollout.rewards)
    assert set(stats.action_counts) == {
        "MOVE",
        "MAIN_ACTION_ATTACK",
        "END_TURN",
        "BONUS_ACTION",
        "REACTION",
    }
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


def test_train_iteration_saves_checkpoint() -> None:
    trainer = make_trainer(rollout_steps=4)

    metrics = trainer.train_iteration(checkpoint_name="test_model.pt")

    checkpoint_path = Path(str(metrics["checkpoint_path"]))
    assert checkpoint_path.exists()
    assert checkpoint_path.parent == CHECKPOINT_DIR
    assert checkpoint_path.name == "test_model.pt"
