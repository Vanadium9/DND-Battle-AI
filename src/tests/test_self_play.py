from pathlib import Path
from uuid import uuid4

import torch

from agents import PPOActorCritic
from combat import CombatEnvironment, FighterArcher, Goblin, GridMap, Position
from configs import PPOConfig
from training import (
    OpponentPool,
    PPOTrainer,
    SelfPlayConfig,
    SelfPlayManager,
    load_self_play_config,
)


CHECKPOINT_DIR = Path("checkpoints") / "test_self_play"


def _pool_dir(name: str) -> Path:
    path = CHECKPOINT_DIR / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _model() -> PPOActorCritic:
    return PPOActorCritic(target_count=6, move_count=64, hidden_sizes=(16,))


def _environment() -> CombatEnvironment:
    return CombatEnvironment(
        characters=[
            FighterArcher(Position(0, 0)),
            Goblin(Position(1, 0)),
        ],
        grid_map=GridMap(width=8, height=8),
        use_initiative=False,
        log_to_console=False,
    )


def test_opponent_pool_adds_checkpoint() -> None:
    pool = OpponentPool(_pool_dir("adds_checkpoint"), seed=1)
    model = _model()

    checkpoint = pool.add_checkpoint(model, update_index=3)

    assert checkpoint.path.exists()
    assert checkpoint in pool.checkpoints
    loaded = torch.load(checkpoint.path, map_location="cpu", weights_only=False)
    assert loaded["update_index"] == 3
    assert "model_state_dict" in loaded


def test_opponent_pool_selects_random_opponent() -> None:
    pool = OpponentPool(_pool_dir("selects_random"), seed=0)
    first = pool.add_checkpoint(_model(), update_index=1, label="first")
    second = pool.add_checkpoint(_model(), update_index=2, label="second")

    sampled_paths = {pool.sample_opponent().path for _ in range(20)}

    assert sampled_paths == {first.path, second.path}


def test_frozen_enemy_policy_is_not_updated() -> None:
    model = _model()
    manager = SelfPlayManager(
        SelfPlayConfig(
            enabled=True,
            opponent_pool_dir=str(_pool_dir("frozen_enemy")),
            add_current_every_updates=10,
            freeze_enemy_policy=True,
            train_player_side=True,
            train_enemy_side=False,
            seed=0,
        )
    )
    trainer = PPOTrainer(
        environment=_environment(),
        model=model,
        config=PPOConfig(
            rollout_steps=4,
            update_epochs=1,
            minibatch_size=2,
            learning_rate=1.0e-3,
        ),
        self_play_manager=manager,
    )
    manager.before_rollout(trainer)
    enemy_policy = trainer.policy_router.enemy_policy
    before = [parameter.detach().clone() for parameter in enemy_policy.parameters()]

    rollout = trainer.collect_rollout(rollout_steps=4)
    trainer.update(rollout)

    assert all(not parameter.requires_grad for parameter in enemy_policy.parameters())
    assert all(
        torch.allclose(previous, current.detach())
        for previous, current in zip(before, enemy_policy.parameters())
    )


def test_self_play_config_loads_from_yaml() -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "train_self_play.yaml"

    config = load_self_play_config(config_path)

    assert config.enabled is True
    assert config.freeze_enemy_policy is True
    assert config.train_player_side is True
