"""Self-play opponent pool and PPO trainer integration helpers."""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass, field
import logging
from pathlib import Path
import random
from typing import Any

import torch
import yaml

from agents import ActionCategory, MainActionType
from combat import Team


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SelfPlayConfig:
    """Runtime settings for self-play training."""

    enabled: bool = False
    opponent_pool_dir: str = "checkpoints/self_play_opponents"
    add_current_every_updates: int = 10
    max_opponents: int = 16
    seed: int | None = None
    freeze_enemy_policy: bool = True
    train_player_side: bool = True
    train_enemy_side: bool = False
    add_initial_policy: bool = True


@dataclass(frozen=True)
class OpponentCheckpoint:
    """Saved historical policy checkpoint available as an opponent."""

    path: Path
    update_index: int


@dataclass
class OpponentStats:
    """Per-opponent evaluation counters."""

    games: int = 0
    wins: int = 0
    total_reward: float = 0.0
    action_counts: Counter[str] = field(default_factory=Counter)

    @property
    def win_rate(self) -> float:
        if self.games == 0:
            return 0.0
        return self.wins / self.games

    @property
    def average_reward(self) -> float:
        if self.games == 0:
            return 0.0
        return self.total_reward / self.games

    def as_dict(self) -> dict[str, object]:
        return {
            "games": self.games,
            "wins": self.wins,
            "win_rate": self.win_rate,
            "average_reward": self.average_reward,
            "action_distribution": dict(self.action_counts),
        }


class OpponentPool:
    """Filesystem-backed pool of historical opponent checkpoints."""

    def __init__(
        self,
        directory: str | Path,
        *,
        seed: int | None = None,
        max_opponents: int = 16,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(seed)
        self.max_opponents = max(1, int(max_opponents))
        self.checkpoints: list[OpponentCheckpoint] = []
        self._load_existing_checkpoints()

    def add_checkpoint(
        self,
        model: torch.nn.Module,
        update_index: int,
        *,
        label: str = "policy",
    ) -> OpponentCheckpoint:
        """Save the current policy and add it to the opponent pool."""

        safe_label = "".join(character for character in label if character.isalnum() or character in "-_")
        safe_label = safe_label or "policy"
        path = self.directory / f"opponent_{int(update_index):06d}_{safe_label}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "update_index": int(update_index),
                "label": safe_label,
                "model_class": model.__class__.__name__,
            },
            path,
        )
        checkpoint = OpponentCheckpoint(path=path, update_index=int(update_index))
        self.checkpoints.append(checkpoint)
        self._trim_to_limit()
        LOGGER.info("Added self-play opponent checkpoint: %s", checkpoint.path)
        return checkpoint

    def sample_opponent(self) -> OpponentCheckpoint | None:
        """Select a random opponent checkpoint."""

        if not self.checkpoints:
            return None
        return self.rng.choice(self.checkpoints)

    def load_policy(
        self,
        checkpoint: OpponentCheckpoint,
        model_template: torch.nn.Module,
        *,
        device: torch.device | str = "cpu",
        freeze: bool = True,
    ) -> torch.nn.Module:
        """Load an opponent policy by copying the current model architecture."""

        loaded = torch.load(checkpoint.path, map_location=device, weights_only=False)
        policy = copy.deepcopy(model_template)
        policy.load_state_dict(loaded["model_state_dict"])
        policy.to(device)
        policy.eval()
        if freeze:
            for parameter in policy.parameters():
                parameter.requires_grad = False
        return policy

    def _load_existing_checkpoints(self) -> None:
        for path in sorted(self.directory.glob("opponent_*.pt")):
            update_index = _update_index_from_path(path)
            self.checkpoints.append(
                OpponentCheckpoint(path=path, update_index=update_index)
            )
        self._trim_to_limit()

    def _trim_to_limit(self) -> None:
        if len(self.checkpoints) <= self.max_opponents:
            return
        self.checkpoints = self.checkpoints[-self.max_opponents :]


class SelfPlayManager:
    """Coordinate opponent sampling and checkpointing for PPO self-play."""

    def __init__(
        self,
        config: SelfPlayConfig | None = None,
        *,
        opponent_pool: OpponentPool | None = None,
    ) -> None:
        self.config = config or SelfPlayConfig()
        self.opponent_pool = opponent_pool or OpponentPool(
            self.config.opponent_pool_dir,
            seed=self.config.seed,
            max_opponents=self.config.max_opponents,
        )
        self.update_count = 0
        self.current_opponent: OpponentCheckpoint | None = None
        self.current_opponent_policy: torch.nn.Module | None = None
        self.opponent_stats: dict[str, OpponentStats] = {}
        self.last_log: dict[str, object] = {}

    def before_rollout(self, trainer: Any) -> None:
        """Install the current policy and a sampled historical opponent."""

        if not self.config.enabled:
            return
        if self.config.add_initial_policy and not self.opponent_pool.checkpoints:
            self.opponent_pool.add_checkpoint(
                trainer.model,
                self.update_count,
                label="initial",
            )

        opponent = self.opponent_pool.sample_opponent()
        if opponent is None:
            return

        self.current_opponent = opponent
        self.current_opponent_policy = self.opponent_pool.load_policy(
            opponent,
            trainer.model,
            device=trainer.device,
            freeze=self.config.freeze_enemy_policy,
        )
        trainer.policy_router.player_policy = trainer.model
        trainer.policy_router.enemy_policy = (
            self.current_opponent_policy
            if self.config.freeze_enemy_policy or not self.config.train_enemy_side
            else trainer.model
        )
        trainer.trainable_teams = self._trainable_teams()
        LOGGER.info("Selected self-play opponent checkpoint: %s", opponent.path)

    def after_update(
        self,
        trainer: Any,
        rollout: Any,
        metrics: dict[str, float | str],
    ) -> dict[str, object]:
        """Update self-play statistics and checkpoint the current policy if due."""

        if not self.config.enabled:
            return {}

        self.update_count += 1
        opponent_key = self._current_opponent_key()
        if opponent_key is not None:
            stats = self.opponent_stats.setdefault(opponent_key, OpponentStats())
            winners = list(getattr(rollout, "episode_winners", ()))
            if winners:
                for winner in winners:
                    stats.games += 1
                    stats.wins += int(winner is Team.PLAYERS)
                    stats.total_reward += float(sum(getattr(rollout, "rewards", ())))
            stats.action_counts.update(_action_distribution(rollout))

        if (
            self.config.add_current_every_updates > 0
            and self.update_count % self.config.add_current_every_updates == 0
        ):
            self.opponent_pool.add_checkpoint(
                trainer.model,
                self.update_count,
                label="update",
            )

        self.last_log = {
            "self_play/opponent_checkpoint": opponent_key or "",
            "self_play/average_reward": _average_reward(rollout),
            "self_play/action_distribution": _action_distribution(rollout),
            "self_play/resource_usage": _resource_usage(rollout),
            "self_play/win_rate_by_opponent": self.win_rate_by_opponent(),
        }
        LOGGER.info("Self-play metrics: %s", self.last_log)
        return self.last_log

    def win_rate_by_opponent(self) -> dict[str, float]:
        """Return player-side win rate grouped by opponent checkpoint."""

        return {
            opponent: stats.win_rate
            for opponent, stats in self.opponent_stats.items()
        }

    def _current_opponent_key(self) -> str | None:
        if self.current_opponent is None:
            return None
        return str(self.current_opponent.path)

    def _trainable_teams(self) -> set[Team]:
        teams: set[Team] = set()
        if self.config.train_player_side:
            teams.add(Team.PLAYERS)
        if self.config.train_enemy_side and not self.config.freeze_enemy_policy:
            teams.add(Team.ENEMIES)
        return teams


def load_self_play_config(path: str | Path) -> SelfPlayConfig:
    """Load self-play training settings from YAML."""

    with Path(path).open("r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}
    return SelfPlayConfig(
        enabled=bool(data.get("enabled", False)),
        opponent_pool_dir=str(
            data.get("opponent_pool_dir", "checkpoints/self_play_opponents")
        ),
        add_current_every_updates=int(data.get("add_current_every_updates", 10)),
        max_opponents=int(data.get("max_opponents", 16)),
        seed=data.get("seed"),
        freeze_enemy_policy=bool(data.get("freeze_enemy_policy", True)),
        train_player_side=bool(data.get("train_player_side", True)),
        train_enemy_side=bool(data.get("train_enemy_side", False)),
        add_initial_policy=bool(data.get("add_initial_policy", True)),
    )


def _action_distribution(rollout: Any) -> dict[str, int]:
    actions = getattr(rollout, "actions", {})
    categories = actions.get("action_category", ())
    main_actions = actions.get("main_action_type", ())
    counts: Counter[str] = Counter()
    for category_tensor, main_tensor in zip(categories, main_actions):
        category = ActionCategory(int(category_tensor.item()))
        if category is ActionCategory.MAIN_ACTION:
            counts[MainActionType(int(main_tensor.item())).name] += 1
        else:
            counts[category.name] += 1
    return dict(counts)


def _resource_usage(rollout: Any) -> dict[str, int]:
    actions = getattr(rollout, "actions", {})
    categories = actions.get("action_category", ())
    main_actions = actions.get("main_action_type", ())
    usage = {
        "class_feature_actions": 0,
        "spell_cast_actions": 0,
        "bonus_actions": 0,
        "reactions": 0,
    }
    for category_tensor, main_tensor in zip(categories, main_actions):
        category = ActionCategory(int(category_tensor.item()))
        if category is ActionCategory.CLASS_FEATURE:
            usage["class_feature_actions"] += 1
        elif category is ActionCategory.BONUS_ACTION:
            usage["bonus_actions"] += 1
        elif category is ActionCategory.REACTION:
            usage["reactions"] += 1
        elif (
            category is ActionCategory.MAIN_ACTION
            and MainActionType(int(main_tensor.item())) is MainActionType.CAST_SPELL
        ):
            usage["spell_cast_actions"] += 1
    return usage


def _average_reward(rollout: Any) -> float:
    rewards = list(getattr(rollout, "rewards", ()))
    if not rewards:
        return 0.0
    return float(sum(rewards) / len(rewards))


def _update_index_from_path(path: Path) -> int:
    parts = path.stem.split("_")
    for part in parts:
        if part.isdigit():
            return int(part)
    return 0


__all__ = [
    "OpponentCheckpoint",
    "OpponentPool",
    "OpponentStats",
    "SelfPlayConfig",
    "SelfPlayManager",
    "load_self_play_config",
]
