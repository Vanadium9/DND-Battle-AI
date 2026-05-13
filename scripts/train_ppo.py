from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import random
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents import PPOActorCritic
from combat import EncounterGenerator, Team
from configs import PPOConfig
from training import EpisodeStats, PPOTrainer


LOG_INTERVAL = 10
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "ppo_actor_critic.pt"


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    generator = EncounterGenerator(seed=args.seed)
    environment = generator.generate_environment(log_to_console=False)
    model = PPOActorCritic(target_count=6, move_count=64)
    config = PPOConfig(checkpoint_dir=str(checkpoint_path.parent))
    trainer = PPOTrainer(environment=environment, model=model, config=config)

    recent_stats: list[EpisodeStats] = []
    last_checkpoint = checkpoint_path

    for episode in range(1, args.episodes + 1):
        trainer.environment = generator.generate_environment(log_to_console=False)
        rollout, stats = trainer.collect_episode()
        recent_stats.append(stats)

        if len(rollout) > 0:
            trainer.update(rollout)

        should_report = episode % LOG_INTERVAL == 0 or episode == args.episodes
        if should_report:
            last_checkpoint = trainer.save_checkpoint(checkpoint_path.name)
            print(format_report(episode, recent_stats, last_checkpoint), flush=True)
            recent_stats = []

    if last_checkpoint != checkpoint_path:
        trainer.save_checkpoint(checkpoint_path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on random D&D-like encounters.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of training episodes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for encounters and PyTorch.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Checkpoint file path.",
    )
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be greater than zero")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def format_report(
    episode: int,
    stats: list[EpisodeStats],
    checkpoint_path: Path,
) -> str:
    if not stats:
        return f"episode={episode} no episodes collected checkpoint={checkpoint_path}"

    win_rate = sum(1 for item in stats if item.winner is Team.PLAYERS) / len(stats)
    average_reward = sum(item.total_reward for item in stats) / len(stats)
    average_length = sum(item.length for item in stats) / len(stats)
    action_distribution = aggregate_action_distribution(stats)

    return (
        f"episode={episode} "
        f"win_rate={win_rate:.3f} "
        f"average_reward={average_reward:.3f} "
        f"average_episode_length={average_length:.2f} "
        f"action_distribution={action_distribution} "
        f"checkpoint={checkpoint_path}"
    )


def aggregate_action_distribution(stats: list[EpisodeStats]) -> str:
    counts: Counter[str] = Counter()
    for item in stats:
        counts.update(item.action_counts)

    total = sum(counts.values())
    if total == 0:
        return "{}"

    parts = [
        f"{name}:{counts[name] / total:.3f}"
        for name in sorted(counts)
        if counts[name] > 0
    ]
    return "{" + ", ".join(parts) + "}"


if __name__ == "__main__":
    main()
