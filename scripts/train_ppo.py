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

from agents import GNNPPOActorCritic, PPOActorCritic
from combat import EncounterGenerator, Team
from configs import PPOConfig
from training import CurriculumConfig, EpisodeStats, PPOTrainer, RolloutBuffer, load_curriculum_config


LOG_INTERVAL = 10
DEFAULT_MLP_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "ppo_actor_critic.pt"
DEFAULT_GNN_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "gnn_ppo_actor_critic.pt"


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)

    checkpoint_path = Path(args.checkpoint or default_checkpoint_for_model(args.model_type))
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    curriculum_config = build_curriculum_config(args)
    generator = EncounterGenerator(
        seed=args.seed,
        curriculum_level=(
            curriculum_config.initial_level
            if curriculum_config.enabled
            else None
        ),
    )
    environment = (
        generator.generate_curriculum_environment(
            curriculum_config.initial_level,
            log_to_console=False,
        )
        if curriculum_config.enabled
        else generator.generate_environment(log_to_console=False)
    )
    model = build_model(args.model_type)
    config = PPOConfig(
        checkpoint_dir=str(checkpoint_path.parent),
        model_type=args.model_type,
        rollout_steps=args.rollout_steps,
        minibatch_size=args.minibatch_size,
        update_epochs=args.update_epochs,
    )
    trainer = PPOTrainer(
        environment=environment,
        model=model,
        config=config,
        device=device,
        encounter_generator=generator,
        curriculum_config=curriculum_config,
        fast_action_masks=args.fast_action_masks,
        fast_observation=args.fast_observation,
        profile_rollout=args.profile_training,
        num_envs=args.num_envs,
    )
    checkpoint_status = load_checkpoint_for_training(
        trainer,
        checkpoint_path,
        resume=not args.no_resume,
        restore_curriculum=args.curriculum_level is None,
    )
    print(
        f"training model_type={args.model_type} device={trainer.device} "
        f"num_envs={trainer.num_envs} "
        f"curriculum={trainer.curriculum_config.enabled} "
        f"curriculum_level={trainer.curriculum_level} "
        f"checkpoint_status={checkpoint_status} "
        f"checkpoint={checkpoint_path}",
        flush=True,
    )

    update_count = args.updates if args.updates is not None else args.episodes
    last_checkpoint = checkpoint_path
    reported_curriculum_transitions = 0

    for update_index in range(1, update_count + 1):
        rollout = trainer.collect_rollout(
            args.rollout_steps,
            max_episode_steps=args.max_episode_steps,
        )
        metrics = trainer.update(rollout)
        for message in trainer.curriculum_transition_log[reported_curriculum_transitions:]:
            print(f"curriculum_transition={message}", flush=True)
        reported_curriculum_transitions = len(trainer.curriculum_transition_log)

        should_report = update_index % args.log_interval == 0 or update_index == update_count
        if should_report:
            last_checkpoint = trainer.save_checkpoint(checkpoint_path.name)
            print(
                format_update_report(update_index, rollout, metrics, last_checkpoint),
                flush=True,
            )

    if last_checkpoint != checkpoint_path:
        trainer.save_checkpoint(checkpoint_path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on random D&D-like encounters.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Backward-compatible alias for --updates when --updates is not set.",
    )
    parser.add_argument(
        "--updates",
        type=int,
        default=None,
        help="Number of PPO update iterations.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=512,
        help="Environment steps collected before each PPO update.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=256,
        help="Force-reset a combat episode after this many environment steps.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of independent combat environments sampled per rollout tick.",
    )
    parser.add_argument(
        "--fast-action-masks",
        action="store_true",
        help="Use reduced training action masks for faster early curriculum runs.",
    )
    parser.add_argument(
        "--fast-observation",
        action="store_true",
        help="Use cheaper observation features with the same tensor sizes for warm-up training.",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Train on staged curriculum encounters instead of random encounters.",
    )
    parser.add_argument(
        "--curriculum-config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "train_curriculum.yaml"),
        help="YAML config for staged curriculum training.",
    )
    parser.add_argument(
        "--curriculum-level",
        type=int,
        default=None,
        help="Initial curriculum level. Defaults to config initial_level.",
    )
    parser.add_argument(
        "--curriculum-max-level",
        type=int,
        default=None,
        help="Maximum curriculum level. Defaults to config max_level.",
    )
    parser.add_argument(
        "--curriculum-threshold",
        type=float,
        default=None,
        help="Win-rate threshold for advancing curriculum difficulty.",
    )
    parser.add_argument(
        "--curriculum-window-size",
        type=int,
        default=None,
        help="Completed episode window used for curriculum advancement.",
    )
    parser.add_argument(
        "--profile-training",
        action="store_true",
        help="Print rollout timing breakdown in training logs.",
    )
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=128,
        help="Minibatch size used during PPO update.",
    )
    parser.add_argument(
        "--update-epochs",
        type=int,
        default=4,
        help="Optimization epochs over each collected rollout.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=LOG_INTERVAL,
        help="Checkpoint and metric print interval in update iterations.",
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
        default=None,
        help=(
            "Checkpoint file path. Defaults to checkpoints/ppo_actor_critic.pt for "
            "MLP and checkpoints/gnn_ppo_actor_critic.pt for GNN."
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start from a fresh model even if the checkpoint file already exists.",
    )
    parser.add_argument(
        "--model-type",
        choices=("mlp", "gnn"),
        default="gnn",
        help="Policy architecture to train.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Training device. auto uses CUDA when PyTorch can see it.",
    )
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be greater than zero")
    if args.updates is not None and args.updates <= 0:
        parser.error("--updates must be greater than zero")
    if args.rollout_steps <= 0:
        parser.error("--rollout-steps must be greater than zero")
    if args.max_episode_steps <= 0:
        parser.error("--max-episode-steps must be greater than zero")
    if args.num_envs <= 0:
        parser.error("--num-envs must be greater than zero")
    if args.curriculum_level is not None and args.curriculum_level <= 0:
        parser.error("--curriculum-level must be greater than zero")
    if args.curriculum_max_level is not None and args.curriculum_max_level <= 0:
        parser.error("--curriculum-max-level must be greater than zero")
    if args.curriculum_threshold is not None and not (0.0 <= args.curriculum_threshold <= 1.0):
        parser.error("--curriculum-threshold must be between 0 and 1")
    if args.curriculum_window_size is not None and args.curriculum_window_size <= 0:
        parser.error("--curriculum-window-size must be greater than zero")
    if args.minibatch_size <= 0:
        parser.error("--minibatch-size must be greater than zero")
    if args.update_epochs <= 0:
        parser.error("--update-epochs must be greater than zero")
    if args.log_interval <= 0:
        parser.error("--log-interval must be greater than zero")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda was requested, but torch.cuda.is_available() is False")
    return args


def build_curriculum_config(args: argparse.Namespace) -> CurriculumConfig:
    """Resolve curriculum settings from CLI without enabling it by accident."""

    if args.curriculum:
        config_path = Path(args.curriculum_config)
        base = load_curriculum_config(config_path) if config_path.exists() else CurriculumConfig()
        return CurriculumConfig(
            enabled=True,
            initial_level=int(args.curriculum_level or base.initial_level),
            max_level=int(args.curriculum_max_level or base.max_level),
            win_rate_threshold=float(
                args.curriculum_threshold
                if args.curriculum_threshold is not None
                else base.win_rate_threshold
            ),
            window_size=int(args.curriculum_window_size or base.window_size),
        )
    if args.curriculum_level is not None:
        return CurriculumConfig(
            enabled=True,
            initial_level=int(args.curriculum_level),
            max_level=int(args.curriculum_max_level or args.curriculum_level),
            win_rate_threshold=float(
                args.curriculum_threshold
                if args.curriculum_threshold is not None
                else 1.0
            ),
            window_size=int(args.curriculum_window_size or 1),
        )
    return CurriculumConfig(enabled=False)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    """Resolve CLI device choice into a torch device."""

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def default_checkpoint_for_model(model_type: str) -> Path:
    """Return the conventional checkpoint path for the selected model."""

    return DEFAULT_GNN_CHECKPOINT if model_type == "gnn" else DEFAULT_MLP_CHECKPOINT


def build_model(model_type: str) -> torch.nn.Module:
    """Create the trainable policy model selected by CLI."""

    if model_type == "gnn":
        return GNNPPOActorCritic()
    return PPOActorCritic(target_count=6, move_count=64)


def load_checkpoint_for_training(
    trainer: PPOTrainer,
    checkpoint_path: Path,
    *,
    resume: bool = True,
    restore_curriculum: bool = True,
) -> str:
    """Load a compatible checkpoint into the trainer, if requested and available."""

    if not resume:
        return "fresh_requested"
    if not checkpoint_path.exists():
        return "fresh_missing"
    try:
        checkpoint = torch.load(checkpoint_path, map_location=trainer.device, weights_only=False)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
    except Exception as exc:
        return f"fresh_incompatible:{type(exc).__name__}"

    optimizer_status = "optimizer_loaded"
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state is not None:
        try:
            trainer.optimizer.load_state_dict(optimizer_state)
        except Exception:
            optimizer_status = "optimizer_fresh"
    else:
        optimizer_status = "optimizer_missing"

    if restore_curriculum and trainer.curriculum_config.enabled:
        _restore_curriculum_state(trainer, checkpoint)
        for env_index in range(len(trainer.environments)):
            trainer._reset_environment_for_episode(env_index)
    return f"loaded:{optimizer_status}"


def _restore_curriculum_state(
    trainer: PPOTrainer,
    checkpoint: dict[str, object],
) -> None:
    curriculum_level = checkpoint.get("curriculum_level")
    if curriculum_level is not None:
        trainer.current_curriculum_level = min(
            trainer.curriculum_config.max_level,
            max(1, int(curriculum_level)),
        )
        if trainer.encounter_generator is not None:
            trainer.encounter_generator.set_curriculum_level(trainer.current_curriculum_level)

    curriculum_state = checkpoint.get("curriculum_state")
    if isinstance(curriculum_state, dict):
        recent_wins = curriculum_state.get("recent_wins", [])
        if isinstance(recent_wins, list):
            trainer.curriculum_recent_wins = [bool(value) for value in recent_wins]
        transition_log = curriculum_state.get("transition_log", [])
        if isinstance(transition_log, list):
            trainer.curriculum_transition_log = [str(value) for value in transition_log]


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


def format_update_report(
    update_index: int,
    rollout: RolloutBuffer,
    metrics: dict[str, float],
    checkpoint_path: Path,
) -> str:
    """Format batched PPO training metrics."""

    finished = len(rollout.episode_winners)
    completed = finished - rollout.episode_timeouts
    winners = [winner for winner in rollout.episode_winners if winner is not None]
    win_rate = (
        sum(1 for winner in winners if winner is Team.PLAYERS) / len(winners)
        if winners
        else 0.0
    )
    average_reward = (
        sum(rollout.rewards) / len(rollout.rewards)
        if rollout.rewards
        else 0.0
    )
    return (
        f"update={update_index} "
        f"rollout_steps={len(rollout)} "
        f"episodes_finished={finished} "
        f"completed={completed} "
        f"timeouts={rollout.episode_timeouts} "
        f"win_rate={win_rate:.3f} "
        f"average_step_reward={average_reward:.3f} "
        f"policy_loss={metrics.get('policy_loss', 0.0):.4f} "
        f"value_loss={metrics.get('value_loss', 0.0):.4f} "
        f"entropy={metrics.get('entropy', 0.0):.4f} "
        f"loss={metrics.get('loss', 0.0):.4f} "
        f"{format_profile_times(rollout)}"
        f"checkpoint={checkpoint_path}"
    )


def format_profile_times(rollout: RolloutBuffer) -> str:
    """Format optional rollout timing metrics."""

    if not rollout.profile_times:
        return ""
    parts = []
    for key in ("observation", "mask", "model_act", "decode", "env_step", "update"):
        seconds = rollout.profile_times.get(key)
        if seconds is None:
            continue
        parts.append(f"{key}_ms={seconds * 1000.0:.1f}")
    return " ".join(parts) + " "


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
