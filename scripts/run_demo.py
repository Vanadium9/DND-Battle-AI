from __future__ import annotations

import argparse
from pathlib import Path
import sys
from collections.abc import Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents import PPOActorCritic, build_action_masks, decode_action, encode_observation
from combat import (
    AttackAction,
    Character,
    CombatAction,
    CombatEnvironment,
    EndTurnAction,
    MoveAction,
    create_test_encounter,
)


DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "ppo_actor_critic.pt"
DEFAULT_MAX_STEPS = 200


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    model = load_ppo_checkpoint(checkpoint_path)
    environment = create_demo_environment()

    print(f"Loaded checkpoint: {checkpoint_path}")
    run_battle_demo(model, environment, max_steps=args.max_steps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained PPO model in a test combat.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Path to a trained PPO checkpoint.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Maximum number of combat steps before stopping the demo.",
    )
    args = parser.parse_args()
    if args.max_steps <= 0:
        parser.error("--max-steps must be greater than zero")
    return args


def resolve_checkpoint_path(checkpoint: str) -> Path:
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_absolute():
        return checkpoint_path
    return PROJECT_ROOT / checkpoint_path


def load_ppo_checkpoint(checkpoint_path: Path) -> PPOActorCritic:
    """Load a PPO model and infer architecture sizes from its state dict."""

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"PPO checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported PPO checkpoint format: {checkpoint_path}")

    model = PPOActorCritic(
        observation_size=_infer_observation_size(state_dict),
        target_count=state_dict["target_head.weight"].shape[0],
        move_count=state_dict["move_head.weight"].shape[0],
        action_type_count=state_dict["action_type_head.weight"].shape[0],
        hidden_sizes=_infer_hidden_sizes(state_dict),
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def create_demo_environment() -> CombatEnvironment:
    combat_state = create_test_encounter()
    return CombatEnvironment(
        characters=combat_state.characters,
        grid_map=combat_state.grid_map,
        log_to_console=False,
    )


def run_battle_demo(
    model: PPOActorCritic,
    environment: CombatEnvironment,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> None:
    for _ in range(max_steps):
        if environment.is_done():
            break

        state = environment.combat_state
        actor_id = state.turn_index % len(state.characters)
        actor = state.characters[actor_id]
        observation = encode_observation(state, actor_id)
        masks = build_action_masks(state, actor_id)

        with torch.no_grad():
            model_action = model.act(observation, masks, deterministic=True)

        action = decode_action(
            int(model_action["action_type"].item()),
            int(model_action["target_index"].item()),
            int(model_action["move_index"].item()),
            state,
            actor_id,
        )

        print(f"Round: {state.round_number}")
        print(f"Actor: {actor.name}")
        print(f"HP: {format_hp(state.characters)}")
        print(f"Action: {describe_action(action, environment)}")
        result = environment.step(action)
        print(f"Result: {result.description}")
        print("")

    if not environment.is_done():
        print(f"Demo stopped after {max_steps} steps.")

    winner = environment.get_winner()
    winner_text = winner.value if winner is not None else "none"
    print(f"Winner: {winner_text}")


def format_hp(characters: Sequence[Character]) -> str:
    parts = []
    for character in characters:
        parts.append(f"{character.name} {character.hp}/{character.max_hp}")
    return "; ".join(parts)


def describe_action(action: CombatAction, environment: CombatEnvironment) -> str:
    if isinstance(action, MoveAction):
        return f"MOVE to ({action.destination.x}, {action.destination.y})"
    if isinstance(action, AttackAction):
        target = environment.combat_state.character_at(action.target_id)
        target_name = target.name if target is not None else f"target {action.target_id}"
        weapon_name = action.weapon.name if action.weapon is not None else "weapon"
        return f"MAIN_ACTION_ATTACK {target_name} with {weapon_name}"
    if isinstance(action, EndTurnAction):
        return "END_TURN"
    return action.__class__.__name__


def _infer_hidden_sizes(state_dict: dict[str, torch.Tensor]) -> tuple[int, ...]:
    hidden_sizes = []
    index = 0
    while f"encoder.{index}.weight" in state_dict:
        hidden_sizes.append(int(state_dict[f"encoder.{index}.weight"].shape[0]))
        index += 2
    return tuple(hidden_sizes)


def _infer_observation_size(state_dict: dict[str, torch.Tensor]) -> int:
    if "encoder.0.weight" in state_dict:
        return int(state_dict["encoder.0.weight"].shape[1])
    return int(state_dict["action_type_head.weight"].shape[1])


if __name__ == "__main__":
    main()
