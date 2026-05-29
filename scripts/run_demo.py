from __future__ import annotations

import argparse
from datetime import datetime
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
    CastSpellAction,
    Character,
    CombatAction,
    CombatEnvironment,
    BattleReplay,
    DashAction,
    DisengageAction,
    DodgeAction,
    EndTurnAction,
    GrappleAction,
    HelpAction,
    HideAction,
    ImprovisedAction,
    MoveAction,
    ReadyAction,
    SearchAction,
    ShoveAction,
    StabilizeAction,
    UseObjectAction,
    create_test_encounter,
)


DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "ppo_actor_critic.pt"
DEFAULT_MAX_STEPS = 200
DEFAULT_REPLAY_DIR = PROJECT_ROOT / "replays"


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    model = load_ppo_checkpoint(checkpoint_path)
    environment = create_demo_environment()

    print(f"Loaded checkpoint: {checkpoint_path}")
    run_battle_demo(
        model,
        environment,
        max_steps=args.max_steps,
        save_replay=args.save_replay,
        replay_metadata={
            "checkpoint": str(checkpoint_path),
            "max_steps": args.max_steps,
        },
    )


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
    parser.add_argument(
        "--save-replay",
        action="store_true",
        help="Save a structured BattleReplay JSON file into replays/.",
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
        option_count=_infer_option_count(state_dict),
        action_category_count=_infer_action_category_count(state_dict),
        main_action_type_count=_infer_main_action_type_count(state_dict),
        hidden_sizes=_infer_hidden_sizes(state_dict),
    )
    if "action_category_head.weight" in state_dict:
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(_compatible_legacy_state_dict(model, state_dict), strict=False)
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
    *,
    save_replay: bool = False,
    replay_dir: Path = DEFAULT_REPLAY_DIR,
    replay_metadata: dict[str, object] | None = None,
) -> Path | None:
    replay = BattleReplay(metadata=replay_metadata or {}) if save_replay else None
    for _ in range(max_steps):
        if environment.is_done():
            break

        state = environment.combat_state
        actor_id = state.active_actor_id
        if actor_id is None:
            break
        actor = state.characters[actor_id]
        observation = encode_observation(state, actor_id)
        masks = fit_masks_for_model(build_action_masks(state, actor_id), model)

        with torch.no_grad():
            model_action = model.act(observation, masks, deterministic=True)

        action = decode_action(
            int(model_action["action_category"].item()),
            int(model_action["main_action_type"].item()),
            int(model_action["target_index"].item()),
            int(model_action["move_index"].item()),
            int(model_action["option_index"].item()),
            state,
            actor_id,
        )

        print(f"Round: {state.round_number}")
        print(f"Actor: {actor.name}")
        print(f"HP: {format_hp(state.characters)}")
        print(f"Action: {describe_action(action, environment)}")
        before_step = replay.snapshot_state(state) if replay is not None else None
        result = environment.step(action)
        if replay is not None and before_step is not None:
            replay.record_step(before_step, environment.combat_state, action, result)
        print(f"Result: {result.description}")
        print("")

    if not environment.is_done():
        print(f"Demo stopped after {max_steps} steps.")

    winner = environment.get_winner()
    winner_text = winner.value if winner is not None else "none"
    print(f"Winner: {winner_text}")
    if replay is None:
        return None

    replay_path = save_demo_replay(replay, replay_dir)
    print(f"Replay saved: {replay_path}")
    return replay_path


def save_demo_replay(
    replay: BattleReplay,
    replay_dir: Path = DEFAULT_REPLAY_DIR,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return replay.save(replay_dir / f"battle_replay_{timestamp}.json")


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
    if isinstance(action, CastSpellAction):
        return "CAST_SPELL"
    if isinstance(action, DashAction):
        return "DASH"
    if isinstance(action, DisengageAction):
        return "DISENGAGE"
    if isinstance(action, DodgeAction):
        return "DODGE"
    if isinstance(action, HelpAction):
        return "HELP"
    if isinstance(action, HideAction):
        return "HIDE"
    if isinstance(action, SearchAction):
        return "SEARCH"
    if isinstance(action, UseObjectAction):
        return "USE_OBJECT"
    if isinstance(action, ReadyAction):
        return "READY"
    if isinstance(action, GrappleAction):
        return "GRAPPLE"
    if isinstance(action, ShoveAction):
        return "SHOVE"
    if isinstance(action, StabilizeAction):
        return "STABILIZE"
    if isinstance(action, ImprovisedAction):
        return "IMPROVISED_ACTION"
    if isinstance(action, EndTurnAction):
        return "END_TURN"
    return action.__class__.__name__


def fit_masks_for_model(
    masks: dict[str, torch.Tensor],
    model: PPOActorCritic,
) -> dict[str, torch.Tensor]:
    return {
        "action_category": _fit_mask(
            masks["action_category"],
            model.action_category_count,
        ),
        "main_action_type": _fit_mask(
            masks["main_action_type"],
            model.main_action_type_count,
        ),
        "target_index": _fit_mask(masks["target_index"], model.target_count),
        "move_index": _fit_mask(masks["move_index"], model.move_count),
        "option_index": _fit_mask(masks["option_index"], model.option_count),
    }


def _fit_mask(mask: torch.Tensor, size: int) -> torch.Tensor:
    if mask.shape[0] == size:
        return mask
    if mask.shape[0] > size:
        return mask[:size]
    padding = torch.zeros(size - mask.shape[0], dtype=torch.bool)
    return torch.cat((mask, padding), dim=0)


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
    for head_name in (
        "action_category_head.weight",
        "action_type_head.weight",
        "main_action_type_head.weight",
    ):
        if head_name in state_dict:
            return int(state_dict[head_name].shape[1])
    raise ValueError("Cannot infer observation size from checkpoint")


def _infer_action_category_count(state_dict: dict[str, torch.Tensor]) -> int:
    if "action_category_head.weight" in state_dict:
        return int(state_dict["action_category_head.weight"].shape[0])
    from agents import ACTION_CATEGORY_COUNT

    return ACTION_CATEGORY_COUNT


def _infer_main_action_type_count(state_dict: dict[str, torch.Tensor]) -> int:
    if "main_action_type_head.weight" in state_dict:
        return int(state_dict["main_action_type_head.weight"].shape[0])
    from agents import MAIN_ACTION_TYPE_COUNT

    return MAIN_ACTION_TYPE_COUNT


def _infer_option_count(state_dict: dict[str, torch.Tensor]) -> int:
    if "option_head.weight" in state_dict:
        return int(state_dict["option_head.weight"].shape[0])
    from agents import DEFAULT_OPTION_COUNT

    return DEFAULT_OPTION_COUNT


def _compatible_legacy_state_dict(
    model: PPOActorCritic,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    model_state = model.state_dict()
    return {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }


if __name__ == "__main__":
    main()
