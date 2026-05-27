from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
import random
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agents import PPOActorCritic, build_action_masks, decode_action, encode_observation
from agents.action_space import explain_action_mask
from combat import (
    ActionResult,
    ActionSurgeAction,
    CastSpellAction,
    ChannelDivinityPreserveLifeAction,
    CombatAction,
    CombatState,
    CoverType,
    EndTurnAction,
    MoveAction,
    Position,
    SecondWindAction,
    Team,
    UseObjectAction,
    spell_cast_level,
)
from combat.evaluation_scenarios import (
    EvaluationScenario,
    get_evaluation_scenarios,
    get_evaluation_scenarios_by_level,
    get_scenario,
    scenario_names,
)
from scripts.run_demo import (
    DEFAULT_CHECKPOINT,
    fit_masks_for_model,
    load_ppo_checkpoint,
    resolve_checkpoint_path,
)


DEFAULT_EPISODES = 5
DEFAULT_MAX_STEPS = 200


@dataclass
class EpisodeEvaluation:
    """Collected metrics for one scenario episode."""

    scenario_name: str
    level: int
    total_reward: float = 0.0
    xp_gained: int = 0
    winner: Team | None = None
    steps: int = 0
    resource_usage: Counter[str] = field(default_factory=Counter)
    spell_slots_used_by_level: Counter[int] = field(default_factory=Counter)
    class_features_usage: Counter[str] = field(default_factory=Counter)
    item_usage: Counter[str] = field(default_factory=Counter)
    cover_usage: int = 0
    movement_cost: int = 0
    movement_actions: int = 0
    deaths: Counter[str] = field(default_factory=Counter)
    action_distribution: Counter[str] = field(default_factory=Counter)
    masked_allowed: int = 0
    masked_blocked: int = 0
    masked_block_reasons: Counter[str] = field(default_factory=Counter)
    decode_errors: int = 0


@dataclass
class ScenarioEvaluation:
    """Aggregated metrics for one named scenario."""

    scenario: EvaluationScenario
    episodes: list[EpisodeEvaluation] = field(default_factory=list)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    model, model_source = load_or_create_model(
        checkpoint_path,
        require_checkpoint=args.require_checkpoint,
    )
    scenarios = select_scenarios(
        scenario_names_arg=args.scenario,
        by_level=args.by_level,
    )
    results = run_evaluation(
        model,
        scenarios,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        deterministic=not args.stochastic,
    )
    print(f"Policy source: {model_source}")
    print(format_evaluation_report(results, by_level=args.by_level))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a PPO policy on fixed D&D-like combat scenarios.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Path to a trained PPO checkpoint. Missing paths use a fresh model by default.",
    )
    parser.add_argument(
        "--require-checkpoint",
        action="store_true",
        help="Fail instead of using a fresh model when --checkpoint is missing.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        help="Episodes per selected scenario.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Maximum combat steps per episode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed for initiatives and stochastic policies.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=scenario_names(),
        help="Scenario name to evaluate. May be passed multiple times.",
    )
    parser.add_argument(
        "--by-level",
        action="store_true",
        help="Run scenarios grouped across levels 1-5.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample policy actions instead of using deterministic argmax actions.",
    )
    args = parser.parse_args(argv)
    if args.episodes <= 0:
        parser.error("--episodes must be greater than zero")
    if args.max_steps <= 0:
        parser.error("--max-steps must be greater than zero")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def load_or_create_model(
    checkpoint_path: Path,
    *,
    require_checkpoint: bool = False,
) -> tuple[PPOActorCritic, str]:
    """Load a PPO checkpoint or create a fresh model for pipeline evaluation."""

    if checkpoint_path.exists():
        return load_ppo_checkpoint(checkpoint_path), str(checkpoint_path)
    if require_checkpoint:
        raise FileNotFoundError(f"PPO checkpoint not found: {checkpoint_path}")
    model = PPOActorCritic()
    model.eval()
    return model, "fresh PPOActorCritic (checkpoint missing)"


def select_scenarios(
    *,
    scenario_names_arg: Sequence[str] | None = None,
    by_level: bool = False,
) -> tuple[EvaluationScenario, ...]:
    """Resolve CLI scenario selection."""

    if scenario_names_arg:
        selected: list[EvaluationScenario] = []
        for name in scenario_names_arg:
            scenario = get_scenario(name)
            if scenario is None:
                raise ValueError(f"Unknown evaluation scenario: {name}")
            selected.append(scenario)
        return tuple(selected)

    if by_level:
        grouped = get_evaluation_scenarios_by_level()
        selected = []
        for level in range(1, 6):
            selected.extend(grouped[level])
        return tuple(selected)

    return get_evaluation_scenarios()


def run_evaluation(
    model: PPOActorCritic,
    scenarios: Sequence[EvaluationScenario],
    *,
    episodes: int = DEFAULT_EPISODES,
    max_steps: int = DEFAULT_MAX_STEPS,
    seed: int = 0,
    deterministic: bool = True,
) -> list[ScenarioEvaluation]:
    """Run all selected scenarios and return aggregate metrics."""

    results: list[ScenarioEvaluation] = []
    for scenario_index, scenario in enumerate(scenarios):
        scenario_result = ScenarioEvaluation(scenario=scenario)
        for episode_index in range(episodes):
            episode_seed = seed + scenario_index * 1000 + episode_index
            episode = run_episode(
                model,
                scenario,
                max_steps=max_steps,
                seed=episode_seed,
                deterministic=deterministic,
            )
            scenario_result.episodes.append(episode)
        results.append(scenario_result)
    return results


def run_episode(
    model: PPOActorCritic,
    scenario: EvaluationScenario,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    seed: int = 0,
    deterministic: bool = True,
) -> EpisodeEvaluation:
    """Run one evaluation episode for a scenario."""

    seed_everything(seed)
    environment = scenario.create_environment(
        initiative_seed=seed,
        log_to_console=False,
    )
    metrics = EpisodeEvaluation(
        scenario_name=scenario.name,
        level=scenario.level,
    )
    initial_player_xp = _team_experience(environment.combat_state, Team.PLAYERS)

    for _ in range(max_steps):
        if environment.is_done():
            break

        state = environment.combat_state
        actor_id = state.active_actor_id
        if actor_id is None:
            break

        _record_mask_statistics(metrics, state, actor_id)
        action, decode_error = _policy_action(
            model,
            state,
            actor_id,
            deterministic=deterministic,
        )
        if decode_error:
            metrics.decode_errors += 1
        movement_cost = _planned_movement_cost(state, actor_id, action)
        cover_used = _actor_has_cover_from_enemy(state, actor_id)
        result = environment.step(action)
        _record_step_metrics(
            metrics,
            action,
            result,
            movement_cost=movement_cost,
            cover_used=cover_used,
        )

    final_state = environment.combat_state
    metrics.winner = environment.get_winner()
    metrics.xp_gained = max(
        0,
        _team_experience(final_state, Team.PLAYERS) - initial_player_xp,
    )
    _record_deaths(metrics, final_state)
    return metrics


def format_evaluation_report(
    results: Sequence[ScenarioEvaluation],
    *,
    by_level: bool = False,
) -> str:
    """Format evaluation metrics for console output."""

    lines: list[str] = []
    if by_level:
        lines.append("Evaluation mode: by-level")

    for scenario_result in results:
        summary = summarize_scenario(scenario_result)
        lines.append(
            f"Scenario: {scenario_result.scenario.name} "
            f"(level {scenario_result.scenario.level})"
        )
        lines.append(f"  episodes={summary['episodes']}")
        lines.append(f"  win rate={summary['win_rate']:.3f}")
        lines.append(f"  average reward={summary['average_reward']:.3f}")
        lines.append(f"  average XP gained={summary['average_xp_gained']:.2f}")
        lines.append(
            "  average resource usage="
            f"{_format_average_counter(summary['resource_usage'], summary['episodes'])}"
        )
        lines.append(
            "  spell slots used by level="
            f"{_format_average_counter(summary['spell_slots_used_by_level'], summary['episodes'])}"
        )
        lines.append(
            "  class features usage="
            f"{_format_average_counter(summary['class_features_usage'], summary['episodes'])}"
        )
        lines.append(
            "  item usage="
            f"{_format_average_counter(summary['item_usage'], summary['episodes'])}"
        )
        lines.append(f"  cover usage={summary['cover_usage']:.2f}")
        lines.append(f"  average movement cost={summary['average_movement_cost']:.2f}")
        lines.append(
            "  deaths="
            f"{_format_deaths(summary['deaths'], summary['episodes'])}"
        )
        lines.append(
            "  action distribution="
            f"{_format_distribution(summary['action_distribution'])}"
        )
        lines.append(
            "  masked action statistics="
            f"{_format_masked_statistics(summary)}"
        )
    return "\n".join(lines)


def summarize_scenario(
    scenario_result: ScenarioEvaluation,
) -> dict[str, object]:
    episodes = scenario_result.episodes
    episode_count = max(1, len(episodes))
    win_count = sum(1 for item in episodes if item.winner is Team.PLAYERS)
    movement_actions = sum(item.movement_actions for item in episodes)
    movement_cost = sum(item.movement_cost for item in episodes)
    masked_allowed = sum(item.masked_allowed for item in episodes)
    masked_blocked = sum(item.masked_blocked for item in episodes)
    decode_errors = sum(item.decode_errors for item in episodes)

    return {
        "episodes": len(episodes),
        "win_rate": win_count / episode_count,
        "average_reward": sum(item.total_reward for item in episodes) / episode_count,
        "average_xp_gained": sum(item.xp_gained for item in episodes) / episode_count,
        "resource_usage": _sum_counters(item.resource_usage for item in episodes),
        "spell_slots_used_by_level": _sum_counters(
            item.spell_slots_used_by_level for item in episodes
        ),
        "class_features_usage": _sum_counters(
            item.class_features_usage for item in episodes
        ),
        "item_usage": _sum_counters(item.item_usage for item in episodes),
        "cover_usage": sum(item.cover_usage for item in episodes) / episode_count,
        "average_movement_cost": (
            movement_cost / movement_actions
            if movement_actions > 0
            else 0.0
        ),
        "deaths": _sum_counters(item.deaths for item in episodes),
        "action_distribution": _sum_counters(
            item.action_distribution for item in episodes
        ),
        "masked_allowed": masked_allowed,
        "masked_blocked": masked_blocked,
        "masked_block_reasons": _sum_counters(
            item.masked_block_reasons for item in episodes
        ),
        "decode_errors": decode_errors,
    }


def _policy_action(
    model: PPOActorCritic,
    state: CombatState,
    actor_id: int,
    *,
    deterministic: bool,
) -> tuple[CombatAction, bool]:
    observation = fit_observation_for_model(encode_observation(state, actor_id), model)
    masks = fit_masks_for_model(build_action_masks(state, actor_id), model)
    with torch.no_grad():
        model_action = model.act(observation, masks, deterministic=deterministic)

    try:
        return decode_action(
            int(model_action["action_category"].item()),
            int(model_action["main_action_type"].item()),
            int(model_action["target_index"].item()),
            int(model_action["move_index"].item()),
            int(model_action["option_index"].item()),
            state,
            actor_id,
        ), False
    except ValueError:
        return EndTurnAction(actor_id=actor_id), True


def fit_observation_for_model(
    observation: torch.Tensor,
    model: PPOActorCritic,
) -> torch.Tensor:
    """Pad or truncate observations for older checkpoints with smaller encoders."""

    expected_size = int(model.observation_size)
    if observation.shape[0] == expected_size:
        return observation
    if observation.shape[0] > expected_size:
        return observation[:expected_size]
    padding = torch.zeros(
        expected_size - observation.shape[0],
        dtype=observation.dtype,
        device=observation.device,
    )
    return torch.cat((observation, padding), dim=0)


def _record_mask_statistics(
    metrics: EpisodeEvaluation,
    state: CombatState,
    actor_id: int,
) -> None:
    explanations = explain_action_mask(state, actor_id)
    for explanation in explanations:
        if bool(explanation.get("allowed", False)):
            metrics.masked_allowed += 1
        else:
            metrics.masked_blocked += 1
            metrics.masked_block_reasons.update([str(explanation.get("reason", "blocked"))])


def _planned_movement_cost(
    state: CombatState,
    actor_id: int,
    action: CombatAction,
) -> int | None:
    if not isinstance(action, MoveAction) or state.grid_map is None:
        return None
    actor = state.character_at(actor_id)
    if actor is None:
        return None
    return state.grid_map.path_movement_cost(
        actor.position,
        action.destination,
        state.characters,
    )


def _actor_has_cover_from_enemy(state: CombatState, actor_id: int) -> bool:
    if state.grid_map is None:
        return False
    actor = state.character_at(actor_id)
    if actor is None or actor.is_dead:
        return False
    for enemy in state.characters:
        if enemy.team == actor.team or enemy.is_dead:
            continue
        cover = state.grid_map.get_cover_between(enemy.position, actor.position)
        if cover is not CoverType.NO_COVER:
            return True
    return False


def _record_step_metrics(
    metrics: EpisodeEvaluation,
    action: CombatAction,
    result: ActionResult,
    *,
    movement_cost: int | None,
    cover_used: bool,
) -> None:
    metrics.steps += 1
    metrics.total_reward += result.reward
    metrics.action_distribution.update([_action_label(action)])
    if result.success and movement_cost is not None:
        metrics.movement_cost += movement_cost
        metrics.movement_actions += 1
    if result.success and cover_used:
        metrics.cover_usage += 1

    if result.success:
        _record_resource_usage(metrics, action)


def _record_resource_usage(
    metrics: EpisodeEvaluation,
    action: CombatAction,
) -> None:
    if isinstance(action, CastSpellAction):
        spell = action.spell
        if spell is not None and spell.spell_level > 0:
            slot_level = spell_cast_level(spell, action.cast_level)
            metrics.spell_slots_used_by_level.update([slot_level])
            metrics.resource_usage.update([f"spell_slot_level_{slot_level}"])
        return

    if isinstance(action, SecondWindAction):
        metrics.class_features_usage.update(["Second Wind"])
        metrics.resource_usage.update(["Second Wind"])
        return

    if isinstance(action, ActionSurgeAction):
        metrics.class_features_usage.update(["Action Surge"])
        metrics.resource_usage.update(["Action Surge"])
        return

    if isinstance(action, ChannelDivinityPreserveLifeAction):
        metrics.class_features_usage.update(["Channel Divinity: Preserve Life"])
        metrics.resource_usage.update(["Channel Divinity"])
        return

    if isinstance(action, UseObjectAction):
        item_name = getattr(action.item, "name", None) or action.object_name
        metrics.item_usage.update([item_name])
        metrics.resource_usage.update([item_name])


def _record_deaths(metrics: EpisodeEvaluation, state: CombatState) -> None:
    player_deaths = sum(
        1 for character in state.characters
        if character.team is Team.PLAYERS and character.is_dead
    )
    enemy_deaths = sum(
        1 for character in state.characters
        if character.team is Team.ENEMIES and character.is_dead
    )
    metrics.deaths.update(
        {
            "players": player_deaths,
            "enemies": enemy_deaths,
        }
    )


def _team_experience(state: CombatState, team: Team) -> int:
    return sum(
        int(getattr(character, "experience", 0))
        for character in state.characters
        if character.team is team
    )


def _action_label(action: CombatAction) -> str:
    if isinstance(action, CastSpellAction) and action.spell is not None:
        return f"CastSpell:{action.spell.name}"
    if isinstance(action, UseObjectAction):
        item_name = getattr(action.item, "name", None) or action.object_name
        return f"UseObject:{item_name}"
    if isinstance(action, MoveAction):
        destination = getattr(action, "destination", Position())
        return f"Move:{destination.x},{destination.y}"
    return action.__class__.__name__


def _sum_counters(counters: Iterable[Counter]) -> Counter:
    total: Counter = Counter()
    for counter in counters:
        total.update(counter)
    return total


def _format_average_counter(counter: Counter, episodes: int) -> str:
    if not counter:
        return "{}"
    divisor = max(1, episodes)
    parts = [
        f"{key}:{counter[key] / divisor:.2f}"
        for key in sorted(counter, key=lambda value: str(value))
        if counter[key] != 0
    ]
    return "{" + ", ".join(parts) + "}"


def _format_distribution(counter: Counter) -> str:
    total = sum(counter.values())
    if total <= 0:
        return "{}"
    parts = [
        f"{key}:{counter[key] / total:.3f}"
        for key in sorted(counter, key=lambda value: str(value))
        if counter[key] > 0
    ]
    return "{" + ", ".join(parts) + "}"


def _format_deaths(counter: Counter, episodes: int) -> str:
    divisor = max(1, episodes)
    players = counter.get("players", 0) / divisor
    enemies = counter.get("enemies", 0) / divisor
    return f"{{players:{players:.2f}, enemies:{enemies:.2f}}}"


def _format_masked_statistics(summary: dict[str, object]) -> str:
    allowed = int(summary["masked_allowed"])
    blocked = int(summary["masked_blocked"])
    total = allowed + blocked
    blocked_ratio = blocked / total if total > 0 else 0.0
    reason_counter = summary["masked_block_reasons"]
    if not isinstance(reason_counter, Counter):
        reason_counter = Counter()
    top_reasons = ", ".join(
        f"{reason}:{count}"
        for reason, count in reason_counter.most_common(5)
    )
    if not top_reasons:
        top_reasons = "none"
    return (
        f"allowed={allowed} "
        f"blocked={blocked} "
        f"blocked_ratio={blocked_ratio:.3f} "
        f"decode_errors={int(summary['decode_errors'])} "
        f"top_blocked_reasons={{{top_reasons}}}"
    )


if __name__ == "__main__":
    main()
