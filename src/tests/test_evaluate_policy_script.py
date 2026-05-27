import torch

from agents import PPOActorCritic
from combat.evaluation_scenarios import get_scenario
from scripts.evaluate_policy import (
    fit_observation_for_model,
    format_evaluation_report,
    run_evaluation,
    select_scenarios,
)


def test_select_scenarios_by_level_covers_one_to_five() -> None:
    scenarios = select_scenarios(by_level=True)

    assert {scenario.level for scenario in scenarios} == {1, 2, 3, 4, 5}


def test_evaluate_policy_report_contains_requested_metrics() -> None:
    scenario = get_scenario("Level 1 Fighter vs 1 Goblin")
    model = PPOActorCritic(
        observation_size=180,
        target_count=4,
        move_count=36,
        option_count=8,
        hidden_sizes=(16,),
    )

    results = run_evaluation(
        model,
        (scenario,),
        episodes=1,
        max_steps=1,
        seed=0,
    )
    report = format_evaluation_report(results)

    assert "win rate=" in report
    assert "average reward=" in report
    assert "average XP gained=" in report
    assert "average resource usage=" in report
    assert "spell slots used by level=" in report
    assert "class features usage=" in report
    assert "item usage=" in report
    assert "cover usage=" in report
    assert "average movement cost=" in report
    assert "deaths=" in report
    assert "action distribution=" in report
    assert "masked action statistics=" in report


def test_fit_observation_for_older_checkpoint_shapes() -> None:
    model = PPOActorCritic(
        observation_size=4,
        target_count=4,
        move_count=16,
        option_count=8,
        hidden_sizes=(8,),
    )
    observation = torch.arange(8, dtype=torch.float32)

    fitted = fit_observation_for_model(observation, model)

    assert fitted.shape == (4,)
    assert fitted.tolist() == [0.0, 1.0, 2.0, 3.0]
