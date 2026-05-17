import pytest

from combat.checks import (
    ability_modifier,
    passive_perception,
    roll_ability_check,
    roll_contested_check,
)
from combat import Character, Position, Stats, Team


def make_character(
    name: str = "Hero",
    stats: Stats | None = None,
    proficiency_bonus: int = 2,
) -> Character:
    return Character(
        name=name,
        hp=10,
        max_hp=10,
        ac=12,
        position=Position(0, 0),
        speed=3,
        stats=stats or Stats(),
        team=Team.PLAYERS,
        proficiency_bonus=proficiency_bonus,
    )


def test_ability_modifier_uses_raw_score() -> None:
    assert ability_modifier(8) == -1
    assert ability_modifier(10) == 0
    assert ability_modifier(18) == 4


def test_roll_ability_check_logs_roll_components(monkeypatch) -> None:
    hero = make_character(stats=Stats(dex=16), proficiency_bonus=2)
    monkeypatch.setattr("combat.checks.random.randint", lambda _low, _high: 12)

    result = roll_ability_check(hero, "stealth", proficiency=True)

    assert result.total == 17
    assert result.rolls == (12,)
    assert result.kept_roll == 12
    assert "Stealth" in result.log
    assert "dex_mod=3" in result.log
    assert "proficiency 2" in result.log


def test_roll_ability_check_supports_advantage(monkeypatch) -> None:
    hero = make_character(stats=Stats(wis=14), proficiency_bonus=2)
    rolls = iter([3, 17])
    monkeypatch.setattr("combat.checks.random.randint", lambda _low, _high: next(rolls))

    result = roll_ability_check(
        hero,
        "perception",
        proficiency=True,
        advantage_state="advantage",
    )

    assert result.rolls == (3, 17)
    assert result.kept_roll == 17
    assert result.total == 21


def test_roll_contested_check_uses_best_target_option(monkeypatch) -> None:
    actor = make_character("Actor", stats=Stats(str=18), proficiency_bonus=2)
    target = make_character("Target", stats=Stats(str=8, dex=18), proficiency_bonus=2)
    rolls = iter([10, 6, 12])
    monkeypatch.setattr("combat.checks.random.randint", lambda _low, _high: next(rolls))

    contest = roll_contested_check(
        actor,
        target,
        "athletics",
        ("athletics", "acrobatics"),
    )

    assert contest.actor_result.total == 16
    assert contest.target_result.check_name == "Acrobatics"
    assert contest.target_result.total == 18
    assert contest.actor_wins is False
    assert "best of" in contest.log


def test_passive_perception_uses_wisdom_and_proficiency() -> None:
    character = make_character(stats=Stats(wis=14), proficiency_bonus=3)

    assert passive_perception(character) == 15
    assert passive_perception(character, proficiency=False) == 12


def test_roll_ability_check_rejects_unknown_check() -> None:
    character = make_character()

    with pytest.raises(ValueError, match="Unknown ability or skill"):
        roll_ability_check(character, "alchemy")
