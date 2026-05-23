import combat.initiative as initiative_module
from combat import (
    Character,
    CombatEnvironment,
    Condition,
    EndTurnAction,
    GridMap,
    InitiativeCheckResult,
    Position,
    Stats,
    Team,
)


def make_character(
    name: str,
    team: Team,
    dex: int = 10,
    hp: int = 10,
    incapacitated: bool = False,
) -> Character:
    conditions = [Condition("incapacitated")] if incapacitated else []
    return Character(
        name=name,
        hp=hp,
        max_hp=10,
        ac=12,
        position=Position(0, 0),
        speed=3,
        stats=Stats(dex=dex),
        team=team,
        conditions=conditions,
    )


def patch_initiative_totals(monkeypatch, totals: dict[str, int]) -> None:
    def fake_roll_initiative_check(character: Character, rng=None) -> InitiativeCheckResult:
        total = totals[character.name]
        return InitiativeCheckResult(
            character_name=character.name,
            roll=10,
            dex_modifier=total - 10,
            total=total,
        )

    monkeypatch.setattr(
        initiative_module,
        "roll_initiative_check",
        fake_roll_initiative_check,
    )


def test_initiative_order_uses_descending_totals(monkeypatch) -> None:
    characters = [
        make_character("Slow", Team.PLAYERS),
        make_character("Fast", Team.ENEMIES),
        make_character("Middle", Team.PLAYERS),
    ]
    patch_initiative_totals(
        monkeypatch,
        {"Slow": 8, "Fast": 18, "Middle": 12},
    )

    environment = CombatEnvironment(
        characters=characters,
        grid_map=GridMap(width=4, height=4),
        use_initiative=True,
        seed=7,
        log_to_console=False,
    )

    assert environment.initiative_order == [1, 2, 0]
    assert environment.current_turn_index == 0
    assert environment.round_number == 1
    assert environment.combat_state.turn_index == 1
    assert environment.combat_state.active_character is not None
    assert environment.combat_state.active_character.name == "Fast"
    assert environment.combat_state.initiative_totals == {0: 8, 1: 18, 2: 12}
    assert any("Initiative roll: Fast" in entry for entry in environment.action_log)
    assert any("Initiative order: Fast(18), Middle(12), Slow(8)." in entry for entry in environment.action_log)
    assert "Round 1 begins." in environment.action_log
    assert "Active actor: Fast." in environment.action_log


def test_initiative_tie_breaker_is_reproducible_with_seed(monkeypatch) -> None:
    characters = [
        make_character("Alpha", Team.PLAYERS),
        make_character("Beta", Team.ENEMIES),
        make_character("Gamma", Team.PLAYERS),
        make_character("Delta", Team.ENEMIES),
    ]
    patch_initiative_totals(
        monkeypatch,
        {"Alpha": 10, "Beta": 10, "Gamma": 10, "Delta": 10},
    )

    first = CombatEnvironment(
        characters=characters,
        grid_map=GridMap(width=4, height=4),
        use_initiative=True,
        seed=123,
        log_to_console=False,
    )
    second = CombatEnvironment(
        characters=characters,
        grid_map=GridMap(width=4, height=4),
        use_initiative=True,
        seed=123,
        log_to_console=False,
    )

    assert first.initiative_order == second.initiative_order
    assert first.combat_state.initiative_tie_breakers == second.combat_state.initiative_tie_breakers


def test_dead_creature_is_skipped_in_initiative_order(monkeypatch) -> None:
    characters = [
        make_character("Hero", Team.PLAYERS),
        make_character("Dead", Team.ENEMIES, hp=0),
        make_character("Living", Team.ENEMIES),
    ]
    patch_initiative_totals(
        monkeypatch,
        {"Hero": 20, "Dead": 15, "Living": 10},
    )
    environment = CombatEnvironment(
        characters=characters,
        grid_map=GridMap(width=4, height=4),
        use_initiative=True,
        seed=1,
        log_to_console=False,
    )

    result = environment.step(EndTurnAction(actor_id=0))

    assert result.success
    assert environment.combat_state.turn_index == 2
    assert environment.current_turn_index == 2
    assert environment.combat_state.active_character is not None
    assert environment.combat_state.active_character.name == "Living"
    assert "Dead is dead and skips turn." in environment.action_log


def test_incapacitated_creature_is_skipped_in_initiative_order(monkeypatch) -> None:
    characters = [
        make_character("Hero", Team.PLAYERS),
        make_character("Stunned", Team.ENEMIES, incapacitated=True),
        make_character("Living", Team.ENEMIES),
    ]
    patch_initiative_totals(
        monkeypatch,
        {"Hero": 20, "Stunned": 15, "Living": 10},
    )
    environment = CombatEnvironment(
        characters=characters,
        grid_map=GridMap(width=4, height=4),
        use_initiative=True,
        seed=1,
        log_to_console=False,
    )

    result = environment.step(EndTurnAction(actor_id=0))

    assert result.success
    assert environment.combat_state.turn_index == 2
    assert "Stunned is incapacitated and skips turn." in environment.action_log


def test_round_number_increases_after_full_initiative_cycle(monkeypatch) -> None:
    characters = [
        make_character("Hero", Team.PLAYERS),
        make_character("Enemy", Team.ENEMIES),
    ]
    patch_initiative_totals(monkeypatch, {"Hero": 20, "Enemy": 10})
    environment = CombatEnvironment(
        characters=characters,
        grid_map=GridMap(width=4, height=4),
        use_initiative=True,
        seed=1,
        log_to_console=False,
    )

    first = environment.step(EndTurnAction(actor_id=0))
    second = environment.step(EndTurnAction(actor_id=1))

    assert first.success
    assert second.success
    assert environment.round_number == 2
    assert environment.combat_state.round_number == 2
    assert environment.current_turn_index == 0
    assert environment.combat_state.turn_index == 0
    assert "Round 2 begins." in environment.action_log
