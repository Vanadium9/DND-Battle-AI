from combat import (
    CombatEnvironment,
    CombatState,
    EncounterGenerator,
    Team,
)


def encounter_summary(combat_state: CombatState) -> tuple[tuple[str, str, int, int], ...]:
    return tuple(
        (
            character.name,
            character.team.value,
            character.position.x,
            character.position.y,
        )
        for character in combat_state.characters
    )


def test_encounter_generator_creates_valid_state() -> None:
    generator = EncounterGenerator(seed=42)

    combat_state = generator.generate_state()

    assert combat_state.grid_map is not None
    assert combat_state.grid_map.width == 8
    assert combat_state.grid_map.height == 8

    players = combat_state.characters_for_team(Team.PLAYERS)
    enemies = combat_state.characters_for_team(Team.ENEMIES)
    positions = [character.position for character in combat_state.characters]

    assert 1 <= len(players) <= 2
    assert 1 <= len(enemies) <= 4
    assert len(positions) == len(set(positions))
    assert all(combat_state.grid_map.in_bounds(position) for position in positions)
    assert {character.name for character in players}.issubset(
        {"Fighter Champion Greatsword", "Fighter Archer"}
    )
    assert {character.name for character in enemies}.issubset({"Goblin", "Orc"})


def test_encounter_generator_seed_is_deterministic() -> None:
    first = EncounterGenerator(seed=7).generate_state()
    second = EncounterGenerator(seed=7).generate_state()
    different = EncounterGenerator(seed=8).generate_state()

    assert encounter_summary(first) == encounter_summary(second)
    assert encounter_summary(first) != encounter_summary(different)


def test_encounter_generator_can_return_environment() -> None:
    generator = EncounterGenerator(seed=5)

    environment = generator.generate_environment(log_to_console=False)

    assert isinstance(environment, CombatEnvironment)
    assert environment.combat_state.grid_map is not None
    assert environment.combat_state.grid_map.width == 8
    assert not environment.is_done()


def test_encounter_generator_generate_switches_return_type() -> None:
    generator = EncounterGenerator(seed=11)

    combat_state = generator.generate()
    environment = generator.generate(as_environment=True, log_to_console=False)

    assert isinstance(combat_state, CombatState)
    assert isinstance(environment, CombatEnvironment)
