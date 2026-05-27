from combat import (
    CURRICULUM_STAGES,
    CombatEnvironment,
    CombatState,
    EncounterGenerator,
    MAX_CURRICULUM_LEVEL,
    Team,
    TerrainType,
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


def test_curriculum_stages_cover_expected_difficulty_levels() -> None:
    assert len(CURRICULUM_STAGES) == 9
    assert MAX_CURRICULUM_LEVEL == 9
    assert [stage.level for stage in CURRICULUM_STAGES] == list(range(1, 10))


def test_curriculum_level_generates_fixed_encounter() -> None:
    generator = EncounterGenerator(seed=3, curriculum_level=2)

    combat_state = generator.generate_state()
    players = combat_state.characters_for_team(Team.PLAYERS)
    enemies = combat_state.characters_for_team(Team.ENEMIES)

    assert len(players) == 1
    assert players[0].name == "Fighter Level 1 Basic"
    assert players[0].level == 1
    assert len(enemies) == 2
    assert {enemy.name for enemy in enemies} == {"Goblin Melee", "Goblin Archer"}


def test_curriculum_map_levels_include_cover_obstacles_and_difficult_terrain() -> None:
    generator = EncounterGenerator(seed=4)

    cover_state = generator.generate_curriculum_state(8)
    difficult_state = generator.generate_curriculum_state(9)
    cover_terrain = {
        terrain
        for row in cover_state.grid_map.terrain_grid
        for terrain in row
    }
    difficult_terrain = {
        terrain
        for row in difficult_state.grid_map.terrain_grid
        for terrain in row
    }

    assert TerrainType.BLOCKED in cover_terrain
    assert TerrainType.LOW_COVER in cover_terrain
    assert TerrainType.HIGH_COVER in cover_terrain
    assert TerrainType.DIFFICULT_TERRAIN in difficult_terrain
