from combat import (
    CLASS_CURRICULUM_STAGES,
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
    assert len(CURRICULUM_STAGES) == 13
    assert MAX_CURRICULUM_LEVEL == 13
    assert [stage.level for stage in CURRICULUM_STAGES] == list(range(1, 14))
    stage_names = " ".join(stage.name for stage in CURRICULUM_STAGES)
    assert "Cleric" in stage_names
    assert "Wizard" in stage_names
    assert "FireElementalSimple" in stage_names


def test_curriculum_level_generates_fixed_encounter() -> None:
    generator = EncounterGenerator(seed=3, curriculum_level=2)

    combat_state = generator.generate_state()
    players = combat_state.characters_for_team(Team.PLAYERS)
    enemies = combat_state.characters_for_team(Team.ENEMIES)

    assert len(players) == 2
    assert {player.class_name for player in players} == {"Fighter", "Cleric"}
    assert len(enemies) == 2
    assert {enemy.name for enemy in enemies} == {"Goblin Melee", "Goblin Archer"}


def test_curriculum_map_levels_include_cover_obstacles_and_difficult_terrain() -> None:
    generator = EncounterGenerator(seed=4)

    cover_state = generator.generate_curriculum_state(12)
    difficult_state = generator.generate_curriculum_state(13)
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


def test_wizard_curriculum_intro_is_not_outnumbered_by_archers() -> None:
    generator = EncounterGenerator(seed=4)

    intro_state = generator.generate_curriculum_state(7)
    aoe_state = generator.generate_curriculum_state(8)

    intro_players = intro_state.characters_for_team(Team.PLAYERS)
    intro_enemies = intro_state.characters_for_team(Team.ENEMIES)
    assert {player.class_name for player in intro_players} == {"Fighter", "Wizard"}
    assert len(intro_enemies) == 1
    assert all("Archer" not in enemy.name for enemy in intro_enemies)

    aoe_enemies = aoe_state.characters_for_team(Team.ENEMIES)
    assert len(aoe_enemies) == 2
    assert all("Archer" not in enemy.name for enemy in aoe_enemies)


def test_mixed_party_curriculum_intro_limits_enemy_pressure() -> None:
    generator = EncounterGenerator(seed=4)

    mixed_intro = generator.generate_curriculum_state(9)
    players = mixed_intro.characters_for_team(Team.PLAYERS)
    enemies = mixed_intro.characters_for_team(Team.ENEMIES)

    assert {player.class_name for player in players} == {"Fighter", "Cleric", "Wizard"}
    assert len(enemies) == 2
    assert {enemy.name for enemy in enemies} == {"Orc Warrior", "Goblin Melee"}
    assert all("Archer" not in enemy.name for enemy in enemies)


def test_class_curriculum_has_separate_class_and_integration_phases() -> None:
    assert len(CLASS_CURRICULUM_STAGES) == 14
    assert [stage.phase for stage in CLASS_CURRICULUM_STAGES[:3]] == ["fighter"] * 3
    assert [stage.phase for stage in CLASS_CURRICULUM_STAGES[3:6]] == ["cleric"] * 3
    assert [stage.phase for stage in CLASS_CURRICULUM_STAGES[6:9]] == ["wizard"] * 3
    assert all(stage.phase == "integration" for stage in CLASS_CURRICULUM_STAGES[9:])


def test_first_stage_of_new_class_disables_rehearsal() -> None:
    generator = EncounterGenerator(
        seed=4,
        curriculum_level=4,
        curriculum_kind="class",
        rehearsal_probability=1.0,
    )

    state = generator.generate_curriculum_state()

    assert state.curriculum_is_rehearsal is False
    assert state.curriculum_source_level == 4
    assert state.training_classes == ("Cleric",)


def test_cleric_resource_stage_uses_only_melee_goblins() -> None:
    generator = EncounterGenerator(
        seed=5,
        curriculum_level=5,
        curriculum_kind="class",
    )

    state = generator.generate_curriculum_state()
    enemies = state.characters_for_team(Team.ENEMIES)

    assert state.training_classes == ("Cleric",)
    assert len(enemies) == 2
    assert {enemy.name for enemy in enemies} == {"Goblin Melee"}


def test_curriculum_environment_preserves_target_class_metadata() -> None:
    generator = EncounterGenerator(
        seed=6,
        curriculum_level=4,
        curriculum_kind="class",
    )

    environment = generator.generate_curriculum_environment(log_to_console=False)

    assert environment.combat_state.training_classes == ("Cleric",)
    assert environment.combat_state.curriculum_source_level == 4
    assert environment.combat_state.curriculum_is_rehearsal is False
