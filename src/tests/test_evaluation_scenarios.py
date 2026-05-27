from combat import Team
from combat.evaluation_scenarios import (
    get_evaluation_scenarios,
    get_evaluation_scenarios_by_level,
    get_scenario,
)
from combat.terrain import TerrainType


REQUIRED_SCENARIOS = {
    "Level 1 Fighter vs 1 Goblin",
    "Level 1 Fighter + Cleric vs 2 Goblins",
    "Level 3 Fighter Champion vs Orc",
    "Level 3 Cleric Life + Fighter vs Orc + Goblin",
    "Level 5 Wizard Evoker vs 3 Goblins",
    "Level 5 Wizard Evoker vs FireElementalSimple",
    "Level 5 Fighter + Cleric + Wizard vs mixed enemies",
    "Level 5 ranged party on map with cover vs mixed enemies",
    "Level 5 melee party on difficult terrain map vs archers",
}


def test_required_evaluation_scenarios_are_registered() -> None:
    names = {scenario.name for scenario in get_evaluation_scenarios()}

    assert REQUIRED_SCENARIOS.issubset(names)


def test_evaluation_scenarios_create_valid_combat_environments() -> None:
    for scenario in get_evaluation_scenarios():
        environment = scenario.create_environment(
            initiative_seed=0,
            log_to_console=False,
        )
        state = environment.combat_state

        assert state.grid_map is not None
        assert state.active_actor_id is not None
        assert any(character.team is Team.PLAYERS for character in state.characters)
        assert any(character.team is Team.ENEMIES for character in state.characters)


def test_by_level_scenarios_cover_levels_one_to_five() -> None:
    grouped = get_evaluation_scenarios_by_level()

    assert set(grouped) == {1, 2, 3, 4, 5}
    assert all(grouped[level] for level in range(1, 6))


def test_cover_and_difficult_terrain_scenarios_have_expected_maps() -> None:
    cover_state = get_scenario(
        "Level 5 ranged party on map with cover vs mixed enemies"
    ).create_state()
    difficult_state = get_scenario(
        "Level 5 melee party on difficult terrain map vs archers"
    ).create_state()

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

    assert TerrainType.LOW_COVER in cover_terrain
    assert TerrainType.HIGH_COVER in cover_terrain
    assert TerrainType.DIFFICULT_TERRAIN in difficult_terrain
