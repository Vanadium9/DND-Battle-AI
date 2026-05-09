from combat import Character, GridMap, Position, Stats, Team


def make_character(name: str, position: Position, hp: int = 10) -> Character:
    return Character(
        name=name,
        hp=hp,
        max_hp=10,
        ac=12,
        position=position,
        speed=3,
        stats=Stats(),
        team=Team.PLAYERS,
    )


def test_grid_map_bounds_distance_and_neighbors() -> None:
    grid_map = GridMap(width=3, height=2)

    assert grid_map.in_bounds(Position(0, 0))
    assert grid_map.in_bounds(Position(2, 1))
    assert not grid_map.in_bounds(Position(3, 1))
    assert not grid_map.in_bounds(Position(1, -1))
    assert grid_map.manhattan_distance(Position(0, 0), Position(2, 1)) == 3
    assert set(grid_map.neighbors(Position(0, 0))) == {
        Position(1, 0),
        Position(0, 1),
    }


def test_grid_map_occupancy_and_movement_cells() -> None:
    grid_map = GridMap(width=5, height=5)
    hero = make_character("Hero", Position(2, 2))
    blocker = make_character("Blocker", Position(3, 2))
    defeated = make_character("Defeated", Position(2, 3), hp=0)

    characters = [hero, blocker, defeated]

    assert grid_map.is_occupied(Position(3, 2), characters)
    assert not grid_map.is_occupied(Position(2, 3), characters)

    movement_cells = grid_map.movement_cells(hero.position, hero.speed, characters)

    assert Position(2, 2) in movement_cells
    assert Position(2, 3) in movement_cells
    assert Position(3, 2) not in movement_cells
    assert Position(2, 4) in movement_cells
    assert Position(4, 4) not in movement_cells
