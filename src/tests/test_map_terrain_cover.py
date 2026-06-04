from agents import MainActionType, build_action_masks
from combat import (
    AttackAction,
    Character,
    CombatState,
    CoverType,
    GridMap,
    MoveAction,
    Position,
    Stats,
    Team,
    TerrainType,
    WeaponAttack,
)


def make_character(
    name: str,
    position: Position,
    team: Team,
    weapon: WeaponAttack | None = None,
) -> Character:
    return Character(
        name=name,
        hp=10,
        max_hp=10,
        ac=12,
        position=position,
        speed=3,
        stats=Stats(),
        team=team,
        weapons=[] if weapon is None else [weapon],
    )


def terrain_map(rows: list[list[TerrainType]]) -> GridMap:
    return GridMap(width=len(rows[0]), height=len(rows), terrain_grid=rows)


def test_blocked_cell_is_not_available_for_movement() -> None:
    hero = make_character("Hero", Position(0, 0), Team.PLAYERS)
    state = CombatState(
        characters=[hero],
        grid_map=terrain_map(
            [
                [TerrainType.NORMAL, TerrainType.BLOCKED, TerrainType.NORMAL],
                [TerrainType.NORMAL, TerrainType.NORMAL, TerrainType.NORMAL],
                [TerrainType.NORMAL, TerrainType.NORMAL, TerrainType.NORMAL],
            ]
        ),
    )

    assert not state.grid_map.is_walkable(Position(1, 0))
    assert not MoveAction(actor_id=0, destination=Position(1, 0)).is_valid(state)
    assert Position(1, 0) not in state.grid_map.movement_cells(
        hero.position,
        hero.action_economy.movement_remaining,
        state.characters,
    )


def test_difficult_terrain_spends_extra_movement() -> None:
    hero = make_character("Hero", Position(0, 0), Team.PLAYERS)
    state = CombatState(
        characters=[hero],
        grid_map=terrain_map(
            [
                [
                    TerrainType.NORMAL,
                    TerrainType.DIFFICULT_TERRAIN,
                    TerrainType.NORMAL,
                ],
            ]
        ),
    )

    result = MoveAction(actor_id=0, destination=Position(2, 0)).execute(state)

    assert result.success
    assert "Movement spent: 3" in result.description
    assert hero.position == Position(2, 0)
    assert hero.action_economy.movement_remaining == 0


def test_line_of_sight_is_blocked_by_obstacle() -> None:
    grid_map = terrain_map(
        [[TerrainType.NORMAL, TerrainType.BLOCKED, TerrainType.NORMAL]]
    )

    assert not grid_map.line_of_sight(Position(0, 0), Position(2, 0))


def test_low_cover_does_not_block_line_of_sight() -> None:
    grid_map = terrain_map(
        [[TerrainType.NORMAL, TerrainType.LOW_COVER, TerrainType.NORMAL]]
    )

    assert grid_map.line_of_sight(Position(0, 0), Position(2, 0))


def test_grid_map_reuses_los_cover_and_neighbor_caches() -> None:
    grid_map = GridMap(width=3, height=3)

    assert grid_map.line_of_sight(Position(0, 0), Position(2, 2))
    assert grid_map.get_cover_between(Position(0, 0), Position(2, 2)) is CoverType.NO_COVER
    assert grid_map.neighbor_costs(Position(1, 1))

    assert (0, 0, 2, 2) in grid_map._line_of_sight_cache
    assert (0, 0, 2, 2) in grid_map._cover_cache
    assert (1, 1) in grid_map._neighbor_cost_cache


def test_full_cover_blocks_ranged_attack() -> None:
    bow = WeaponAttack(name="Bow", range=3, damage=1, ability_score="dex")
    hero = make_character("Hero", Position(0, 0), Team.PLAYERS, bow)
    target = make_character("Target", Position(2, 0), Team.ENEMIES)
    state = CombatState(
        characters=[hero, target],
        grid_map=terrain_map(
            [[TerrainType.NORMAL, TerrainType.HIGH_COVER, TerrainType.NORMAL]]
        ),
    )

    assert state.grid_map.get_cover_between(hero.position, target.position).blocks_targeting
    assert not AttackAction(actor_id=0, target_id=1, weapon=bow).is_valid(state)


def test_half_cover_adds_ac_bonus(monkeypatch) -> None:
    bow = WeaponAttack(name="Bow", range=3, damage=1, ability_score="dex")
    hero = make_character("Hero", Position(0, 0), Team.PLAYERS, bow)
    target = make_character("Target", Position(2, 0), Team.ENEMIES)
    state = CombatState(
        characters=[hero, target],
        grid_map=terrain_map(
            [[TerrainType.NORMAL, TerrainType.LOW_COVER, TerrainType.NORMAL]]
        ),
    )
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: 11)

    result = AttackAction(actor_id=0, target_id=1, weapon=bow).execute(state)

    assert result.success
    assert "miss (13 vs AC 14" in result.description
    assert target.hp == target.max_hp


def test_hide_is_masked_without_cover() -> None:
    hero = make_character("Hero", Position(0, 0), Team.PLAYERS)
    enemy = make_character("Enemy", Position(1, 0), Team.ENEMIES)
    state = CombatState(
        characters=[hero, enemy],
        grid_map=GridMap(width=3, height=3),
    )

    masks = build_action_masks(state, actor_id=0)

    assert not masks["main_action_type"][MainActionType.HIDE]
