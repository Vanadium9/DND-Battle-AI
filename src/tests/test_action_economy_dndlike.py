from combat import (
    Character,
    CombatEnvironment,
    CombatState,
    DisengageAction,
    DodgeAction,
    EndTurnAction,
    GrappleAction,
    GridMap,
    MoveAction,
    Position,
    ReadyAction,
    Stats,
    Team,
    UseObjectAction,
)


def make_character(
    name: str,
    position: Position,
    team: Team,
    speed: int = 4,
    stats: Stats | None = None,
) -> Character:
    return Character(
        name=name,
        hp=10,
        max_hp=10,
        ac=12,
        position=position,
        speed=speed,
        stats=stats or Stats(),
        team=team,
    )


def test_action_economy_has_full_turn_resources() -> None:
    hero = make_character("Hero", Position(0, 0), Team.PLAYERS)
    economy = hero.action_economy

    economy.action_available = False
    economy.bonus_action_available = False
    economy.reaction_available = False
    economy.free_object_interaction_available = False
    economy.movement_remaining = 0
    economy.reset_for_turn(hero.speed)

    assert economy.action_available is True
    assert economy.bonus_action_available is True
    assert economy.reaction_available is True
    assert economy.free_object_interaction_available is True
    assert economy.movement_remaining == hero.speed


def test_ready_reserves_reaction_and_reaction_can_be_spent() -> None:
    hero = make_character("Hero", Position(0, 0), Team.PLAYERS)
    state = CombatState(characters=[hero], grid_map=GridMap(width=3, height=3))

    result = ReadyAction(
        actor_id=0,
        prepared_action="attack",
        trigger_description="enemy enters reach",
    ).execute(state)

    assert result.success
    assert hero.action_economy.action_available is False
    assert hero.action_economy.reaction_available is True
    assert hero.prepared_action == "attack"
    assert hero.trigger_description == "enemy enters reach"

    hero.action_economy.spend_reaction()

    assert hero.action_economy.reaction_available is False
    assert hero.action_economy.reaction_used_this_round is True
    assert not ReadyAction(actor_id=0).is_valid(state)


def test_dodge_and_disengage_expire_at_correct_turn_boundaries() -> None:
    hero = make_character("Hero", Position(0, 0), Team.PLAYERS)
    enemy = make_character("Enemy", Position(1, 0), Team.ENEMIES)
    environment = CombatEnvironment(
        characters=[hero, enemy],
        grid_map=GridMap(width=3, height=3),
        log_to_console=False,
    )

    environment.step(DodgeAction(actor_id=0))
    assert environment.combat_state.characters[0].dodging_until_start_of_next_turn is True

    environment.step(EndTurnAction(actor_id=0))
    environment.step(EndTurnAction(actor_id=1))

    assert environment.combat_state.characters[0].dodging_until_start_of_next_turn is False

    environment.step(DisengageAction(actor_id=0))
    assert environment.combat_state.characters[0].disengaged_until_end_of_turn is True

    environment.step(EndTurnAction(actor_id=0))

    assert environment.combat_state.characters[0].disengaged_until_end_of_turn is False


def test_prone_standing_costs_half_speed_before_moving() -> None:
    hero = make_character("Hero", Position(0, 0), Team.PLAYERS, speed=4)
    state = CombatState(characters=[hero], grid_map=GridMap(width=4, height=4))
    hero.prone = True
    hero.action_economy.movement_remaining = 4

    result = MoveAction(actor_id=0, destination=Position(1, 0)).execute(state)

    assert result.success
    assert hero.prone is False
    assert hero.position == Position(1, 0)
    assert hero.action_economy.movement_remaining == 1


def test_grapple_sets_grappled_state_and_zeroes_movement(monkeypatch) -> None:
    hero = make_character(
        "Hero",
        Position(0, 0),
        Team.PLAYERS,
        stats=Stats(str=18),
    )
    enemy = make_character(
        "Enemy",
        Position(1, 0),
        Team.ENEMIES,
        stats=Stats(str=8, dex=8),
    )
    state = CombatState(characters=[hero, enemy], grid_map=GridMap(width=3, height=3))
    rolls = iter([20, 1, 1])
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: next(rolls))

    result = GrappleAction(actor_id=0, target_id=1).execute(state)

    assert result.success
    assert hero.grappling_target_id == 1
    assert enemy.grappled is True
    assert enemy.grappled_by == 0
    assert enemy.action_economy.movement_remaining == 0

    state.turn_index = 1
    state.reset_turn_resources(actor_id=1)

    assert enemy.action_economy.movement_remaining == 0


def test_use_object_spends_action_and_free_object_interaction() -> None:
    hero = make_character("Hero", Position(0, 0), Team.PLAYERS)
    state = CombatState(characters=[hero], grid_map=GridMap(width=3, height=3))

    result = UseObjectAction(actor_id=0, object_name="lever").execute(state)

    assert result.success
    assert hero.action_economy.action_available is False
    assert hero.action_economy.free_object_interaction_available is False
