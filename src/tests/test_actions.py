from combat import (
    AttackAction,
    Character,
    CombatState,
    DisengageAction,
    EndTurnAction,
    GridMap,
    MoveAction,
    Position,
    Stats,
    Team,
    WeaponAttack,
)


def make_character(
    name: str,
    position: Position,
    team: Team,
    weapon: WeaponAttack | None = None,
) -> Character:
    abilities = [] if weapon is None else [weapon]
    return Character(
        name=name,
        hp=10,
        max_hp=10,
        ac=12,
        position=position,
        speed=2,
        stats=Stats(),
        team=team,
        abilities=abilities,
    )


def test_move_action_moves_actor_to_reachable_unoccupied_cell() -> None:
    hero = make_character("Hero", Position(0, 0), Team.PLAYERS)
    blocker = make_character("Blocker", Position(1, 0), Team.ENEMIES)
    combat_state = CombatState(
        characters=[hero, blocker],
        grid_map=GridMap(width=4, height=4),
    )

    valid_move = MoveAction(actor_id=0, destination=Position(0, 2))
    occupied_move = MoveAction(actor_id=0, destination=Position(1, 0))
    distant_move = MoveAction(actor_id=0, destination=Position(3, 3))

    assert valid_move.is_valid(combat_state)
    assert not occupied_move.is_valid(combat_state)
    assert not distant_move.is_valid(combat_state)

    result = valid_move.execute(combat_state)

    assert result.success
    assert "moves" in result.description
    assert "Movement spent: 2" in result.description
    assert hero.position == Position(0, 2)
    assert hero.action_economy.movement_remaining == 0
    assert not MoveAction(actor_id=0, destination=Position(0, 1)).is_valid(combat_state)


def test_move_action_triggers_opportunity_attack_when_leaving_reach(monkeypatch) -> None:
    sword = WeaponAttack(name="Sword", range=1, damage=4, attack_bonus=20)
    hero = make_character("Hero", Position(1, 0), Team.PLAYERS)
    attacker = make_character("Attacker", Position(0, 0), Team.ENEMIES, sword)
    combat_state = CombatState(
        characters=[hero, attacker],
        grid_map=GridMap(width=4, height=4),
    )
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: 10)

    result = MoveAction(actor_id=0, destination=Position(2, 0)).execute(combat_state)

    assert result.success
    assert hero.position == Position(2, 0)
    assert hero.hp == 6
    assert attacker.action_economy.reaction_available is False
    assert "Opportunity attacks" in result.description
    assert "opportunity attacks Hero" in result.description


def test_move_action_does_not_trigger_opportunity_attack_after_disengage(monkeypatch) -> None:
    sword = WeaponAttack(name="Sword", range=1, damage=4, attack_bonus=20)
    hero = make_character("Hero", Position(1, 0), Team.PLAYERS)
    attacker = make_character("Attacker", Position(0, 0), Team.ENEMIES, sword)
    combat_state = CombatState(
        characters=[hero, attacker],
        grid_map=GridMap(width=4, height=4),
    )
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: 10)

    disengage_result = DisengageAction(actor_id=0).execute(combat_state)
    move_result = MoveAction(actor_id=0, destination=Position(2, 0)).execute(combat_state)

    assert disengage_result.success
    assert move_result.success
    assert hero.position == Position(2, 0)
    assert hero.hp == 10
    assert attacker.action_economy.reaction_available is True
    assert "Opportunity attacks" not in move_result.description


def test_move_action_does_not_trigger_opportunity_attack_inside_reach(monkeypatch) -> None:
    sword = WeaponAttack(name="Sword", range=1, damage=4, attack_bonus=20)
    hero = make_character("Hero", Position(1, 0), Team.PLAYERS)
    attacker = make_character("Attacker", Position(0, 0), Team.ENEMIES, sword)
    combat_state = CombatState(
        characters=[hero, attacker],
        grid_map=GridMap(width=4, height=4),
    )
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: 10)

    result = MoveAction(actor_id=0, destination=Position(0, 1)).execute(combat_state)

    assert result.success
    assert hero.position == Position(0, 1)
    assert hero.hp == 10
    assert attacker.action_economy.reaction_available is True
    assert "Opportunity attacks" not in result.description


def test_move_action_does_not_trigger_opportunity_attack_without_reaction(monkeypatch) -> None:
    sword = WeaponAttack(name="Sword", range=1, damage=4, attack_bonus=20)
    hero = make_character("Hero", Position(1, 0), Team.PLAYERS)
    attacker = make_character("Attacker", Position(0, 0), Team.ENEMIES, sword)
    attacker.action_economy.reaction_available = False
    combat_state = CombatState(
        characters=[hero, attacker],
        grid_map=GridMap(width=4, height=4),
    )
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: 10)

    result = MoveAction(actor_id=0, destination=Position(2, 0)).execute(combat_state)

    assert result.success
    assert hero.position == Position(2, 0)
    assert hero.hp == 10
    assert attacker.action_economy.reaction_available is False
    assert "Opportunity attacks" not in result.description


def test_attack_action_hits_with_fixed_damage_in_weapon_range() -> None:
    sword = WeaponAttack(name="Sword", range=1, damage=4, attack_bonus=20)
    hero = make_character("Hero", Position(0, 0), Team.PLAYERS, sword)
    target = make_character("Target", Position(1, 0), Team.ENEMIES)
    combat_state = CombatState(
        characters=[hero, target],
        grid_map=GridMap(width=4, height=4),
    )

    action = AttackAction(actor_id=0, target_id=1, weapon=sword)

    assert action.is_valid(combat_state)

    result = action.execute(combat_state)

    assert result.success
    assert "hit" in result.description
    assert "Action spent" in result.description
    assert not hero.action_economy.action_available
    assert target.hp == 6
    assert not action.is_valid(combat_state)


def test_attack_action_rejects_targets_outside_weapon_range() -> None:
    dagger = WeaponAttack(name="Dagger", range=1, damage=3, attack_bonus=20)
    hero = make_character("Hero", Position(0, 0), Team.PLAYERS, dagger)
    target = make_character("Target", Position(2, 0), Team.ENEMIES)
    combat_state = CombatState(
        characters=[hero, target],
        grid_map=GridMap(width=4, height=4),
    )

    action = AttackAction(actor_id=0, target_id=1, weapon=dagger)

    assert not action.is_valid(combat_state)

    result = action.execute(combat_state)

    assert not result.success
    assert target.hp == 10


def test_end_turn_action_advances_turn_and_round() -> None:
    hero = make_character("Hero", Position(0, 0), Team.PLAYERS)
    target = make_character("Target", Position(1, 0), Team.ENEMIES)
    combat_state = CombatState(characters=[hero, target], turn_index=1)
    hero.action_economy.action_available = False
    hero.action_economy.bonus_action_available = False
    hero.action_economy.reaction_available = False
    hero.action_economy.movement_remaining = 0

    result = EndTurnAction(actor_id=1).execute(combat_state)

    assert result.success
    assert combat_state.turn_index == 0
    assert combat_state.round_number == 2
    assert hero.action_economy.action_available
    assert hero.action_economy.bonus_action_available
    assert hero.action_economy.reaction_available
    assert hero.action_economy.movement_remaining == hero.speed
    assert "starts turn" in result.description


def test_character_starts_with_action_economy_resources() -> None:
    hero = make_character("Hero", Position(0, 0), Team.PLAYERS)

    assert hero.action_economy.action_available
    assert hero.action_economy.bonus_action_available
    assert hero.action_economy.reaction_available
    assert hero.action_economy.movement_remaining == hero.speed
