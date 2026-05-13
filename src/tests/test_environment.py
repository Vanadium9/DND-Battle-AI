from combat import (
    AttackAction,
    Character,
    CombatEnvironment,
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
    hp: int = 10,
    speed: int = 2,
    weapon: WeaponAttack | None = None,
) -> Character:
    abilities = [] if weapon is None else [weapon]
    return Character(
        name=name,
        hp=hp,
        max_hp=10,
        ac=12,
        position=position,
        speed=speed,
        stats=Stats(),
        team=team,
        abilities=abilities,
    )


def test_environment_reset_and_observation() -> None:
    hero = make_character("Hero", Position(0, 0), Team.PLAYERS)
    target = make_character("Target", Position(1, 0), Team.ENEMIES)
    environment = CombatEnvironment(
        characters=[hero, target],
        grid_map=GridMap(width=4, height=4),
        log_to_console=False,
    )

    combat_state = environment.reset()
    observation = environment.get_observation(actor_id=0)

    assert combat_state.turn_index == 0
    assert observation["active_actor_id"] == 0
    assert observation["is_done"] is False
    assert observation["winner"] is None
    assert combat_state.characters[0].action_economy.movement_remaining == 2


def test_environment_allows_move_and_attack_in_one_turn() -> None:
    sword = WeaponAttack(name="Sword", range=1, damage=4, attack_bonus=20)
    hero = make_character("Hero", Position(0, 0), Team.PLAYERS, weapon=sword)
    target = make_character("Target", Position(1, 1), Team.ENEMIES)
    environment = CombatEnvironment(
        characters=[hero, target],
        grid_map=GridMap(width=4, height=4),
        log_to_console=False,
    )

    move_result = environment.step(MoveAction(actor_id=0, destination=Position(0, 1)))
    attack_result = environment.step(AttackAction(actor_id=0, target_id=1, weapon=sword))

    actor = environment.combat_state.characters[0]
    target = environment.combat_state.characters[1]

    assert move_result.success
    assert move_result.reward < 0
    assert attack_result.success
    assert attack_result.reward > 0
    assert actor.position == Position(0, 1)
    assert actor.action_economy.movement_remaining == 1
    assert not actor.action_economy.action_available
    assert target.hp == 6
    assert environment.combat_state.turn_index == 0


def test_environment_rejects_second_attack_without_action() -> None:
    sword = WeaponAttack(name="Sword", range=1, damage=1, attack_bonus=20)
    hero = make_character("Hero", Position(0, 0), Team.PLAYERS, weapon=sword)
    target = make_character("Target", Position(1, 0), Team.ENEMIES)
    environment = CombatEnvironment(
        characters=[hero, target],
        grid_map=GridMap(width=4, height=4),
        log_to_console=False,
    )

    first_result = environment.step(AttackAction(actor_id=0, target_id=1, weapon=sword))
    second_result = environment.step(AttackAction(actor_id=0, target_id=1, weapon=sword))

    assert first_result.success
    assert not second_result.success
    assert "not valid" in second_result.description


def test_environment_end_turn_skips_dead_creature() -> None:
    hero = make_character("Hero", Position(0, 0), Team.PLAYERS)
    dead_enemy = make_character("Dead", Position(1, 0), Team.ENEMIES, hp=0)
    living_enemy = make_character("Living", Position(2, 0), Team.ENEMIES)
    environment = CombatEnvironment(
        characters=[hero, dead_enemy, living_enemy],
        grid_map=GridMap(width=4, height=4),
        log_to_console=False,
    )

    result = environment.step(EndTurnAction(actor_id=0))

    assert result.success
    assert environment.combat_state.turn_index == 2
    assert environment.combat_state.active_character is not None
    assert environment.combat_state.active_character.name == "Living"


def test_environment_done_and_winner_when_one_team_dead() -> None:
    sword = WeaponAttack(name="Sword", range=1, damage=10, attack_bonus=20)
    hero = make_character("Hero", Position(0, 0), Team.PLAYERS, weapon=sword)
    target = make_character("Target", Position(1, 0), Team.ENEMIES)
    environment = CombatEnvironment(
        characters=[hero, target],
        grid_map=GridMap(width=4, height=4),
        log_to_console=False,
    )

    result = environment.step(AttackAction(actor_id=0, target_id=1, weapon=sword))

    assert result.success
    assert result.reward > 0
    assert environment.is_done()
    assert environment.get_winner() is Team.PLAYERS


def test_available_actions_include_end_turn_and_respect_action_economy() -> None:
    sword = WeaponAttack(name="Sword", range=1, damage=1, attack_bonus=20)
    hero = make_character("Hero", Position(0, 0), Team.PLAYERS, weapon=sword)
    target = make_character("Target", Position(1, 0), Team.ENEMIES)
    environment = CombatEnvironment(
        characters=[hero, target],
        grid_map=GridMap(width=4, height=4),
        log_to_console=False,
    )

    available_actions = environment.get_available_actions(actor_id=0)
    assert any(isinstance(action, MoveAction) for action in available_actions)
    assert any(isinstance(action, AttackAction) for action in available_actions)
    assert any(isinstance(action, EndTurnAction) for action in available_actions)

    environment.step(AttackAction(actor_id=0, target_id=1, weapon=sword))
    available_actions = environment.get_available_actions(actor_id=0)

    assert not any(isinstance(action, AttackAction) for action in available_actions)
    assert any(isinstance(action, EndTurnAction) for action in available_actions)
