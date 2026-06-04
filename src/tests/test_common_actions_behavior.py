from combat import (
    AttackAction,
    Character,
    CombatState,
    DashAction,
    DisengageAction,
    DodgeAction,
    GrappleAction,
    GridMap,
    HelpAction,
    HideAction,
    OpportunityAttackAction,
    Position,
    ReadyAction,
    ShoveAction,
    StabilizeAction,
    Stats,
    Team,
    WeaponAttack,
)


def make_weapon(
    name: str = "Sword",
    damage: int = 3,
    attack_bonus: int = 0,
    range: int = 1,
) -> WeaponAttack:
    return WeaponAttack(
        name=name,
        range=range,
        damage=damage,
        attack_bonus=attack_bonus,
        ability_score="str",
        damage_ability_score=None,
    )


def make_character(
    name: str,
    position: Position,
    team: Team,
    hp: int = 10,
    ac: int = 12,
    stats: Stats | None = None,
    weapon: WeaponAttack | None = None,
) -> Character:
    return Character(
        name=name,
        hp=hp,
        max_hp=max(10, hp),
        ac=ac,
        position=position,
        speed=3,
        stats=stats or Stats(),
        team=team,
        weapons=[] if weapon is None else [weapon],
    )


def make_state(*characters: Character) -> CombatState:
    return CombatState(
        characters=list(characters),
        grid_map=GridMap(width=5, height=5),
    )


def test_dash_increases_movement_remaining_and_spends_action() -> None:
    hero = make_character("Hero", Position(0, 0), Team.PLAYERS)
    state = make_state(hero)

    result = DashAction(actor_id=0).execute(state)

    assert result.success
    assert hero.action_economy.movement_remaining == hero.speed * 2
    assert hero.action_economy.action_available is False


def test_disengage_prevents_opportunity_attack() -> None:
    target = make_character("Target", Position(1, 0), Team.PLAYERS)
    attacker = make_character(
        "Attacker",
        Position(0, 0),
        Team.ENEMIES,
        weapon=make_weapon(),
    )
    state = make_state(target, attacker)

    result = DisengageAction(actor_id=0).execute(state)
    opportunity = OpportunityAttackAction(actor_id=1, target_id=0)

    assert result.success
    assert target.disengaged_until_end_of_turn is True
    assert opportunity.is_valid(state) is False
    assert opportunity.execute(state).success is False
    assert attacker.action_economy.reaction_available is True


def test_dodge_gives_attackers_disadvantage_until_next_turn(monkeypatch) -> None:
    target = make_character("Target", Position(0, 0), Team.PLAYERS, ac=15)
    attacker = make_character(
        "Attacker",
        Position(1, 0),
        Team.ENEMIES,
        weapon=make_weapon(),
    )
    state = make_state(target, attacker)

    dodge_result = DodgeAction(actor_id=0).execute(state)
    rolls = iter([20, 1])
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: next(rolls))
    attack_result = AttackAction(actor_id=1, target_id=0).execute(state)

    assert dodge_result.success
    assert target.dodging_until_start_of_next_turn is True
    assert attack_result.success
    assert "miss" in attack_result.description
    assert "d20=1" in attack_result.description
    assert target.hp == target.max_hp

    state.reset_turn_resources(actor_id=0)

    assert target.dodging_until_start_of_next_turn is False


def test_help_gives_ally_advantage_on_next_attack(monkeypatch) -> None:
    helper = make_character("Helper", Position(0, 0), Team.PLAYERS)
    ally = make_character(
        "Ally",
        Position(0, 1),
        Team.PLAYERS,
        weapon=make_weapon(damage=4),
    )
    enemy = make_character("Enemy", Position(1, 1), Team.ENEMIES, ac=15)
    state = make_state(helper, ally, enemy)

    help_result = HelpAction(actor_id=0, target_id=2).execute(state)
    assert help_result.success
    assert helper.help_against_target_id == 2

    rolls = iter([1, 20])
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: next(rolls))
    attack_result = AttackAction(actor_id=1, target_id=2).execute(state)

    assert attack_result.success
    assert "hit" in attack_result.description
    assert enemy.hp == 6
    assert helper.help_against_target_id is None


def test_hide_can_set_hidden_true(monkeypatch) -> None:
    hero = make_character("Hero", Position(0, 0), Team.PLAYERS)
    state = make_state(hero)
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: 20)

    result = HideAction(actor_id=0).execute(state)

    assert result.success
    assert hero.hidden is True
    assert "succeeds" in result.description


def test_grapple_sets_grappled_on_successful_contested_check(monkeypatch) -> None:
    actor = make_character("Actor", Position(0, 0), Team.PLAYERS, stats=Stats(str=18))
    target = make_character("Target", Position(1, 0), Team.ENEMIES, stats=Stats(str=8, dex=8))
    state = make_state(actor, target)
    rolls = iter([20, 1, 1])
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: next(rolls))

    result = GrappleAction(actor_id=0, target_id=1).execute(state)

    assert result.success
    assert target.grappled is True
    assert target.grappled_by == 0
    assert actor.grappling_target_id == 1


def test_shove_can_make_target_prone(monkeypatch) -> None:
    actor = make_character("Actor", Position(0, 0), Team.PLAYERS, stats=Stats(str=18))
    target = make_character("Target", Position(1, 0), Team.ENEMIES, stats=Stats(str=8, dex=8))
    state = make_state(actor, target)
    rolls = iter([20, 1, 1])
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: next(rolls))

    result = ShoveAction(actor_id=0, target_id=1, shove_effect="prone").execute(state)

    assert result.success
    assert target.prone is True


def test_shove_can_push_target_one_cell(monkeypatch) -> None:
    actor = make_character("Actor", Position(0, 0), Team.PLAYERS, stats=Stats(str=18))
    target = make_character("Target", Position(1, 0), Team.ENEMIES, stats=Stats(str=8, dex=8))
    state = make_state(actor, target)
    rolls = iter([20, 1, 1])
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: next(rolls))

    result = ShoveAction(actor_id=0, target_id=1, shove_effect="push").execute(state)

    assert result.success
    assert target.position == Position(2, 0)
    assert target.prone is False


def test_stabilize_works_on_zero_hp_creature(monkeypatch) -> None:
    actor = make_character("Actor", Position(0, 0), Team.PLAYERS, stats=Stats(wis=10))
    target = make_character("Target", Position(1, 0), Team.PLAYERS, hp=0)
    actor.common_actions.append("stabilize")
    state = make_state(actor, target)
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: 8)

    result = StabilizeAction(actor_id=0, target_id=1).execute(state)

    assert result.success
    assert target.stable is True
    assert "Medicine:" in result.description


def test_ready_saves_prepared_action() -> None:
    actor = make_character("Actor", Position(0, 0), Team.PLAYERS)
    state = make_state(actor)

    result = ReadyAction(
        actor_id=0,
        prepared_action="attack",
        trigger_description="enemy leaves cover",
    ).execute(state)

    assert result.success
    assert actor.prepared_action == "attack"
    assert actor.trigger_description == "enemy leaves cover"


def test_opportunity_attack_spends_reaction(monkeypatch) -> None:
    attacker = make_character(
        "Attacker",
        Position(0, 0),
        Team.PLAYERS,
        weapon=make_weapon(damage=4),
    )
    target = make_character("Target", Position(1, 0), Team.ENEMIES)
    state = make_state(attacker, target)
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: 20)

    result = OpportunityAttackAction(actor_id=0, target_id=1).execute(state)

    assert result.success
    assert "opportunity attacks" in result.description
    assert attacker.action_economy.reaction_available is False
    assert attacker.action_economy.reaction_used_this_round is True
    assert target.hp == 6


def test_cannot_take_two_main_actions_without_action_surge() -> None:
    actor = make_character(
        "Actor",
        Position(0, 0),
        Team.PLAYERS,
        weapon=make_weapon(),
    )
    target = make_character("Target", Position(1, 0), Team.ENEMIES)
    state = make_state(actor, target)

    result = DashAction(actor_id=0).execute(state)

    assert result.success
    assert actor.action_economy.action_available is False
    assert AttackAction(actor_id=0, target_id=1).is_valid(state) is False
    assert ShoveAction(actor_id=0, target_id=1).is_valid(state) is False


def test_cannot_take_reaction_twice_in_one_round(monkeypatch) -> None:
    attacker = make_character(
        "Attacker",
        Position(0, 0),
        Team.PLAYERS,
        weapon=make_weapon(damage=2),
    )
    first_target = make_character("First Target", Position(1, 0), Team.ENEMIES)
    second_target = make_character("Second Target", Position(0, 1), Team.ENEMIES)
    state = make_state(attacker, first_target, second_target)
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: 20)

    first_result = OpportunityAttackAction(actor_id=0, target_id=1).execute(state)
    second_action = OpportunityAttackAction(actor_id=0, target_id=2)
    second_result = second_action.execute(state)

    assert first_result.success
    assert attacker.action_economy.reaction_available is False
    assert second_action.is_valid(state) is False
    assert second_result.success is False
    assert second_target.hp == second_target.max_hp
