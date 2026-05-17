from combat import (
    AttackAction,
    Character,
    CombatRewardSnapshot,
    DashAction,
    DisengageAction,
    DodgeAction,
    GrappleAction,
    GridMap,
    HelpAction,
    HideAction,
    ImprovisedAction,
    Position,
    ReadyAction,
    RewardConfig,
    ShoveAction,
    Stats,
    Team,
    UseObjectAction,
    WeaponAttack,
    calculate_combat_reward,
    snapshot_combat_state,
)


def snapshot(
    player_hp: int,
    enemy_hp: int,
    player_alive: int = 1,
    enemy_alive: int = 1,
    round_number: int = 1,
) -> CombatRewardSnapshot:
    return CombatRewardSnapshot(
        hp_by_team={
            Team.PLAYERS: player_hp,
            Team.ENEMIES: enemy_hp,
        },
        alive_by_team={
            Team.PLAYERS: player_alive,
            Team.ENEMIES: enemy_alive,
        },
        round_number=round_number,
    )


def test_reward_positive_for_damage_kill_and_victory() -> None:
    before = snapshot(player_hp=10, enemy_hp=10)
    after = snapshot(player_hp=10, enemy_hp=0, enemy_alive=0)

    reward = calculate_combat_reward(before, after, actor_team=Team.PLAYERS)

    assert reward.breakdown["damage_dealt"] == 1.0
    assert reward.breakdown["enemy_kill"] == 1.0
    assert reward.breakdown["victory"] == 5.0
    assert reward.total == 7.0


def test_reward_penalizes_damage_ally_death_and_defeat() -> None:
    before = snapshot(player_hp=10, enemy_hp=10)
    after = snapshot(player_hp=0, enemy_hp=10, player_alive=0)

    reward = calculate_combat_reward(before, after, actor_team=Team.PLAYERS)

    assert reward.breakdown["damage_taken"] == -1.0
    assert reward.breakdown["ally_death"] == -1.0
    assert reward.breakdown["defeat"] == -5.0
    assert reward.total == -7.0


def test_reward_penalizes_useless_turn_and_long_combat() -> None:
    before = snapshot(player_hp=10, enemy_hp=10, round_number=2)
    after = snapshot(player_hp=10, enemy_hp=10, round_number=3)

    reward = calculate_combat_reward(before, after, actor_team=Team.PLAYERS)

    assert reward.breakdown["useless_turn"] == -0.02
    assert reward.breakdown["long_combat"] == -0.002
    assert reward.total == -0.022


def test_reward_config_can_tune_weights() -> None:
    before = snapshot(player_hp=10, enemy_hp=10)
    after = snapshot(player_hp=10, enemy_hp=5)
    config = RewardConfig(damage_dealt_reward=0.5)

    reward = calculate_combat_reward(
        before,
        after,
        actor_team=Team.PLAYERS,
        config=config,
    )

    assert reward.breakdown["damage_dealt"] == 2.5
    assert reward.total == 2.5


def make_character(
    name: str,
    position: Position,
    team: Team,
    hp: int = 10,
    max_hp: int = 10,
    ac: int = 12,
    stats: Stats | None = None,
    weapon: WeaponAttack | None = None,
) -> Character:
    return Character(
        name=name,
        hp=hp,
        max_hp=max_hp,
        ac=ac,
        position=position,
        speed=3,
        stats=stats or Stats(),
        team=team,
        weapons=[] if weapon is None else [weapon],
    )


def make_state(*characters: Character) -> object:
    from combat import CombatState

    return CombatState(
        characters=list(characters),
        grid_map=GridMap(width=6, height=6),
    )


def test_tactical_rewards_are_smaller_than_major_rewards() -> None:
    config = RewardConfig()
    tactical_values = [
        config.tactical_grapple_reward,
        config.tactical_shove_prone_reward,
        config.tactical_dodge_reward,
        config.tactical_disengage_reward,
        config.tactical_help_reward,
    ]
    penalties = [
        config.useless_dash_penalty,
        config.useless_hide_penalty,
        config.untriggered_ready_penalty,
        config.no_effect_action_penalty,
    ]

    assert max(tactical_values) < config.enemy_kill_reward
    assert max(tactical_values) < config.victory_reward
    assert max(penalties) < config.enemy_kill_reward


def test_reward_adds_tactical_grapple_for_dangerous_target(monkeypatch) -> None:
    actor = make_character("Actor", Position(0, 0), Team.PLAYERS, stats=Stats(str=18))
    target = make_character(
        "Ogre",
        Position(1, 0),
        Team.ENEMIES,
        hp=20,
        max_hp=20,
        weapon=WeaponAttack(name="Club", range=1, damage="1d12"),
    )
    state = make_state(actor, target)
    action = GrappleAction(actor_id=0, target_id=1)
    before = snapshot_combat_state(state)
    rolls = iter([20, 1, 1])
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: next(rolls))

    result = action.execute(state)
    reward = calculate_combat_reward(
        before,
        snapshot_combat_state(state),
        actor_team=Team.PLAYERS,
        action=action,
        action_result=result,
    )

    assert result.success
    assert reward.breakdown["tactical_grapple"] > 0
    assert reward.breakdown["useless_turn"] == 0


def test_reward_adds_shove_prone_when_ally_can_attack(monkeypatch) -> None:
    actor = make_character("Actor", Position(0, 0), Team.PLAYERS, stats=Stats(str=18))
    ally = make_character(
        "Ally",
        Position(1, 1),
        Team.PLAYERS,
        weapon=WeaponAttack(name="Sword", range=1, damage=3),
    )
    target = make_character("Target", Position(1, 0), Team.ENEMIES, stats=Stats(str=8, dex=8))
    state = make_state(actor, ally, target)
    action = ShoveAction(actor_id=0, target_id=2, shove_effect="prone")
    before = snapshot_combat_state(state)
    rolls = iter([20, 1, 1])
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: next(rolls))

    result = action.execute(state)
    reward = calculate_combat_reward(
        before,
        snapshot_combat_state(state),
        actor_team=Team.PLAYERS,
        action=action,
        action_result=result,
    )

    assert result.success
    assert target.prone is True
    assert reward.breakdown["tactical_shove_prone"] > 0


def test_reward_adds_help_bonus_when_helped_ally_hits(monkeypatch) -> None:
    helper = make_character("Helper", Position(0, 0), Team.PLAYERS)
    ally = make_character(
        "Ally",
        Position(0, 1),
        Team.PLAYERS,
        weapon=WeaponAttack(name="Sword", range=1, damage=3, attack_bonus=20),
    )
    target = make_character("Target", Position(1, 1), Team.ENEMIES)
    state = make_state(helper, ally, target)
    HelpAction(actor_id=0, target_id=2).execute(state)
    action = AttackAction(actor_id=1, target_id=2)
    before = snapshot_combat_state(state)
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: 10)

    result = action.execute(state)
    reward = calculate_combat_reward(
        before,
        snapshot_combat_state(state),
        actor_team=Team.PLAYERS,
        action=action,
        action_result=result,
    )

    assert result.success
    assert target.hp < target.max_hp
    assert reward.breakdown["tactical_help"] > 0


def test_reward_adds_dodge_bonus_from_defender_perspective(monkeypatch) -> None:
    target = make_character("Dodger", Position(0, 0), Team.PLAYERS, ac=15)
    attacker = make_character(
        "Attacker",
        Position(1, 0),
        Team.ENEMIES,
        weapon=WeaponAttack(name="Sword", range=1, damage=3),
    )
    state = make_state(target, attacker)
    DodgeAction(actor_id=0).execute(state)
    before = snapshot_combat_state(state)
    rolls = iter([20, 1])
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: next(rolls))
    action = AttackAction(actor_id=1, target_id=0)

    result = action.execute(state)
    reward = calculate_combat_reward(
        before,
        snapshot_combat_state(state),
        actor_team=Team.PLAYERS,
        action=action,
        action_result=result,
    )

    assert result.success
    assert "miss" in result.description
    assert reward.breakdown["tactical_dodge"] > 0


def test_reward_adds_disengage_bonus_when_actor_was_in_danger() -> None:
    actor = make_character("Actor", Position(0, 0), Team.PLAYERS)
    enemy = make_character("Enemy", Position(1, 0), Team.ENEMIES)
    state = make_state(actor, enemy)
    action = DisengageAction(actor_id=0)
    before = snapshot_combat_state(state)

    result = action.execute(state)
    reward = calculate_combat_reward(
        before,
        snapshot_combat_state(state),
        actor_team=Team.PLAYERS,
        action=action,
        action_result=result,
    )

    assert result.success
    assert reward.breakdown["tactical_disengage"] > 0


def test_reward_penalizes_common_actions_without_tactical_value(monkeypatch) -> None:
    actor = make_character("Actor", Position(0, 0), Team.PLAYERS)
    enemy = make_character("Enemy", Position(1, 0), Team.ENEMIES)
    state = make_state(actor, enemy)

    before = snapshot_combat_state(state)
    dash = DashAction(actor_id=0)
    dash_result = dash.execute(state)
    dash_reward = calculate_combat_reward(
        before,
        snapshot_combat_state(state),
        actor_team=Team.PLAYERS,
        action=dash,
        action_result=dash_result,
    )
    assert dash_reward.breakdown["useless_dash"] < 0

    actor.action_economy.action_available = True
    before = snapshot_combat_state(state)
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: 1)
    hide = HideAction(actor_id=0, dc=30)
    hide_result = hide.execute(state)
    hide_reward = calculate_combat_reward(
        before,
        snapshot_combat_state(state),
        actor_team=Team.PLAYERS,
        action=hide,
        action_result=hide_result,
    )
    assert hide_reward.breakdown["useless_hide"] < 0

    actor.action_economy.action_available = True
    before = snapshot_combat_state(state)
    ready = ReadyAction(actor_id=0)
    ready_result = ready.execute(state)
    ready_reward = calculate_combat_reward(
        before,
        snapshot_combat_state(state),
        actor_team=Team.PLAYERS,
        action=ready,
        action_result=ready_result,
    )
    assert ready_reward.breakdown["untriggered_ready"] < 0

    actor.action_economy.action_available = True
    before = snapshot_combat_state(state)
    use_object = UseObjectAction(actor_id=0)
    use_object_result = use_object.execute(state)
    use_object_reward = calculate_combat_reward(
        before,
        snapshot_combat_state(state),
        actor_team=Team.PLAYERS,
        action=use_object,
        action_result=use_object_result,
    )
    assert use_object_reward.breakdown["no_effect_action"] < 0

    actor.action_economy.action_available = True
    before = snapshot_combat_state(state)
    improvised = ImprovisedAction(actor_id=0)
    improvised_result = improvised.execute(state)
    improvised_reward = calculate_combat_reward(
        before,
        snapshot_combat_state(state),
        actor_team=Team.PLAYERS,
        action=improvised,
        action_result=improvised_result,
    )
    assert improvised_reward.breakdown["no_effect_action"] < 0
