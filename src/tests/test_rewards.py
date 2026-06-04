from combat import (
    ActionResult,
    ActionSurgeAction,
    AttackAction,
    CastSpellAction,
    Character,
    CombatEnvironment,
    CombatRewardSnapshot,
    CombatState,
    DashAction,
    DamageType,
    DisengageAction,
    DodgeAction,
    GrappleAction,
    GridMap,
    HelpAction,
    HideAction,
    MoveAction,
    PotionOfHealing,
    Position,
    ReadyAction,
    RewardConfig,
    Resource,
    SecondWindAction,
    ShoveAction,
    SpellAbility,
    Stats,
    Team,
    TerrainType,
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

def test_spell_slot_penalty_scales_by_slot_level() -> None:
    actor = make_character("Wizard", Position(0, 0), Team.PLAYERS)
    target = make_character("Target", Position(1, 0), Team.ENEMIES)
    actor.spell_slots_remaining = {1: 1, 3: 1}
    actor.spell_slots = {1: 1, 3: 1}
    spell = SpellAbility(
        name="Magic Missile",
        range=6,
        spell_level=1,
        damage=1,
        damage_type=DamageType.FORCE,
    )
    state = make_state(actor, target)

    before_level_1 = snapshot_combat_state(state)
    actor.spell_slots_remaining[1] = 0
    level_1_reward = calculate_combat_reward(
        before_level_1,
        snapshot_combat_state(state),
        actor_team=Team.PLAYERS,
        action=CastSpellAction(actor_id=0, target_id=1, spell=spell),
        action_result=ActionResult(True, "Wizard casts Magic Missile at level 1."),
    )

    actor.spell_slots_remaining = {1: 1, 3: 1}
    before_level_3 = snapshot_combat_state(state)
    actor.spell_slots_remaining[3] = 0
    level_3_reward = calculate_combat_reward(
        before_level_3,
        snapshot_combat_state(state),
        actor_team=Team.PLAYERS,
        action=CastSpellAction(actor_id=0, target_id=1, spell=spell, cast_level=3),
        action_result=ActionResult(True, "Wizard casts Magic Missile at level 3."),
    )

    assert level_1_reward.breakdown["spell_slot_spent"] == -0.015
    assert level_3_reward.breakdown["spell_slot_spent"] == -0.045
    assert level_3_reward.breakdown["spell_slot_spent"] < level_1_reward.breakdown["spell_slot_spent"]


def test_reward_penalizes_action_surge_and_ineffective_second_wind() -> None:
    actor = make_character("Fighter", Position(0, 0), Team.PLAYERS, hp=19, max_hp=20)
    actor.resources = {
        "action_surge": Resource("action_surge", max_uses=1),
        "second_wind": Resource("second_wind", max_uses=1),
    }
    enemy = make_character("Enemy", Position(1, 0), Team.ENEMIES)
    state = make_state(actor, enemy)

    before = snapshot_combat_state(state)
    actor.resources["action_surge"].spend()
    action_surge_reward = calculate_combat_reward(
        before,
        snapshot_combat_state(state),
        actor_team=Team.PLAYERS,
        action=ActionSurgeAction(actor_id=0),
        action_result=ActionResult(True, "Fighter uses Action Surge."),
    )

    actor.resources["second_wind"].uses_remaining = 1
    before = snapshot_combat_state(state)
    actor.resources["second_wind"].spend()
    actor.hp = 20
    second_wind_reward = calculate_combat_reward(
        before,
        snapshot_combat_state(state),
        actor_team=Team.PLAYERS,
        action=SecondWindAction(actor_id=0),
        action_result=ActionResult(True, "Fighter uses Second Wind and heals 1 HP."),
    )

    assert action_surge_reward.breakdown["action_surge_spent"] < 0
    assert second_wind_reward.breakdown["ineffective_second_wind"] < 0


def test_reward_penalizes_immunity_and_resisted_damage_with_better_alternative() -> None:
    fire_weapon = WeaponAttack(
        name="Fire Blade",
        range=1,
        damage=5,
        damage_type=DamageType.FIRE,
    )
    force_weapon = WeaponAttack(
        name="Force Blade",
        range=1,
        damage=5,
        damage_type=DamageType.FORCE,
    )
    actor = make_character(
        "Actor",
        Position(0, 0),
        Team.PLAYERS,
        weapon=fire_weapon,
    )
    actor.weapons.append(force_weapon)
    target = make_character("Target", Position(1, 0), Team.ENEMIES)
    target.immunities = {DamageType.FIRE}
    target.resistances = {DamageType.SLASHING}
    state = make_state(actor, target)

    before = snapshot_combat_state(state)
    immunity_reward = calculate_combat_reward(
        before,
        snapshot_combat_state(state),
        actor_team=Team.PLAYERS,
        action=AttackAction(actor_id=0, target_id=1, weapon=fire_weapon),
        action_result=ActionResult(True, "Actor attacks Target for 0 damage."),
    )

    slash_weapon = WeaponAttack(
        name="Sword",
        range=1,
        damage=5,
        damage_type=DamageType.SLASHING,
    )
    actor.weapons = [slash_weapon, force_weapon]
    before = snapshot_combat_state(state)
    target.hp -= 2
    resisted_reward = calculate_combat_reward(
        before,
        snapshot_combat_state(state),
        actor_team=Team.PLAYERS,
        action=AttackAction(actor_id=0, target_id=1, weapon=slash_weapon),
        action_result=ActionResult(True, "Actor attacks Target for 2 damage."),
    )

    assert immunity_reward.breakdown["immunity_damage"] < 0
    assert resisted_reward.breakdown["resisted_damage"] < 0


def test_reward_balances_effective_expensive_resource_and_overkill() -> None:
    actor = make_character("Wizard", Position(0, 0), Team.PLAYERS)
    target = make_character("Target", Position(1, 0), Team.ENEMIES, hp=1, max_hp=10)
    actor.spell_slots_remaining = {3: 1}
    actor.spell_slots = {3: 1}
    spell = SpellAbility(
        name="Fireball",
        range=6,
        spell_level=3,
        damage="8d6",
        damage_type=DamageType.FIRE,
    )
    state = make_state(actor, target)
    before = snapshot_combat_state(state)
    actor.spell_slots_remaining[3] = 0
    target.hp = 0

    reward = calculate_combat_reward(
        before,
        snapshot_combat_state(state),
        actor_team=Team.PLAYERS,
        action=CastSpellAction(actor_id=0, target_id=1, spell=spell),
        action_result=ActionResult(True, "Wizard casts Fireball for 20 damage."),
    )

    assert reward.breakdown["effective_resource"] > 0
    assert reward.breakdown["resource_overkill"] < 0
    assert reward.breakdown["victory"] > abs(reward.breakdown["spell_slot_spent"])
    assert reward.total > reward.breakdown["victory"]


def test_reward_adds_cover_bonus_when_cover_prevents_damage() -> None:
    target = make_character("Covered", Position(1, 0), Team.PLAYERS, ac=12)
    attacker = make_character(
        "Archer",
        Position(0, 0),
        Team.ENEMIES,
        weapon=WeaponAttack(name="Bow", range=6, damage=3),
    )
    state = CombatState(
        characters=[target, attacker],
        grid_map=GridMap(
            width=3,
            height=3,
            terrain_grid=[
                [TerrainType.NORMAL, TerrainType.LOW_COVER, TerrainType.NORMAL],
                [TerrainType.NORMAL, TerrainType.NORMAL, TerrainType.NORMAL],
                [TerrainType.NORMAL, TerrainType.NORMAL, TerrainType.NORMAL],
            ],
        ),
    )
    before = snapshot_combat_state(state)

    reward = calculate_combat_reward(
        before,
        snapshot_combat_state(state),
        actor_team=Team.PLAYERS,
        action=AttackAction(actor_id=1, target_id=0, weapon=attacker.weapons[0]),
        action_result=ActionResult(True, "Archer attacks Covered: miss."),
    )

    assert reward.breakdown["cover_avoid_damage"] > 0


def test_reward_penalizes_bad_position_movement_without_tactical_gain() -> None:
    actor = make_character("Actor", Position(0, 0), Team.PLAYERS)
    enemy = make_character("Enemy", Position(2, 0), Team.ENEMIES)
    state = make_state(actor, enemy)
    before = snapshot_combat_state(state)
    actor.position = Position(1, 0)

    reward = calculate_combat_reward(
        before,
        snapshot_combat_state(state),
        actor_team=Team.PLAYERS,
        action=MoveAction(actor_id=0, destination=Position(1, 0)),
        action_result=ActionResult(True, "Actor moves into reach."),
    )

    assert reward.breakdown["bad_position"] < 0


def test_reward_penalizes_wasted_consumable_item() -> None:
    actor = make_character("Actor", Position(0, 0), Team.PLAYERS)
    potion = PotionOfHealing(quantity=1)
    actor.inventory = [potion]
    enemy = make_character("Enemy", Position(1, 0), Team.ENEMIES)
    state = make_state(actor, enemy)
    before = snapshot_combat_state(state)
    potion.quantity = 0

    reward = calculate_combat_reward(
        before,
        snapshot_combat_state(state),
        actor_team=Team.PLAYERS,
        action=UseObjectAction(actor_id=0, item=potion, target_id=0),
        action_result=ActionResult(True, "Actor uses Potion of Healing and heals 0 HP."),
    )

    assert reward.breakdown["wasted_item"] < 0
    assert reward.breakdown["no_effect_action"] < 0


def test_environment_logs_reward_breakdown(monkeypatch) -> None:
    hero = make_character(
        "Hero",
        Position(0, 0),
        Team.PLAYERS,
        weapon=WeaponAttack(name="Sword", range=1, damage=4, attack_bonus=20),
    )
    enemy = make_character("Enemy", Position(1, 0), Team.ENEMIES)
    environment = CombatEnvironment(
        characters=[hero, enemy],
        grid_map=GridMap(width=3, height=3),
        use_initiative=False,
        log_to_console=False,
    )
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: 10)

    result = environment.step(AttackAction(actor_id=0, target_id=1, weapon=hero.weapons[0]))

    assert result.reward_breakdown["damage_dealt"] > 0
    assert any("Reward breakdown:" in entry for entry in environment.action_log)
