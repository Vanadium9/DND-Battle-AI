from combat import (
    CombatRewardSnapshot,
    RewardConfig,
    Team,
    calculate_combat_reward,
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
