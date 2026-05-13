"""Reward shaping for D&D-like tactical combat."""

from __future__ import annotations

from dataclasses import dataclass, field

from combat.models import CombatState, Team


@dataclass(frozen=True)
class RewardConfig:
    """Tunable reward weights for combat learning."""

    damage_dealt_reward: float = 0.1
    enemy_kill_reward: float = 1.0
    victory_reward: float = 5.0
    ally_death_penalty: float = 1.0
    damage_taken_penalty: float = 0.1
    defeat_penalty: float = 5.0
    useless_turn_penalty: float = 0.02
    long_combat_round_penalty: float = 0.001


@dataclass(frozen=True)
class CombatRewardSnapshot:
    """Small immutable state snapshot used for reward calculation."""

    hp_by_team: dict[Team, int]
    alive_by_team: dict[Team, int]
    round_number: int

    @property
    def winner(self) -> Team | None:
        living_teams = [
            team
            for team, alive_count in self.alive_by_team.items()
            if alive_count > 0
        ]
        if len(living_teams) != 1:
            return None
        return living_teams[0]


@dataclass(frozen=True)
class CombatReward:
    """Total reward and signed component breakdown."""

    total: float
    breakdown: dict[str, float] = field(default_factory=dict)


def snapshot_combat_state(state: CombatState) -> CombatRewardSnapshot:
    """Capture HP, alive counts, and round number without keeping mutable state."""

    hp_by_team = {Team.PLAYERS: 0, Team.ENEMIES: 0}
    alive_by_team = {Team.PLAYERS: 0, Team.ENEMIES: 0}
    for character in state.characters:
        hp_by_team[character.team] += max(0, character.hp)
        if character.is_alive:
            alive_by_team[character.team] += 1

    return CombatRewardSnapshot(
        hp_by_team=hp_by_team,
        alive_by_team=alive_by_team,
        round_number=state.round_number,
    )


def calculate_combat_reward(
    before: CombatRewardSnapshot,
    after: CombatRewardSnapshot,
    actor_team: Team,
    action_success: bool = True,
    config: RewardConfig | None = None,
) -> CombatReward:
    """Calculate shaped reward from one actor team's perspective."""

    reward_config = config or RewardConfig()
    enemy_team = opposing_team(actor_team)

    enemy_damage = max(0, before.hp_by_team[enemy_team] - after.hp_by_team[enemy_team])
    ally_damage = max(0, before.hp_by_team[actor_team] - after.hp_by_team[actor_team])
    enemy_kills = max(
        0,
        before.alive_by_team[enemy_team] - after.alive_by_team[enemy_team],
    )
    ally_deaths = max(
        0,
        before.alive_by_team[actor_team] - after.alive_by_team[actor_team],
    )

    breakdown = {
        "damage_dealt": enemy_damage * reward_config.damage_dealt_reward,
        "enemy_kill": enemy_kills * reward_config.enemy_kill_reward,
        "victory": 0.0,
        "ally_death": -ally_deaths * reward_config.ally_death_penalty,
        "damage_taken": -ally_damage * reward_config.damage_taken_penalty,
        "defeat": 0.0,
        "useless_turn": 0.0,
        "long_combat": -max(0, after.round_number - 1)
        * reward_config.long_combat_round_penalty,
    }

    if after.winner is actor_team:
        breakdown["victory"] = reward_config.victory_reward
    elif after.winner is enemy_team:
        breakdown["defeat"] = -reward_config.defeat_penalty

    has_meaningful_event = (
        enemy_damage > 0
        or ally_damage > 0
        or enemy_kills > 0
        or ally_deaths > 0
        or after.winner is not None
    )
    if action_success and not has_meaningful_event:
        breakdown["useless_turn"] = -reward_config.useless_turn_penalty

    return CombatReward(total=sum(breakdown.values()), breakdown=breakdown)


def opposing_team(team: Team) -> Team:
    """Return the only opposing side currently modeled."""

    return Team.ENEMIES if team is Team.PLAYERS else Team.PLAYERS
