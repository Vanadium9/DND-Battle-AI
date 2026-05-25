"""Reward shaping for D&D-like tactical combat."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from combat.models import CombatState, Position, Team


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
    tactical_grapple_reward: float = 0.08
    tactical_shove_prone_reward: float = 0.08
    tactical_dodge_reward: float = 0.06
    tactical_disengage_reward: float = 0.05
    tactical_help_reward: float = 0.06
    useless_dash_penalty: float = 0.03
    useless_hide_penalty: float = 0.03
    untriggered_ready_penalty: float = 0.02
    no_effect_action_penalty: float = 0.03
    friendly_fire_penalty: float = 0.05


@dataclass(frozen=True)
class CharacterRewardSnapshot:
    """Immutable per-character combat state used for tactical rewards."""

    character_id: int
    name: str
    team: Team
    hp: int
    max_hp: int
    ac: int
    position: Position
    speed: int
    alive: bool
    action_available: bool
    movement_remaining: int
    reaction_available: bool
    hidden: bool
    prone: bool
    grappled: bool
    dodging: bool
    disengaged: bool
    prepared_action: str | None
    help_against_target_id: int | None
    grappling_target_id: int | None
    grappled_by: int | None
    max_weapon_range: int
    weapon_damage_potential: int


@dataclass(frozen=True)
class CombatRewardSnapshot:
    """Small immutable state snapshot used for reward calculation."""

    hp_by_team: dict[Team, int]
    alive_by_team: dict[Team, int]
    round_number: int
    characters: tuple[CharacterRewardSnapshot, ...] = ()

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
    character_snapshots = []
    for character_id, character in enumerate(state.characters):
        hp_by_team[character.team] += max(0, character.hp)
        if character.is_alive:
            alive_by_team[character.team] += 1
        character_snapshots.append(
            CharacterRewardSnapshot(
                character_id=character_id,
                name=character.name,
                team=character.team,
                hp=character.hp,
                max_hp=character.max_hp,
                ac=character.ac,
                position=character.position,
                speed=character.speed,
                alive=character.is_alive,
                action_available=character.action_economy.action_available,
                movement_remaining=character.action_economy.movement_remaining,
                reaction_available=character.action_economy.reaction_available,
                hidden=character.hidden,
                prone=character.prone,
                grappled=character.grappled,
                dodging=character.dodging_until_start_of_next_turn,
                disengaged=character.disengaged_until_end_of_turn,
                prepared_action=character.prepared_action,
                help_against_target_id=character.help_against_target_id,
                grappling_target_id=character.grappling_target_id,
                grappled_by=character.grappled_by,
                max_weapon_range=max(
                    (weapon.range for weapon in character.weapons),
                    default=0,
                ),
                weapon_damage_potential=max(
                    (_damage_potential(weapon.damage) for weapon in character.weapons),
                    default=0,
                ),
            )
        )

    return CombatRewardSnapshot(
        hp_by_team=hp_by_team,
        alive_by_team=alive_by_team,
        round_number=state.round_number,
        characters=tuple(character_snapshots),
    )


def calculate_combat_reward(
    before: CombatRewardSnapshot,
    after: CombatRewardSnapshot,
    actor_team: Team,
    action_success: bool = True,
    config: RewardConfig | None = None,
    action: Any | None = None,
    action_result: Any | None = None,
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
        "tactical_grapple": 0.0,
        "tactical_shove_prone": 0.0,
        "tactical_dodge": 0.0,
        "tactical_disengage": 0.0,
        "tactical_help": 0.0,
        "useless_dash": 0.0,
        "useless_hide": 0.0,
        "untriggered_ready": 0.0,
        "no_effect_action": 0.0,
        "friendly_fire": 0.0,
    }

    if action is not None and action_success:
        breakdown.update(
            _common_action_rewards(
                before,
                after,
                actor_team,
                action,
                action_result,
                reward_config,
            )
        )

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
        or any(value > 0 for key, value in breakdown.items() if key.startswith("tactical_"))
    )
    if action_success and not has_meaningful_event:
        breakdown["useless_turn"] = -reward_config.useless_turn_penalty

    return CombatReward(total=sum(breakdown.values()), breakdown=breakdown)


def opposing_team(team: Team) -> Team:
    """Return the only opposing side currently modeled."""

    return Team.ENEMIES if team is Team.PLAYERS else Team.PLAYERS


def _team_damage_taken(
    before: CombatRewardSnapshot,
    after: CombatRewardSnapshot,
    team: Team,
) -> int:
    return max(0, before.hp_by_team[team] - after.hp_by_team[team])


def _common_action_rewards(
    before: CombatRewardSnapshot,
    after: CombatRewardSnapshot,
    actor_team: Team,
    action: Any,
    action_result: Any | None,
    config: RewardConfig,
) -> dict[str, float]:
    action_name = action.__class__.__name__
    rewards = {
        "tactical_grapple": 0.0,
        "tactical_shove_prone": 0.0,
        "tactical_dodge": 0.0,
        "tactical_disengage": 0.0,
        "tactical_help": 0.0,
        "useless_dash": 0.0,
        "useless_hide": 0.0,
        "untriggered_ready": 0.0,
        "no_effect_action": 0.0,
        "friendly_fire": 0.0,
    }

    if action_name in {"CastSpellAction", "UseObjectAction"}:
        ally_damage = _team_damage_taken(before, after, actor_team)
        if ally_damage > 0:
            rewards["friendly_fire"] = -ally_damage * config.friendly_fire_penalty

    if action_name == "GrappleAction":
        if _successful_tactical_grapple(before, after, actor_team, action):
            rewards["tactical_grapple"] = config.tactical_grapple_reward
    elif action_name == "ShoveAction":
        if _successful_tactical_shove_prone(before, after, actor_team, action):
            rewards["tactical_shove_prone"] = config.tactical_shove_prone_reward
    elif action_name == "AttackAction":
        if _dodging_ally_avoided_attack(before, after, actor_team, action, action_result):
            rewards["tactical_dodge"] = config.tactical_dodge_reward
        if _helped_attack_hit(before, after, actor_team, action):
            rewards["tactical_help"] = config.tactical_help_reward
    elif action_name == "DisengageAction":
        if _successful_tactical_disengage(before, after, actor_team, action):
            rewards["tactical_disengage"] = config.tactical_disengage_reward
    elif action_name == "DashAction":
        if not _dash_has_tactical_improvement(before, after, actor_team, action):
            rewards["useless_dash"] = -config.useless_dash_penalty
    elif action_name == "HideAction":
        if not _hide_granted_advantage(after, action):
            rewards["useless_hide"] = -config.useless_hide_penalty
    elif action_name == "ReadyAction":
        rewards["untriggered_ready"] = -config.untriggered_ready_penalty
    elif action_name in {"UseObjectAction", "ImprovisedAction"}:
        if not _action_had_state_effect(before, after, actor_team):
            rewards["no_effect_action"] = -config.no_effect_action_penalty

    return rewards


def _successful_tactical_grapple(
    before: CombatRewardSnapshot,
    after: CombatRewardSnapshot,
    actor_team: Team,
    action: Any,
) -> bool:
    actor = _character(before, getattr(action, "actor_id", -1))
    target_before = _character(before, getattr(action, "target_id", -1))
    target_after = _character(after, getattr(action, "target_id", -1))
    if actor is None or target_before is None or target_after is None:
        return False
    return (
        actor.team is actor_team
        and target_before.team is opposing_team(actor_team)
        and target_after.grappled_by == actor.character_id
        and (
            _is_dangerous_target(target_before)
            or _near_vulnerable_ally(before, actor_team, target_before)
        )
    )


def _successful_tactical_shove_prone(
    before: CombatRewardSnapshot,
    after: CombatRewardSnapshot,
    actor_team: Team,
    action: Any,
) -> bool:
    actor = _character(before, getattr(action, "actor_id", -1))
    target_before = _character(before, getattr(action, "target_id", -1))
    target_after = _character(after, getattr(action, "target_id", -1))
    if actor is None or target_before is None or target_after is None:
        return False
    return (
        actor.team is actor_team
        and target_before.team is opposing_team(actor_team)
        and not target_before.prone
        and target_after.prone
        and _ally_can_attack_target(after, actor_team, target_after, exclude_id=actor.character_id)
    )


def _dodging_ally_avoided_attack(
    before: CombatRewardSnapshot,
    after: CombatRewardSnapshot,
    actor_team: Team,
    action: Any,
    action_result: Any | None,
) -> bool:
    target_before = _character(before, getattr(action, "target_id", -1))
    target_after = _character(after, getattr(action, "target_id", -1))
    if target_before is None or target_after is None:
        return False
    return (
        target_before.team is actor_team
        and target_before.dodging
        and target_after.hp == target_before.hp
        and _result_mentions(action_result, "miss")
    )


def _helped_attack_hit(
    before: CombatRewardSnapshot,
    after: CombatRewardSnapshot,
    actor_team: Team,
    action: Any,
) -> bool:
    actor = _character(before, getattr(action, "actor_id", -1))
    target_before = _character(before, getattr(action, "target_id", -1))
    target_after = _character(after, getattr(action, "target_id", -1))
    if actor is None or target_before is None or target_after is None:
        return False
    return (
        actor.team is actor_team
        and target_before.team is opposing_team(actor_team)
        and target_after.hp < target_before.hp
        and any(
            helper.team is actor_team
            and helper.character_id != actor.character_id
            and helper.help_against_target_id == target_before.character_id
            for helper in before.characters
        )
    )


def _successful_tactical_disengage(
    before: CombatRewardSnapshot,
    after: CombatRewardSnapshot,
    actor_team: Team,
    action: Any,
) -> bool:
    actor_before = _character(before, getattr(action, "actor_id", -1))
    actor_after = _character(after, getattr(action, "actor_id", -1))
    if actor_before is None or actor_after is None:
        return False
    return (
        actor_before.team is actor_team
        and actor_after.disengaged
        and _adjacent_enemy_count(before, actor_before) > 0
    )


def _dash_has_tactical_improvement(
    before: CombatRewardSnapshot,
    after: CombatRewardSnapshot,
    actor_team: Team,
    action: Any,
) -> bool:
    actor_before = _character(before, getattr(action, "actor_id", -1))
    actor_after = _character(after, getattr(action, "actor_id", -1))
    if actor_before is None or actor_after is None or actor_before.team is not actor_team:
        return False
    before_reach = actor_before.movement_remaining + actor_before.max_weapon_range
    after_reach = actor_after.movement_remaining + actor_after.max_weapon_range
    if after_reach <= before_reach:
        return False
    return any(
        enemy.team is opposing_team(actor_team)
        and enemy.alive
        and before_reach < _distance(actor_after.position, enemy.position) <= after_reach
        for enemy in after.characters
    )


def _hide_granted_advantage(after: CombatRewardSnapshot, action: Any) -> bool:
    actor_after = _character(after, getattr(action, "actor_id", -1))
    return actor_after is not None and actor_after.hidden


def _action_had_state_effect(
    before: CombatRewardSnapshot,
    after: CombatRewardSnapshot,
    actor_team: Team,
) -> bool:
    if before.hp_by_team != after.hp_by_team or before.alive_by_team != after.alive_by_team:
        return True

    for before_character, after_character in zip(before.characters, after.characters):
        if before_character.team is not actor_team:
            continue
        if before_character.position != after_character.position:
            return True
        if (
            before_character.hidden != after_character.hidden
            or before_character.prone != after_character.prone
            or before_character.grappled != after_character.grappled
            or before_character.dodging != after_character.dodging
            or before_character.disengaged != after_character.disengaged
            or before_character.prepared_action != after_character.prepared_action
            or before_character.help_against_target_id
            != after_character.help_against_target_id
            or before_character.grappling_target_id != after_character.grappling_target_id
            or before_character.grappled_by != after_character.grappled_by
        ):
            return True
    return False


def _character(
    snapshot: CombatRewardSnapshot,
    character_id: int,
) -> CharacterRewardSnapshot | None:
    if character_id < 0 or character_id >= len(snapshot.characters):
        return None
    return snapshot.characters[character_id]


def _is_dangerous_target(target: CharacterRewardSnapshot) -> bool:
    return target.weapon_damage_potential >= 8 or target.ac >= 15 or target.hp >= 15


def _near_vulnerable_ally(
    snapshot: CombatRewardSnapshot,
    actor_team: Team,
    target: CharacterRewardSnapshot,
) -> bool:
    return any(
        ally.team is actor_team
        and ally.alive
        and _hp_ratio(ally) <= 0.5
        and _distance(ally.position, target.position) <= 1
        for ally in snapshot.characters
    )


def _ally_can_attack_target(
    snapshot: CombatRewardSnapshot,
    actor_team: Team,
    target: CharacterRewardSnapshot,
    exclude_id: int | None = None,
) -> bool:
    return any(
        ally.team is actor_team
        and ally.alive
        and ally.character_id != exclude_id
        and ally.max_weapon_range > 0
        and _distance(ally.position, target.position) <= ally.max_weapon_range
        for ally in snapshot.characters
    )


def _adjacent_enemy_count(
    snapshot: CombatRewardSnapshot,
    actor: CharacterRewardSnapshot,
) -> int:
    return sum(
        1
        for candidate in snapshot.characters
        if candidate.team is not actor.team
        and candidate.alive
        and _distance(candidate.position, actor.position) <= 1
    )


def _hp_ratio(character: CharacterRewardSnapshot) -> float:
    if character.max_hp <= 0:
        return 0.0
    return max(0, character.hp) / character.max_hp


def _distance(first: Position, second: Position) -> int:
    return abs(first.x - second.x) + abs(first.y - second.y)


def _result_mentions(action_result: Any | None, text: str) -> bool:
    description = str(getattr(action_result, "description", "")).lower()
    return text.lower() in description


def _damage_potential(damage: int | str) -> int:
    if isinstance(damage, int):
        return max(0, damage)

    damage_text = damage.strip().lower()
    if damage_text.isdigit():
        return int(damage_text)

    match = re.fullmatch(r"(\d*)d(\d+)([+-]\d+)?", damage_text)
    if match is None:
        return 0

    dice_count = int(match.group(1) or 1)
    die_size = int(match.group(2))
    modifier = int(match.group(3) or 0)
    return max(0, dice_count * die_size + modifier)
