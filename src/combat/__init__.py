"""Combat simulation primitives."""

from combat.action_economy import ActionEconomy, reset_turn_resources
from combat.actions import (
    ActionResult,
    AttackAction,
    CombatAction,
    EndTurnAction,
    MoveAction,
)
from combat.environment import CombatEnvironment
from combat.encounter_generator import EncounterGenerator
from combat.map import GridMap
from combat.models import (
    Ability,
    Character,
    CombatState,
    Condition,
    Enemy,
    Position,
    SpellAbility,
    Stats,
    Team,
    WeaponAttack,
)
from combat.presets import (
    FighterArcher,
    FighterChampionGreatsword,
    Goblin,
    Orc,
    create_test_encounter,
)
from combat.rewards import (
    CombatReward,
    CombatRewardSnapshot,
    RewardConfig,
    calculate_combat_reward,
    opposing_team,
    snapshot_combat_state,
)

__all__ = [
    "Ability",
    "ActionEconomy",
    "ActionResult",
    "AttackAction",
    "Character",
    "CombatAction",
    "CombatEnvironment",
    "CombatReward",
    "CombatRewardSnapshot",
    "CombatState",
    "Condition",
    "Enemy",
    "EndTurnAction",
    "EncounterGenerator",
    "FighterArcher",
    "FighterChampionGreatsword",
    "Goblin",
    "GridMap",
    "MoveAction",
    "Orc",
    "Position",
    "RewardConfig",
    "calculate_combat_reward",
    "create_test_encounter",
    "opposing_team",
    "reset_turn_resources",
    "SpellAbility",
    "snapshot_combat_state",
    "Stats",
    "Team",
    "WeaponAttack",
]
