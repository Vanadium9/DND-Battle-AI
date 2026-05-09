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

__all__ = [
    "Ability",
    "ActionEconomy",
    "ActionResult",
    "AttackAction",
    "Character",
    "CombatAction",
    "CombatEnvironment",
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
    "create_test_encounter",
    "reset_turn_resources",
    "SpellAbility",
    "Stats",
    "Team",
    "WeaponAttack",
]
