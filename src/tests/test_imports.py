from agents import BaseAgent, RandomAgent
from combat import (
    ActionResult,
    Ability,
    AttackAction,
    Character,
    CombatAction,
    CombatEnvironment,
    CombatState,
    Condition,
    EndTurnAction,
    Enemy,
    GridMap,
    MoveAction,
    Position,
    SpellAbility,
    Team,
    WeaponAttack,
)
from configs import CombatConfig, TrainingConfig
from training import Trainer


def test_project_imports() -> None:
    assert BaseAgent is not None
    assert RandomAgent is not None
    assert ActionResult is not None
    assert AttackAction is not None
    assert CombatAction is not None
    assert CombatEnvironment is not None
    assert CombatState is not None
    assert EndTurnAction is not None
    assert Character is not None
    assert Enemy is not None
    assert Ability is not None
    assert MoveAction is not None
    assert WeaponAttack is not None
    assert SpellAbility is not None
    assert Condition is not None
    assert GridMap is not None
    assert Position is not None
    assert Team is not None
    assert CombatConfig is not None
    assert TrainingConfig is not None
    assert Trainer is not None
