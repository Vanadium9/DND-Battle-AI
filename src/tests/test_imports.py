from agents import BaseAgent, RandomAgent
from combat import CombatAction, CombatEnvironment, CombatState
from configs import CombatConfig, TrainingConfig
from training import Trainer


def test_project_imports() -> None:
    assert BaseAgent is not None
    assert RandomAgent is not None
    assert CombatAction is not None
    assert CombatEnvironment is not None
    assert CombatState is not None
    assert CombatConfig is not None
    assert TrainingConfig is not None
    assert Trainer is not None
