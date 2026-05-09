from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents import RandomAgent
from combat import (
    AttackAction,
    Character,
    CombatEnvironment,
    EndTurnAction,
    GridMap,
    MoveAction,
    Position,
    Stats,
    Team,
    WeaponAttack,
)
from configs import CombatConfig, TrainingConfig
from training import Trainer


def main() -> None:
    environment = CombatEnvironment()
    agent = RandomAgent()
    trainer = Trainer()
    combat_config = CombatConfig()
    training_config = TrainingConfig()

    print("D&D tactical combat RL scaffold")
    print(f"Environment: {environment.__class__.__name__}")
    print(f"Agent: {agent.__class__.__name__}")
    print(f"Trainer: {trainer.__class__.__name__}")
    print(f"Combat config: {combat_config.__class__.__name__}")
    print(f"Training config: {training_config.__class__.__name__}")

    weapon = WeaponAttack(name="Training Sword", range=1, damage=3, attack_bonus=20)
    hero = Character(
        name="Hero",
        hp=10,
        max_hp=10,
        ac=14,
        position=Position(0, 0),
        speed=2,
        stats=Stats(),
        team=Team.PLAYERS,
        abilities=[weapon],
    )
    target = Character(
        name="Target",
        hp=10,
        max_hp=10,
        ac=12,
        position=Position(1, 1),
        speed=2,
        stats=Stats(),
        team=Team.ENEMIES,
    )
    combat_environment = CombatEnvironment(
        characters=[hero, target],
        grid_map=GridMap(width=4, height=4),
        log_to_console=True,
    )

    print("\nAction log:")
    for action in (
        MoveAction(actor_id=0, destination=Position(0, 1)),
        AttackAction(actor_id=0, target_id=1, weapon=weapon),
        EndTurnAction(actor_id=0),
    ):
        combat_environment.step(action)


if __name__ == "__main__":
    main()
