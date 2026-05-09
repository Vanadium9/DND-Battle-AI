from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents import RandomAgent
from combat import CombatEnvironment
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


if __name__ == "__main__":
    main()
