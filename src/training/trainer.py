"""Trainer placeholder."""

import torch

from agents.base import BaseAgent
from combat.environment import CombatEnvironment


class Trainer:
    """Placeholder trainer for future RL algorithms."""

    device: torch.device | None = None
    agent: BaseAgent | None = None
    environment: CombatEnvironment | None = None

    pass
