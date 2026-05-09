"""Turn resource tracking for combat actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class TurnResourceOwner(Protocol):
    speed: int
    action_economy: "ActionEconomy"


@dataclass
class ActionEconomy:
    """Resources available to a creature during its turn."""

    action_available: bool = True
    bonus_action_available: bool = True
    reaction_available: bool = True
    movement_remaining: int = 0

    def reset_for_turn(self, speed: int) -> None:
        self.action_available = True
        self.bonus_action_available = True
        self.reaction_available = True
        self.movement_remaining = max(0, speed)

    def spend_action(self) -> None:
        self.action_available = False

    def spend_movement(self, amount: int) -> None:
        self.movement_remaining = max(0, self.movement_remaining - amount)


def reset_turn_resources(character: TurnResourceOwner) -> None:
    character.action_economy.reset_for_turn(character.speed)
