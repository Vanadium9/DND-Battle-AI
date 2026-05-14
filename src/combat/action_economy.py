"""Turn resource tracking for combat actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class TurnResourceOwner(Protocol):
    speed: int
    action_economy: "ActionEconomy"


@dataclass
class ActionEconomy:
    """Resources and transient state for a D&D-like creature turn."""

    action_available: bool = True
    bonus_action_available: bool = True
    reaction_available: bool = True
    movement_remaining: int = 0
    free_object_interaction_available: bool = True
    dodging_until_start_of_next_turn: bool = False
    disengaged_until_end_of_turn: bool = False
    hidden: bool = False
    prone: bool = False
    grappled: bool = False
    grappled_by: int | None = None
    grappling_target_id: int | None = None
    helped_target_id: int | None = None
    help_against_target_id: int | None = None
    prepared_action: str | None = None
    trigger_description: str | None = None
    reaction_used_this_round: bool = False
    advantage_on_next_check: bool = False

    def reset_for_turn(self, speed: int) -> None:
        """Reset resources available at the start of this creature's turn."""

        self.action_available = True
        self.bonus_action_available = True
        self.reset_reaction_for_round()
        self.free_object_interaction_available = True
        self.dodging_until_start_of_next_turn = False
        self.advantage_on_next_check = False
        self.prepared_action = None
        self.trigger_description = None
        self.movement_remaining = 0 if self.grappled else max(0, speed)

    def end_turn(self) -> None:
        """Clear states that expire at the end of this creature's current turn."""

        self.disengaged_until_end_of_turn = False
        self.helped_target_id = None
        self.help_against_target_id = None

    def spend_action(self) -> None:
        self.action_available = False

    def spend_bonus_action(self) -> None:
        self.bonus_action_available = False

    def spend_reaction(self) -> None:
        self.reaction_available = False
        self.reaction_used_this_round = True

    def reset_reaction_for_round(self) -> None:
        self.reaction_available = True
        self.reaction_used_this_round = False

    def spend_movement(self, amount: int) -> None:
        if self.grappled:
            self.movement_remaining = 0
            return
        self.movement_remaining = max(0, self.movement_remaining - max(0, amount))

    def spend_free_object_interaction(self) -> None:
        self.free_object_interaction_available = False

    def dash(self, speed: int) -> None:
        if not self.grappled:
            self.movement_remaining += max(0, speed)

    def stand_up(self, speed: int) -> bool:
        cost = max(1, max(0, speed) // 2)
        if not self.prone:
            return True
        if self.grappled or self.movement_remaining < cost:
            return False
        self.spend_movement(cost)
        self.prone = False
        return True

    def apply_grappled(self) -> None:
        self.grappled = True
        self.movement_remaining = 0

    def release_grappled(self) -> None:
        self.grappled = False
        self.grappled_by = None

    def reserve_ready_reaction(
        self,
        prepared_action: str,
        trigger_description: str,
    ) -> None:
        self.spend_action()
        self.prepared_action = prepared_action
        self.trigger_description = trigger_description


def reset_turn_resources(character: TurnResourceOwner) -> None:
    character.action_economy.reset_for_turn(character.speed)
