"""Initiative rolling and turn-order helpers."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence

from combat.checks import InitiativeCheckResult, roll_initiative_check
from combat.models import Character, CombatState


@dataclass(frozen=True)
class InitiativeRoll:
    """Initiative roll plus deterministic tie-break data."""

    character_id: int
    character_name: str
    roll: int
    dex_modifier: int
    total: int
    tie_breaker: float

    @classmethod
    def from_check(
        cls,
        character_id: int,
        check: InitiativeCheckResult,
        tie_breaker: float,
    ) -> "InitiativeRoll":
        return cls(
            character_id=character_id,
            character_name=check.character_name,
            roll=check.roll,
            dex_modifier=check.dex_modifier,
            total=check.total,
            tie_breaker=tie_breaker,
        )

    @property
    def log(self) -> str:
        return (
            f"Initiative roll: {self.character_name} "
            f"d20={self.roll}, dex_mod={self.dex_modifier}, "
            f"total={self.total}, tie_breaker={self.tie_breaker:.6f}."
        )


@dataclass(frozen=True)
class InitiativeResult:
    """Full initiative result for one combat start."""

    rolls: tuple[InitiativeRoll, ...]
    order: tuple[int, ...]

    @property
    def ordered_rolls(self) -> tuple[InitiativeRoll, ...]:
        by_id = {roll.character_id: roll for roll in self.rolls}
        return tuple(by_id[character_id] for character_id in self.order)

    @property
    def order_log(self) -> str:
        entries = [
            f"{roll.character_name}({roll.total})"
            for roll in self.ordered_rolls
        ]
        return f"Initiative order: {', '.join(entries)}."


def roll_initiative_order(
    characters: Sequence[Character],
    seed: int | None = None,
    rng: random.Random | None = None,
) -> InitiativeResult:
    """Roll initiative and return character ids sorted by turn order."""

    random_source = rng or random.Random(seed)
    rolls = tuple(
        InitiativeRoll.from_check(
            character_id=character_id,
            check=roll_initiative_check(character, random_source),
            tie_breaker=random_source.random(),
        )
        for character_id, character in enumerate(characters)
    )
    ordered_rolls = sorted(
        rolls,
        key=lambda roll: (roll.total, roll.dex_modifier, roll.tie_breaker),
        reverse=True,
    )
    return InitiativeResult(
        rolls=rolls,
        order=tuple(roll.character_id for roll in ordered_rolls),
    )


def apply_initiative_result(
    combat_state: CombatState,
    initiative: InitiativeResult,
) -> None:
    """Store initiative metadata on a combat state."""

    combat_state.initiative_order = list(initiative.order)
    combat_state.current_turn_index = 0
    combat_state.turn_index = initiative.order[0] if initiative.order else 0
    combat_state.round_number = 1
    combat_state.initiative_rolls = {
        roll.character_id: roll.roll
        for roll in initiative.rolls
    }
    combat_state.initiative_totals = {
        roll.character_id: roll.total
        for roll in initiative.rolls
    }
    combat_state.initiative_dex_modifiers = {
        roll.character_id: roll.dex_modifier
        for roll in initiative.rolls
    }
    combat_state.initiative_tie_breakers = {
        roll.character_id: roll.tie_breaker
        for roll in initiative.rolls
    }


def apply_fixed_turn_order(
    combat_state: CombatState,
    order: Sequence[int] | None = None,
) -> None:
    """Store an explicit non-rolled turn order on a combat state."""

    combat_state.initiative_order = list(order or range(len(combat_state.characters)))
    combat_state.current_turn_index = 0
    combat_state.turn_index = (
        combat_state.initiative_order[0]
        if combat_state.initiative_order
        else 0
    )
    combat_state.round_number = 1
    combat_state.initiative_rolls = {}
    combat_state.initiative_totals = {}
    combat_state.initiative_dex_modifiers = {}
    combat_state.initiative_tie_breakers = {}
